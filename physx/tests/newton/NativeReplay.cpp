#include "NewtonSolver.h"
#include "NewtonPatchProjection.h"
#include <Eigen/QR>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

static bool fixedBounds(int mode)
{
	return mode == 3 || mode == 4 || mode == 6;
}

static void initializeParameter(const newton::Problem& problem, const std::vector<newton::Patch>& groups,
	newton::ConstVector originalFree, const newton::Result* previous, newton::MutableVector parameter, int mode, int seed)
{
	parameter.setZero();
	if(!seed || !previous || previous->primal.size() != 6 * problem.bodyCount())
		return;
	const newton::Vector velocity = problem.jacobian * previous->primal + originalFree;
	for(size_t i = 0; i < groups.size(); ++i)
	{
		const newton::Patch& patch = groups[i];
		double totalNormal = 0.0;
		for(int n = 0; n < patch.normalCount; ++n)
		{
			const newton::CompactContact& contact = problem.contacts[patch.firstContact + n];
			const double cap = contact.hasScalarBounds() ? problem.bounds(contact).upper : std::numeric_limits<double>::infinity();
			totalNormal += std::min(std::max(-velocity[contact.row] / problem.regularization[contact.row], 0.0), cap);
		}
		if(fixedBounds(mode))
		{
			parameter[i] = totalNormal;
			continue;
		}
		if(seed == 1)
		{
			for(int t = 0; t < patch.tangentCount; ++t)
				parameter[i] += patch.friction * std::abs(velocity[problem.contacts[patch.firstContact + patch.normalCount + t].row]);
		}
		else
		{
			// Evaluate the existing cone projection at the velocity seed, then use
			// its complete compliant tangent residual, including R*lambda.
			newton::Vector normalVelocity(patch.normalCount), normalR(patch.normalCount), caps(patch.normalCount), normalImpulse(patch.normalCount);
			double tangentVelocity[4], tangentR[4], friction[4], tangentImpulse[4];
			for(int n = 0; n < patch.normalCount; ++n)
			{
				const newton::CompactContact& contact = problem.contacts[patch.firstContact + n];
				normalVelocity[n] = velocity[contact.row];
				normalR[n] = problem.regularization[contact.row];
				caps[n] = contact.hasScalarBounds() ? problem.bounds(contact).upper : std::numeric_limits<double>::infinity();
			}
			for(int t = 0; t < patch.tangentCount; ++t)
			{
				const int row = problem.contacts[patch.firstContact + patch.normalCount + t].row;
				tangentVelocity[t] = velocity[row];
				tangentR[t] = problem.regularization[row];
				friction[t] = patch.friction;
			}
			newton::PatchProjectionInput input = {patch.normalCount, patch.tangentCount, normalVelocity.data(), normalR.data(), caps.data(), tangentVelocity, tangentR, friction};
			newton::PatchProjectionOutput output = {normalImpulse.data(), tangentImpulse, NULL, NULL, NULL};
			newton::PatchProjectionResult projected;
			if(newton::projectPatch(input, output, projected))
				for(int t = 0; t < patch.tangentCount; ++t)
					parameter[i] += patch.friction * std::abs(tangentVelocity[t] + tangentR[t] * tangentImpulse[t]);
		}
	}
}

// Type-II Anderson mixing of the outer fixed point. This diagnostic keeps a
// small secant history only; core equations, stopping and iteration budget stay
// unchanged. Failed extrapolations restart with a half-relaxed Picard step.
struct AndersonHistory
{
	Eigen::MatrixXd points, residuals;
	int used;
	explicit AndersonHistory(int size, int depth) : points(size, depth + 1), residuals(size, depth + 1), used(0) {}
	void reset() { used = 0; }
	bool next(newton::ConstVector point, newton::ConstVector target, double beta, newton::MutableVector proposal)
	{
		if(used == points.cols())
		{
			for(int i = 1; i < used; ++i)
			{
				points.col(i - 1) = points.col(i);
				residuals.col(i - 1) = residuals.col(i);
			}
			--used;
		}
		points.col(used) = point;
		residuals.col(used) = target - point;
		++used;
		proposal = point + beta * (target - point);
		if(used < 2)
			return false;
		Eigen::MatrixXd dx(point.size(), used - 1), df(point.size(), used - 1);
		for(int i = 0; i < used - 1; ++i)
		{
			dx.col(i) = points.col(i + 1) - points.col(i);
			df.col(i) = residuals.col(i + 1) - residuals.col(i);
		}
		Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(df);
		qr.setThreshold(1.0e-10);
		const newton::Vector weights = qr.solve(target - point);
		const newton::Vector accelerated = proposal - (dx + beta * df) * weights;
		if(!accelerated.allFinite() || (accelerated - point).norm() > 10.0 * (target - point).norm())
		{
			reset();
			return false;
		}
		proposal = accelerated.cwiseMax(0.0);
		return true;
	}
};

static void applyOuterParameter(newton::Problem& problem, const std::vector<newton::Patch>& groups,
	newton::ConstVector originalFree, newton::ConstVector parameter, int mode)
{
	for(size_t i = 0; i < groups.size(); ++i)
	{
		const newton::Patch& patch = groups[i];
		if(fixedBounds(mode))
		{
			const double limit = patch.friction * parameter[i];
			for(int tangent = 0; tangent < patch.tangentCount; ++tangent)
				problem.setScalarBounds(patch.firstContact + patch.normalCount + tangent, -limit, limit);
		}
		else
			for(int normal = 0; normal < patch.normalCount; ++normal)
			{
				const int row = problem.contacts[patch.firstContact + normal].row;
				problem.freeVelocity[row] = originalFree[row] + parameter[i];
			}
	}
}

static double evaluatePhysical(const newton::Problem& problem, const std::vector<newton::Patch>& groups,
	newton::ConstVector originalFree, const newton::Result& result, newton::MutableVector parameter,
	newton::MutableVector velocity, int mode, double& normalResidual, double& tangentResidual)
{
	velocity.noalias() = problem.jacobian * result.primal;
	velocity += originalFree;
	velocity += problem.regularization.cwiseProduct(result.impulse);
	double normalSquare = 0.0, tangentSquare = 0.0;
	int normalRows = 0, tangentRows = 0;
	for(size_t i = 0; i < groups.size(); ++i)
	{
		const newton::Patch& patch = groups[i];
		double totalNormal = 0.0, correction = 0.0;
		for(int normal = 0; normal < patch.normalCount; ++normal)
			totalNormal += result.impulse[problem.contacts[patch.firstContact + normal].row];
		const double limit = patch.friction * totalNormal;
		for(int offset = 0; offset < patch.normalCount + patch.tangentCount; ++offset)
		{
			const newton::CompactContact& contact = problem.contacts[patch.firstContact + offset];
			const int row = contact.row;
			double response = problem.regularization[row];
			for(int end = 0; end < 2; ++end)
				if(contact.body[end] >= 0)
					response += contact.jacobian[end].squaredNorm();
			const bool normal = offset < patch.normalCount;
			const double lower = normal ? 0.0 : -limit;
			const double upper = normal ? (contact.hasScalarBounds() ? problem.bounds(contact).upper : std::numeric_limits<double>::infinity()) : limit;
			const double residual = std::min(std::max(velocity[row], response * (result.impulse[row] - upper)), response * (result.impulse[row] - lower));
			if(normal)
			{
				normalSquare += residual * residual;
				++normalRows;
			}
			else
			{
				tangentSquare += residual * residual;
				++tangentRows;
				correction += patch.friction * std::abs(velocity[row]);
			}
		}
		parameter[i] = fixedBounds(mode) ? totalNormal : correction;
	}
	normalResidual = normalRows ? std::sqrt(normalSquare / normalRows) / problem.timestep : 0.0;
	tangentResidual = tangentRows ? std::sqrt(tangentSquare / tangentRows) / problem.timestep : 0.0;
	return std::max(normalResidual, tangentResidual);
}

static newton::SolveStatus::Enum solveOuter(newton::Problem& problem, const std::vector<newton::Patch>& groups,
	newton::ConstVector originalFree, const newton::Settings& settings, newton::Result& result,
	newton::Workspace& workspace, const newton::Result* previous, int mode, int seed, int depth, bool continuation)
{
	if(fixedBounds(mode))
	{
		// A different convex inner problem: freeze tangent limits at a normal
		// force estimate. The initial zero estimate solves frictionless support.
		problem.patches.clear();
		if(newton::prepareProblem(problem) != newton::SolveStatus::eSUCCESS)
			return newton::SolveStatus::eINVALID_INPUT;
	}
	newton::VectorStorage parameter(groups.size()), target(groups.size()), bestParameter(groups.size()), velocity(problem.rowCount());
	initializeParameter(problem, groups, originalFree, previous, parameter, mode, seed);
	newton::VectorStorage bestTarget(groups.size()), nextParameter(groups.size());
	AndersonHistory history(int(groups.size()), depth);
	bool wasAccelerated = false;
	int acceleratedSteps = 0, rejectedSteps = 0;
	newton::Result best;
	bool haveBest = false, converged = false;
	double bestMerit = std::numeric_limits<double>::infinity();
	int remaining = settings.iterations, attempts = 0, backtracks = 0, iterations = 0, factors = 0, updates = 0, lines = 0;
	newton::SolverStatistics total;
	const newton::Clock::time_point start = newton::Clock::now();
	while(remaining > 0)
	{
		applyOuterParameter(problem, groups, originalFree, parameter, mode);
		newton::Settings inner = settings;
		inner.iterations = remaining;
		const newton::SolveStatus::Enum status = continuation && attempts ?
			newton::continueNewton(problem, inner, result, workspace, previous) :
			newton::solveNewton(problem, inner, result, workspace, previous);
		++attempts;
		remaining -= std::max(1, result.iterations);
		iterations += result.iterations;
		factors += result.factorizations;
		updates += result.rankUpdates;
		lines += result.lineSearchEvaluations;
		total.matrixMs += result.matrixMs;
		total.factorMs += result.factorMs;
		total.updateMs += result.updateMs;
		total.evaluationMs += result.evaluationMs;
		total.lineSearchMs += result.lineSearchMs;
		total.backsolveMs += result.backsolveMs;
		total.symbolicMs += result.symbolicMs;
		total.symbolicAnalyses += result.symbolicAnalyses;
		total.reusedFactors += result.reusedFactors;
		total.factorFallbacks += result.factorFallbacks;
		total.factorError = std::max(total.factorError, result.factorError);
		if(status != newton::SolveStatus::eSUCCESS && status != newton::SolveStatus::eITERATION_LIMIT)
			return status;
		double normal, tangent;
		const double residual = evaluatePhysical(problem, groups, originalFree, result, target, velocity, mode, normal, tangent);
		const double merit = mode == 1 ? std::max(residual, result.scaledGradient) : residual;
		const bool candidate = mode == 1 || status == newton::SolveStatus::eSUCCESS || !haveBest;
		const bool improved = candidate && (!haveBest || merit <= bestMerit);
		const bool acceptable = status == newton::SolveStatus::eSUCCESS && residual <= settings.tolerance;
		std::printf("OUTER,attempt=%d,spent=%d,inner=%d,status=%d,stop=%d,gradient=%.12g,normal=%.12g,tangent=%.12g,parameter_change=%.12g,parameter_max=%.12g,accepted=%d\n",
			attempts, settings.iterations - remaining, result.iterations, int(status), result.stopReason,
			result.scaledGradient, normal, tangent, (target - parameter).lpNorm<Eigen::Infinity>(), parameter.lpNorm<Eigen::Infinity>(), int(improved || acceptable));
		if(improved || acceptable)
		{
			best = result;
			bestParameter = parameter;
			bestTarget = target;
			bestMerit = merit;
			haveBest = true;
		}
		if(acceptable)
		{
			converged = true;
			break;
		}
		if(remaining <= 0)
			break;
		bool relaxedStagnation = false;
		if(mode >= 5)
		{
			if(wasAccelerated && merit > 2.0 * bestMerit)
			{
				++rejectedSteps;
				history.reset();
				nextParameter = 0.5 * (bestParameter + bestTarget);
				wasAccelerated = false;
			}
			else
			{
				wasAccelerated = history.next(parameter, target, fixedBounds(mode) ? 0.5 : 1.0, nextParameter);
				acceleratedSteps += int(wasAccelerated);
			}
			relaxedStagnation = (nextParameter.array() == parameter.array()).all();
			parameter = nextParameter;
		}
		else if(mode == 4)
		{
			// Standard half relaxation of the bound fixed point; retain the best
			// physical iterate, but do not require every intermediate step to improve.
			relaxedStagnation = ((0.5 * (parameter + target)).array() == parameter.array()).all();
			parameter = 0.5 * (parameter + target);
		}
		else if(improved)
			parameter = target;
		else
		{
			++backtracks;
			parameter = 0.5 * (parameter + bestParameter);
		}
		bool same = true;
		for(size_t i = 0; same && i < groups.size(); ++i)
			if(parameter[i] != bestParameter[i])
			{
				if(fixedBounds(mode))
					same = groups[i].friction * parameter[i] == groups[i].friction * bestParameter[i];
				else
					for(int normalRow = 0; same && normalRow < groups[i].normalCount; ++normalRow)
					{
						const int row = problem.contacts[groups[i].firstContact + normalRow].row;
						same = originalFree[row] + parameter[i] == originalFree[row] + bestParameter[i];
					}
			}
		if((mode >= 4 && relaxedStagnation) || (mode < 4 && same))
			break;
		previous = mode >= 4 ? &result : &best;
	}
	result = best;
	applyOuterParameter(problem, groups, originalFree, bestParameter, mode);
	result.iterations = iterations;
	result.factorizations = factors;
	result.rankUpdates = updates;
	result.lineSearchEvaluations = lines;
	result.matrixMs = total.matrixMs;
	result.factorMs = total.factorMs;
	result.updateMs = total.updateMs;
	result.evaluationMs = total.evaluationMs;
	result.lineSearchMs = total.lineSearchMs;
	result.backsolveMs = total.backsolveMs;
	result.symbolicMs = total.symbolicMs;
	result.symbolicAnalyses = total.symbolicAnalyses;
	result.reusedFactors = total.reusedFactors;
	result.factorFallbacks = total.factorFallbacks;
	result.factorError = total.factorError;
	result.elapsedMs = newton::elapsed(start);
	result.status = converged ? newton::SolveStatus::eSUCCESS : newton::SolveStatus::eITERATION_LIMIT;
	std::printf("OUTER_TOTAL,attempts=%d,backtracks=%d,budget=%d,converged=%d,accelerated=%d,rejected=%d,best_physical=%.12g\n", attempts, backtracks, settings.iterations - remaining, int(converged), acceleratedSteps, rejectedSteps, bestMerit);
	return result.status;
}

int main(int argc, char** argv)
{
	if(argc < 2)
	{
		std::fprintf(stderr, "Usage: NewtonNativeReplay capture [iterations] [checkFactor] [cold] [solution] [mode:inner|desaxce|physical|bounds|relaxed|anderson|andersonbounds] [seed:zero|raw|projected] [depth] [continuation=0]\n");
		return 1;
	}
	std::ifstream input(argv[1]);
	std::string magic;
	int version, bodies, rows, patches;
	newton::Problem problem;
	newton::Settings settings;
	input >> magic >> version >> bodies >> rows >> patches >> problem.timestep >> settings.tolerance >> settings.iterations;
	if(!input || magic != "NEWTON_NATIVE_CAPTURE" || version != 1 || bodies < 1 || rows < 0 || patches < 0)
		return 1;
	problem.inverseMass.resize(bodies);
	problem.massDiagonal.resize(6 * bodies);
	for(int i = 0; i < bodies; ++i)
		input >> problem.inverseMass[i];
	for(int i = 0; i < 6 * bodies; ++i)
		input >> problem.massDiagonal[i];
	newton::Vector originalFree(rows);
	for(int i = 0; i < rows; ++i)
	{
		newton::CompactContact contact;
		int lowerFinite, upperFinite;
		double lower, upper;
		input >> contact.body[0] >> contact.body[1] >> originalFree[i] >> contact.freeVelocity >> contact.regularization
			>> lowerFinite >> lower >> upperFinite >> upper;
		for(int end = 0; end < 2; ++end)
			for(int axis = 0; axis < 6; ++axis)
				input >> contact.jacobian[end][axis];
		problem.addScalarContact(contact, lowerFinite ? lower : -std::numeric_limits<double>::infinity(),
			upperFinite ? upper : std::numeric_limits<double>::infinity());
	}
	for(int i = 0; i < patches; ++i)
	{
		int first, normalCount, tangentCount;
		double friction;
		input >> first >> normalCount >> tangentCount >> friction;
		problem.addPatch(first, normalCount, tangentCount, friction);
	}
	int hasSeed;
	input >> hasSeed;
	newton::Result previous;
	if(hasSeed)
	{
		previous.primal.resize(6 * bodies);
		for(int i = 0; i < 6 * bodies; ++i)
			input >> previous.primal[i];
	}
	if(!input || newton::prepareProblem(problem) != newton::SolveStatus::eSUCCESS)
	{
		std::fprintf(stderr, "Invalid native equation capture\n");
		return 1;
	}
	if(argc > 2)
		settings.iterations = std::atoi(argv[2]);
	settings.checkFactor = argc > 3 ? std::atoi(argv[3]) != 0 : true;
	settings.profile = true;
	const bool warm = hasSeed && !(argc > 4 && std::atoi(argv[4]));
	std::printf("PROBLEM,bodies=%d,rows=%d,patches=%d,min_R=%.12g,max_R=%.12g,tolerance=%.12g,cap=%d,warm=%d\n",
		bodies, rows, patches, rows ? problem.regularization.minCoeff() : 0.0,
		rows ? problem.regularization.maxCoeff() : 0.0, settings.tolerance, settings.iterations, int(warm));
	const std::vector<newton::Patch> groups = problem.patches;
	const std::string modeName = argc > 6 ? argv[6] : "inner";
	const int mode = modeName == "desaxce" ? 1 : (modeName == "physical" ? 2 : (modeName == "bounds" ? 3 : (modeName == "relaxed" ? 4 : (modeName == "anderson" ? 5 : (modeName == "andersonbounds" ? 6 : 0)))));
	const std::string seedName = argc > 7 ? argv[7] : "zero";
	const int seed = seedName == "raw" ? 1 : (seedName == "projected" ? 2 : 0);
	const int depth = argc > 8 ? std::max(1, std::min(10, std::atoi(argv[8]))) : 5;
	const bool continuation = argc > 9 && std::atoi(argv[9]);
	newton::Workspace workspace;
	newton::Result result;
	const newton::SolveStatus::Enum status = mode ? solveOuter(problem, groups, originalFree, settings, result, workspace, warm ? &previous : NULL, mode, seed, depth, continuation) :
		newton::solveNewton(problem, settings, result, workspace, warm ? &previous : NULL);
	std::printf("RESULT,status=%d,iterations=%d,factors=%d,updates=%d,reused=%d,fallbacks=%d,line_evaluations=%d,stop=%d,gradient=%.12g,scaled_gradient=%.12g,factor_error=%.12g,ms=%.9g,matrix_ms=%.9g,factor_ms=%.9g,update_ms=%.9g,evaluation_ms=%.9g,line_ms=%.9g\n",
		int(status), result.iterations, result.factorizations, result.rankUpdates, result.reusedFactors, result.factorFallbacks, result.lineSearchEvaluations,
		result.stopReason, result.gradientResidual, result.scaledGradient, result.factorError, result.elapsedMs,
		result.matrixMs, result.factorMs, result.updateMs, result.evaluationMs, result.lineSearchMs);
	if(status != newton::SolveStatus::eSUCCESS && status != newton::SolveStatus::eITERATION_LIMIT)
		return 2;
	const newton::Vector velocity = problem.jacobian * result.primal + originalFree + problem.regularization.cwiseProduct(result.impulse);
	double normalSquare = 0.0, maxCorrection = 0.0;
	int normalRows = 0;
	for(int i = 0; i < patches; ++i)
	{
		const newton::Patch& patch = groups[i];
		for(int normal = 0; normal < patch.normalCount; ++normal)
		{
			const newton::CompactContact& contact = problem.contacts[patch.firstContact + normal];
			double response = problem.regularization[contact.row];
			for(int end = 0; end < 2; ++end)
				if(contact.body[end] >= 0)
					response += contact.jacobian[end].squaredNorm();
			const double upper = contact.hasScalarBounds() ? problem.bounds(contact).upper : std::numeric_limits<double>::infinity();
			const double residual = std::min(std::max(velocity[contact.row], response * (result.impulse[contact.row] - upper)),
				response * result.impulse[contact.row]);
			normalSquare += residual * residual;
			++normalRows;
		}
		double correction = 0.0;
		for(int tangent = 0; tangent < patch.tangentCount; ++tangent)
			correction += patch.friction * std::abs(velocity[problem.contacts[patch.firstContact + patch.normalCount + tangent].row]);
		maxCorrection = std::max(maxCorrection, correction);
	}
	std::printf("RESIDUAL,associated=%.12g,physical_normal=%.12g,max_desaxce_shift=%.12g\n", newton::computeResidual(problem, result.impulse),
		normalRows ? std::sqrt(normalSquare / normalRows) / problem.timestep : 0.0, maxCorrection);
	if(argc > 5 && std::string(argv[5]) != "-")
	{
		std::ofstream output(argv[5]);
		output << std::setprecision(17);
		for(int i = 0; i < result.impulse.size(); ++i)
			output << result.impulse[i] << '\n';
		for(int i = 0; i < result.primal.size(); ++i)
			output << result.primal[i] << '\n';
	}
	return 0;
}
