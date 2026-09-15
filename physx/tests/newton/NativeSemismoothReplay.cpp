// Isolated native-equation experiment. No production solve implementation uses this file.
#include "NewtonSolver.h"
#include <Eigen/SparseLU>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <string>

typedef Eigen::Matrix<double, 12, 1> LocalVector;
typedef Eigen::Matrix<double, 12, 12> LocalMatrix;
typedef Eigen::Triplet<double> Entry;

struct Evaluation
{
	newton::Vector impulse, velocity, gradient, diagonal;
	std::vector<int> state;
	double merit, scaledGradient, physicalResidual;
	explicit Evaluation(int rows, int dofs) : impulse(rows), velocity(rows), gradient(dofs), diagonal(rows), state(rows),
		merit(0.0), scaledGradient(0.0), physicalResidual(0.0) {}
};

static bool loadCapture(const char* filename, newton::Problem& problem, newton::Settings& settings, newton::Vector& seed)
{
	std::ifstream input(filename);
	std::string magic;
	int version, bodies, rows, patches;
	input >> magic >> version >> bodies >> rows >> patches >> problem.timestep >> settings.tolerance >> settings.iterations;
	if(!input || magic != "NEWTON_NATIVE_CAPTURE" || version != 1 || bodies < 1 || rows < 0 || patches < 0)
		return false;
	problem.inverseMass.resize(bodies);
	problem.massDiagonal.resize(6 * bodies);
	for(int i = 0; i < bodies; ++i) input >> problem.inverseMass[i];
	for(int i = 0; i < 6 * bodies; ++i) input >> problem.massDiagonal[i];
	for(int i = 0; i < rows; ++i)
	{
		newton::CompactContact contact;
		int lowerFinite, upperFinite;
		double lower, upper, preparedFree;
		input >> contact.body[0] >> contact.body[1] >> contact.freeVelocity >> preparedFree >> contact.regularization
			>> lowerFinite >> lower >> upperFinite >> upper;
		for(int end = 0; end < 2; ++end)
			for(int axis = 0; axis < 6; ++axis) input >> contact.jacobian[end][axis];
		problem.addScalarContact(contact, lowerFinite ? lower : -std::numeric_limits<double>::infinity(),
			upperFinite ? upper : std::numeric_limits<double>::infinity());
	}
	for(int i = 0; i < patches; ++i)
	{
		int first, normals, tangents;
		double friction;
		input >> first >> normals >> tangents >> friction;
		problem.addPatch(first, normals, tangents, friction);
	}
	int hasSeed;
	input >> hasSeed;
	seed.setZero(6 * bodies);
	if(hasSeed)
		for(int i = 0; i < seed.size(); ++i) input >> seed[i];
	return bool(input) && newton::prepareProblem(problem) == newton::SolveStatus::eSUCCESS;
}

static void projectScalar(double velocity, double regularization, double lower, double upper,
	double& impulse, double& diagonal, int& state)
{
	const double unconstrained = -velocity / regularization;
	if(unconstrained <= lower)
	{
		impulse = lower;
		diagonal = 0.0;
		state = -1;
	}
	else if(unconstrained >= upper)
	{
		impulse = upper;
		diagonal = 0.0;
		state = 1;
	}
	else
	{
		impulse = unconstrained;
		diagonal = 1.0 / regularization;
		state = 0;
	}
}

static bool evaluate(const newton::Problem& problem, newton::ConstVector primal, Evaluation& value)
{
	value.velocity.noalias() = problem.jacobian * primal;
	value.velocity += problem.freeVelocity;
	// Scalar rows (including joint bounds) use their existing physical clamp.
	for(size_t i = 0; i < problem.contacts.size(); ++i)
	{
		const newton::CompactContact& contact = problem.contacts[i];
		const double lower = contact.hasScalarBounds() ? problem.bounds(contact).lower : 0.0;
		const double upper = contact.hasScalarBounds() ? problem.bounds(contact).upper : std::numeric_limits<double>::infinity();
		projectScalar(value.velocity[contact.row], problem.regularization[contact.row], lower, upper,
			value.impulse[contact.row], value.diagonal[contact.row], value.state[contact.row]);
	}
	for(size_t i = 0; i < problem.patches.size(); ++i)
	{
		const newton::Patch& patch = problem.patches[i];
		double totalNormal = 0.0;
		for(int n = 0; n < patch.normalCount; ++n)
			totalNormal += value.impulse[problem.contacts[patch.firstContact + n].row];
		const double limit = patch.friction * totalNormal;
		for(int t = 0; t < patch.tangentCount; ++t)
		{
			const int row = problem.contacts[patch.firstContact + patch.normalCount + t].row;
			projectScalar(value.velocity[row], problem.regularization[row], -limit, limit,
				value.impulse[row], value.diagonal[row], value.state[row]);
		}
	}
	value.gradient = primal;
	value.gradient.noalias() -= problem.jacobian.transpose() * value.impulse;
	const double scale = problem.massDiagonal.sum() * problem.timestep;
	value.merit = 0.5 * value.gradient.array().square().matrix().dot(problem.massDiagonal) / (scale * scale);
	value.scaledGradient = std::sqrt(2.0 * value.merit);
	// Check physical row conditions independently of the body momentum equation.
	// They should be near roundoff because lambda(v) enforces them explicitly.
	double normalSquare = 0.0, tangentSquare = 0.0;
	int normalRows = 0, tangentRows = 0;
	for(size_t i = 0; i < problem.patches.size(); ++i)
	{
		const newton::Patch& patch = problem.patches[i];
		double totalNormal = 0.0;
		for(int n = 0; n < patch.normalCount; ++n)
			totalNormal += value.impulse[problem.contacts[patch.firstContact + n].row];
		const double limit = patch.friction * totalNormal;
		for(int offset = 0; offset < patch.normalCount + patch.tangentCount; ++offset)
		{
			const newton::CompactContact& contact = problem.contacts[patch.firstContact + offset];
			const int row = contact.row;
			const bool normal = offset < patch.normalCount;
			const double lower = normal ? 0.0 : -limit;
			const double upper = normal ? (contact.hasScalarBounds() ? problem.bounds(contact).upper : std::numeric_limits<double>::infinity()) : limit;
			double response = problem.regularization[row];
			for(int end = 0; end < 2; ++end)
				if(contact.body[end] >= 0) response += contact.jacobian[end].squaredNorm();
			const double w = value.velocity[row] + problem.regularization[row] * value.impulse[row];
			const double r = std::min(std::max(w, response * (value.impulse[row] - upper)), response * (value.impulse[row] - lower));
			if(normal) { normalSquare += r * r; ++normalRows; }
			else { tangentSquare += r * r; ++tangentRows; }
		}
	}
	value.physicalResidual = std::max(normalRows ? std::sqrt(normalSquare / normalRows) / problem.timestep : 0.0,
		tangentRows ? std::sqrt(tangentSquare / tangentRows) / problem.timestep : 0.0);
	return value.gradient.allFinite() && value.impulse.allFinite() && std::isfinite(value.merit) && std::isfinite(value.physicalResidual);
}

static LocalVector localJacobian(const newton::CompactContact& contact)
{
	LocalVector result;
	for(int end = 0; end < 2; ++end)
		if(contact.body[end] >= 0) result.segment<6>(6 * end) = contact.jacobian[end];
		else result.segment<6>(6 * end).setZero();
	return result;
}

static void appendBlock(const newton::CompactContact& contact, const LocalMatrix& block, std::vector<Entry>& entries)
{
	for(int end0 = 0; end0 < 2; ++end0)
		if(contact.body[end0] >= 0)
			for(int end1 = 0; end1 < 2; ++end1)
				if(contact.body[end1] >= 0)
					for(int row = 0; row < 6; ++row)
						for(int column = 0; column < 6; ++column)
							// Store zeros too: a fixed problem retains its full symbolic pattern.
							entries.push_back(Entry(6 * contact.body[end0] + row, 6 * contact.body[end1] + column,
								block(6 * end0 + row, 6 * end1 + column)));
}

static void buildJacobian(const newton::Problem& problem, const Evaluation& value, newton::Sparse& matrix, std::vector<Entry>& entries)
{
	entries.clear();
	for(int i = 0; i < problem.bodyCount() * 6; ++i) entries.push_back(Entry(i, i, 1.0));
	size_t patchIndex = 0;
	for(size_t i = 0; i < problem.contacts.size();)
	{
		LocalMatrix block = LocalMatrix::Zero();
		if(patchIndex < problem.patches.size() && problem.patches[patchIndex].firstContact == int(i))
		{
			const newton::Patch& patch = problem.patches[patchIndex++];
			LocalVector normal = LocalVector::Zero(), tangent = LocalVector::Zero();
			for(int offset = 0; offset < patch.normalCount + patch.tangentCount; ++offset)
			{
				const newton::CompactContact& contact = problem.contacts[i + offset];
				const LocalVector jacobian = localJacobian(contact);
				block.noalias() += value.diagonal[contact.row] * jacobian * jacobian.transpose();
				if(offset < patch.normalCount)
					normal += value.diagonal[contact.row] * jacobian;
				else if(value.state[contact.row])
					tangent += (patch.friction * value.state[contact.row]) * jacobian;
			}
			// dN/dv = -sum_free(Jn/Rn). Saturated lambda_t = sign*mu*N,
			// hence -Jt' d(lambda_t)/dv = (sum_bound sign*mu*Jt)' sum_free(Jn/Rn).
			// This rank-one term is generally nonsymmetric.
			block.noalias() += tangent * normal.transpose();
			appendBlock(problem.contacts[i], block, entries);
			i += patch.normalCount + patch.tangentCount;
		}
		else
		{
			const newton::CompactContact& contact = problem.contacts[i++];
			const LocalVector jacobian = localJacobian(contact);
			block.noalias() = value.diagonal[contact.row] * jacobian * jacobian.transpose();
			appendBlock(contact, block, entries);
		}
	}
	matrix.setFromTriplets(entries.begin(), entries.end());
	matrix.makeCompressed();
}

static bool derivativeAudit(const newton::Problem& problem, newton::ConstVector seed)
{
	const int dofs = problem.bodyCount() * 6;
	std::mt19937 random(91241);
	std::uniform_real_distribution<double> sample(-1.0, 1.0);
	Evaluation value(problem.rowCount(), dofs), plus(problem.rowCount(), dofs), minus(problem.rowCount(), dofs);
	newton::Sparse matrix(dofs, dofs);
	std::vector<Entry> entries;
	int checked = 0, skipped = 0;
	double maximum = 0.0;
	for(int point = 0; point < 12; ++point)
	{
		newton::Vector primal = seed;
		for(int i = 0; i < dofs; ++i) primal[i] += 0.1 * sample(random);
		if(!evaluate(problem, primal, value)) return false;
		buildJacobian(problem, value, matrix, entries);
		for(int probe = 0; probe < 8; ++probe)
		{
			newton::Vector direction(dofs);
			for(int i = 0; i < dofs; ++i) direction[i] = sample(random);
			direction.normalize();
			const double step = 1.0e-6 * std::max(1.0, primal.norm());
			if(!evaluate(problem, primal + step * direction, plus) || !evaluate(problem, primal - step * direction, minus)) return false;
			if(plus.state != value.state || minus.state != value.state) { ++skipped; continue; }
			const newton::Vector finite = (plus.gradient - minus.gradient) / (2.0 * step);
			const newton::Vector exact = matrix * direction;
			const double error = (exact - finite).norm() / std::max(1.0, exact.norm());
			maximum = std::max(maximum, error);
			++checked;
		}
	}
	std::printf("DERIVATIVE,checked=%d,skipped_kinks=%d,max_relative=%.12g\n", checked, skipped, maximum);
	return checked > 0 && maximum < 5.0e-7;
}

static int solveSemismooth(const newton::Problem& problem, const newton::Settings& settings,
	newton::Vector& primal, Evaluation& value)
{
	const int dofs = problem.bodyCount() * 6;
	Evaluation trial(problem.rowCount(), dofs);
	newton::Sparse matrix(dofs, dofs);
	std::vector<Entry> entries;
	Eigen::SparseLU<newton::Sparse, Eigen::COLAMDOrdering<int> > factor;
	newton::Vector direction(dofs), candidate(dofs), weighted(dofs), meritGradient(dofs);
	bool analyzed = false;
	int iterations = 0, evaluations = 0, backtracks = 0, fallbacks = 0, status = 1;
	double assemblyMs = 0.0, factorMs = 0.0, evaluationMs = 0.0;
	const newton::Clock::time_point start = newton::Clock::now();
	if(!evaluate(problem, primal, value)) return 3;
	++evaluations;
	while(iterations < settings.iterations)
	{
		if(value.scaledGradient <= settings.tolerance && value.physicalResidual <= settings.tolerance) { status = 0; break; }
		newton::Clock::time_point stamp = newton::Clock::now();
		buildJacobian(problem, value, matrix, entries);
		assemblyMs += newton::elapsed(stamp);
		stamp = newton::Clock::now();
		if(!analyzed) { factor.analyzePattern(matrix); analyzed = true; }
		factor.factorize(matrix);
		const bool factorOk = factor.info() == Eigen::Success;
		if(factorOk) direction = factor.solve(-value.gradient);
		factorMs += newton::elapsed(stamp);
		++iterations;
		weighted = problem.massDiagonal.cwiseProduct(value.gradient);
		meritGradient.noalias() = matrix.transpose() * weighted;
		const double scale = problem.massDiagonal.sum() * problem.timestep;
		meritGradient /= scale * scale;
		bool accepted = false;
		double acceptedAlpha = 0.0;
		for(int attempt = 0; attempt < 2 && !accepted; ++attempt)
		{
			if(attempt == 0 && (!factorOk || !direction.allFinite() || factor.info() != Eigen::Success)) continue;
			if(attempt == 1)
			{
				++fallbacks;
				// Gradient of the actual squared momentum residual in this smooth
				// region. It is a fallback direction, never a convergence criterion.
				direction = -meritGradient;
				const double norm = direction.norm();
				if(norm > 1.0) direction /= norm;
			}
			const double slope = meritGradient.dot(direction);
			if(!(slope < 0.0) || !std::isfinite(slope)) continue;
			double alpha = 1.0;
			for(int line = 0; line < 40; ++line)
			{
				candidate = primal + alpha * direction;
				if((candidate.array() == primal.array()).all()) break;
				stamp = newton::Clock::now();
				const bool finite = evaluate(problem, candidate, trial);
				evaluationMs += newton::elapsed(stamp);
				++evaluations;
				if(finite && trial.merit <= value.merit + 1.0e-4 * alpha * slope)
				{
					accepted = true;
					acceptedAlpha = alpha;
					break;
				}
				alpha *= 0.5;
				++backtracks;
			}
		}
		std::printf("ITERATION,index=%d,gradient=%.12g,physical=%.12g,alpha=%.12g,accepted=%d\n",
			iterations, value.scaledGradient, value.physicalResidual, acceptedAlpha, int(accepted));
		if(!accepted) { status = 2; break; }
		primal = candidate;
		value = trial;
	}
	if(value.scaledGradient <= settings.tolerance && value.physicalResidual <= settings.tolerance) status = 0;
	std::printf("RESULT,status=%d,iterations=%d,evaluations=%d,backtracks=%d,steepest_fallbacks=%d,gradient=%.12g,physical=%.12g,ms=%.9g,assembly_ms=%.9g,factor_ms=%.9g,evaluation_ms=%.9g\n",
		status, iterations, evaluations, backtracks, fallbacks, value.scaledGradient, value.physicalResidual,
		newton::elapsed(start), assemblyMs, factorMs, evaluationMs);
	return status;
}

int main(int argc, char** argv)
{
	if(argc < 2)
	{
		std::fprintf(stderr, "Usage: NewtonNativeSemismoothReplay capture [iterations] [cold] [audit] [solution-or--] [convex_seed]\n");
		return 1;
	}
	newton::Problem problem;
	newton::Settings settings;
	newton::Vector primal;
	if(!loadCapture(argv[1], problem, settings, primal)) { std::fprintf(stderr, "Invalid native capture\n"); return 1; }
	if(argc > 2) settings.iterations = std::atoi(argv[2]);
	if(argc > 3 && std::atoi(argv[3])) primal.setZero();
	std::printf("PROBLEM,bodies=%d,rows=%d,patches=%zu,tolerance=%.12g,cap=%d\n",
		problem.bodyCount(), problem.rowCount(), problem.patches.size(), settings.tolerance, settings.iterations);
	if(argc > 4 && std::atoi(argv[4]) && !derivativeAudit(problem, primal)) return 4;
	Evaluation value(problem.rowCount(), problem.bodyCount() * 6);
	if(argc > 6 && std::atoi(argv[6]))
	{
		newton::Workspace workspace;
		newton::Result initial, seeded;
		initial.primal = primal;
		const newton::SolveStatus::Enum seedStatus = newton::solveNewton(problem, settings, seeded, workspace, &initial);
		Evaluation seededValue(problem.rowCount(), problem.bodyCount() * 6);
		bool accepted = false;
		if((seedStatus == newton::SolveStatus::eSUCCESS || seedStatus == newton::SolveStatus::eITERATION_LIMIT) &&
			evaluate(problem, primal, value) && evaluate(problem, seeded.primal, seededValue) && seededValue.merit < value.merit)
		{
			primal = seeded.primal;
			accepted = true;
		}
		settings.iterations -= std::max(1, seeded.iterations);
		std::printf("CONVEX_SEED,status=%d,iterations=%d,accepted=%d,remaining=%d,ms=%.9g\n",
			int(seedStatus), seeded.iterations, int(accepted), settings.iterations, seeded.elapsedMs);
	}
	const int status = solveSemismooth(problem, settings, primal, value);
	if(argc > 5 && std::string(argv[5]) != "-")
	{
		std::ofstream output(argv[5]);
		output << std::setprecision(17);
		for(int i = 0; i < value.impulse.size(); ++i) output << value.impulse[i] << '\n';
		for(int i = 0; i < primal.size(); ++i) output << primal[i] << '\n';
	}
	// 0 success, 1 cap, 2 no merit decrease, 3 nonfinite equation, 4 audit failure.
	return status;
}
