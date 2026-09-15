#include "NewtonSolver.h"
#include <Eigen/Geometry>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <omp.h>

namespace newton
{
void setMassDiagonal(Problem& problem, double inertiaPerMass)
{
	problem.massDiagonal.resize(problem.bodyCount() * 6);
	for(int i = 0; i < problem.bodyCount(); ++i)
	{
		problem.massDiagonal.segment<3>(6 * i).setConstant(1.0 / problem.inverseMass[i]);
		problem.massDiagonal.segment<3>(6 * i + 3).setConstant(inertiaPerMass / problem.inverseMass[i]);
	}
}

// Prepared MuJoCo equations in mass-scaled coordinates. The file records the
// actual Jacobian, reference acceleration and regularization used by MuJoCo.
static Problem readProblem(const char* path)
{
	std::ifstream input(path);
	Problem problem;
	problem.name = path;
	int bodies, contacts;
	input >> bodies >> contacts >> problem.timestep;
	problem.inverseMass.resize(bodies);
	for(int i = 0; i < bodies; ++i)
		input >> problem.inverseMass[i];
	setMassDiagonal(problem, 1.0 / 6.0);
	problem.freeBodyVelocity = Vector::Zero(bodies * 6);
	reserveStorage(problem.contacts, size_t(contacts));
	for(int i = 0; i < contacts; ++i)
	{
		Contact c;
		input >> c.body[0] >> c.body[1] >> c.friction;
		for(int axis = 0; axis < 3; ++axis)
			input >> c.freeVelocity[axis];
		for(int axis = 0; axis < 3; ++axis)
			input >> c.regularization[axis];
		for(int end = 0; end < 2; ++end)
			for(int axis = 0; axis < 3; ++axis)
				for(int column = 0; column < 6; ++column)
				{
					input >> c.jacobian[end](axis, column);
				}
		problem.addContact(c);
	}
	prepareProblem(problem);
	if(!input || problem.regularization.minCoeff() <= 0.0)
		throw std::runtime_error("Invalid prepared contact problem");
	// Optional trailer preserves the exact physical inertia of new reference exports.
	std::string trailer;
	if(input >> trailer)
	{
		if(trailer != "mass_diagonal")
			throw std::runtime_error("Unknown prepared problem trailer");
		for(int i = 0; i < bodies * 6; ++i)
			input >> problem.massDiagonal[i];
		if(!input || problem.massDiagonal.minCoeff() <= 0.0)
			throw std::runtime_error("Invalid prepared mass diagonal");
	}
	return problem;
}

// Exercise scalar, equality and coupled friction rows, changing island sizes
// between solves to check workspace resizing and symbolic-pattern invalidation.
static int validateMixedConstraints()
{
	Settings settings;
	settings.checkFactor = true;
	settings.tolerance = 1.0e-12;
	const int bodyCounts[] = {8, 3, 8, 12, 5, 8};
	for(int fixture = 0; fixture < 6; ++fixture)
	{
		Problem problem;
		problem.timestep = 0.01;
		const int bodies = bodyCounts[fixture];
		problem.inverseMass.assign(bodies, 1.0);
		setMassDiagonal(problem, 1.0 / 6.0);
		for(int i = 0; i < bodies * 4; ++i)
		{
			Contact contact;
			contact.body[0] = i % bodies;
			contact.body[1] = i < bodies ? -1 : (i + 1) % bodies;
			contact.bilateral = fixture % 3 == 2 && i % 3 == 0;
			contact.friction = fixture % 3 > 0 && i % 3 == 1 ? 0.0 : 0.5;
			contact.regularization.setConstant(0.05);
			contact.freeVelocity = Vec3(std::sin(double(i)), std::cos(double(i)), -0.2);
			for(int end = 0; end < 2; ++end)
				for(int row = 0; row < 3; ++row)
					for(int column = 0; column < 6; ++column)
						contact.jacobian[end](row, column) = contact.body[end] < 0 ? 0.0 :
							std::sin(0.37 * (1 + i * 13 + end * 19 + row * 7 + column));
			problem.addContact(contact);
		}
		prepareProblem(problem);
		Result cold;
		solveNewton(problem, settings, cold);
		problem.freeVelocity *= 1.01;
		Result warm;
		solveNewton(problem, settings, warm, &cold);
		if(!warm.primal.allFinite() || warm.gradientResidual > 1.0e-6 || cold.gradientResidual > 1.0e-6)
			throw std::runtime_error("Mixed constraint solve did not converge");
		std::printf("VALIDATED,fixture=%d,rows=%d,iterations=%d,gradient=%.12g,factor_error=%.12g\n",
			fixture, problem.rowCount(), warm.iterations, warm.gradientResidual, warm.factorError);
	}
	return 0;
}

#include "CableBenchmark.h"
}
int main(int argc, char** argv)
{
	using namespace newton;
	if(argc < 2)
	{
		std::fprintf(stderr, "Usage: NewtonBenchmark input [seed|-] [solution|-] [repeats=8] [iterations=100] [check=0] [islands=1] [threads=1]\n"
			"       NewtonBenchmark cable links steps [trajectory]\n");
		return 2;
	}
	try
	{
		Settings settings;
		if(std::string(argv[1]) == "validate")
			return validateMixedConstraints();
		if(std::string(argv[1]) == "cable")
		{
			if(argc < 4)
				throw std::runtime_error("Specify cable link and step counts");
			return benchmarkCable(std::atoi(argv[2]), std::atoi(argv[3]), argc > 4 ? argv[4] : NULL, settings);
		}
		const Problem problem = readProblem(argv[1]);
		const int repeats = argc > 4 ? std::atoi(argv[4]) : 8;
		settings.iterations = argc > 5 ? std::atoi(argv[5]) : 100;
		settings.checkFactor = argc > 6 && std::atoi(argv[6]) != 0;
		const int islands = argc > 7 ? std::atoi(argv[7]) : 1;
		const int threads = argc > 8 ? std::atoi(argv[8]) : 1;
		if(repeats < 1 || settings.iterations < 1 || islands < 1 || threads < 1)
			throw std::runtime_error("Invalid benchmark counts");
		Result previous;
		const bool warm = argc > 2 && std::string(argv[2]) != "-";
		if(warm)
		{
			std::ifstream seed(argv[2]);
			previous.impulse.resize(problem.rowCount());
			previous.primal.resize(problem.bodyCount() * 6);
			for(size_t i = 0; i < problem.contacts.size(); ++i)
			{
				const CompactContact& contact = problem.contacts[i];
				for(int axis = 0; axis < 3; ++axis)
				{
					double value;
					seed >> value;
					if(contact.rowCount() == 3)
						previous.impulse[contact.row + axis] = value;
					else if(axis == 2)
						previous.impulse[contact.row] = value;
				}
			}
			for(int row = 0; row < previous.primal.size(); ++row)
				seed >> previous.primal[row];
			if(!seed)
				throw std::runtime_error("Invalid warm-start file");
		}
		omp_set_dynamic(0);
		std::vector<Result> results(islands);
		for(int repeat = 0; repeat < repeats; ++repeat)
		{
			const Clock::time_point start = Clock::now();
			#pragma omp parallel for num_threads(threads) if(islands > 1)
			for(int island = 0; island < islands; ++island)
				solveNewton(problem, settings, results[island], warm ? &previous : NULL);
			const double wallMs = elapsed(start);
			const Result& r = results[0];
			const double modelResidual = computeResidual(problem, r.impulse);
			std::printf("RESULT,ms=%.6f,wall_ms=%.6f,iterations=%d,factors=%d,rank_updates=%d,reused_factors=%d,fallbacks=%d,matrix_ms=%.6f,factor_ms=%.6f,update_ms=%.6f,conversion_ms=%.6f,backsolve_ms=%.6f,evaluation_ms=%.6f,line_ms=%.6f,line_evaluations=%d,gradient_residual=%.12g,model_residual=%.12g,scaled_gradient=%.12g,factor_error=%.12g,stop_reason=%d\n",
				r.elapsedMs, wallMs, r.iterations, r.factorizations, r.rankUpdates, r.reusedFactors, r.factorFallbacks,
				r.matrixMs, r.factorMs, r.updateMs, r.conversionMs, r.backsolveMs, r.evaluationMs, r.lineSearchMs,
				r.lineSearchEvaluations, r.gradientResidual, modelResidual, r.scaledGradient, r.factorError, r.stopReason);
			for(int island = 0; island < islands; ++island)
				if(!results[island].primal.allFinite() || !results[island].impulse.allFinite())
					throw std::runtime_error("Non-finite Newton result");
			for(int island = 1; island < islands; ++island)
				if((results[island].primal - r.primal).lpNorm<Eigen::Infinity>() > 1.0e-12)
					throw std::runtime_error("Independent islands produced different solutions");
		}
		if(argc > 3 && std::string(argv[3]) != "-")
		{
			std::ofstream output(argv[3]);
			output << std::setprecision(17);
			// Preserve the prepared-file format; padding exists only in file I/O.
			for(size_t i = 0; i < problem.contacts.size(); ++i)
			{
				const CompactContact& contact = problem.contacts[i];
				if(contact.rowCount() == 1)
					output << "0\n0\n" << results[0].impulse[contact.row] << "\n";
				else
					output << results[0].impulse.segment<3>(contact.row) << "\n";
			}
			output << results[0].primal << "\n";
		}
	}
	catch(const std::exception& error)
	{
		std::fprintf(stderr, "%s\n", error.what());
		return 1;
	}
	return 0;
}
