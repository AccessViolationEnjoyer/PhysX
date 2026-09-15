#include "NewtonSolver.h"
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace newton;

static void preparePatch(Problem& problem, int fixture, int bodies = 2, int normals = 2, int tangents = 2)
{
	problem.timestep = 0.01;
	problem.inverseMass.assign(bodies, 1.0);
	problem.massDiagonal.setZero(6 * bodies);
	problem.massDiagonal.setOnes();
	problem.clearContacts();
	const double infinity = std::numeric_limits<double>::infinity();
	for(int row = 0; row < normals + tangents; ++row)
	{
		CompactContact contact;
		contact.body[0] = fixture % 2 ? bodies - 1 : 0;
		contact.body[1] = bodies == 1 ? -1 : (fixture % 2 ? 0 : bodies - 1);
		contact.freeVelocity = row < normals ? -0.4 + 0.65 * std::sin(0.67 * (fixture + row)) : 1.2 * std::cos(0.31 * (fixture - row));
		contact.regularization = 0.008 * (1 + (fixture + row * 7) % 9);
		for(int end = 0; end < 2; ++end)
			for(int axis = 0; axis < 6; ++axis)
				contact.jacobian[end][axis] = contact.body[end] < 0 ? 0.0 :
					0.35 * std::sin(0.39 * (1 + row * 11 + end * 19 + axis * 3 + fixture));
		const double cap = fixture % 5 == 0 ? 0.0 : (fixture % 3 == 0 ? infinity : 0.07 + 0.03 * row);
		problem.addScalarContact(contact, row < normals ? 0.0 : -infinity, row < normals ? cap : infinity);
	}
	problem.addPatch(0, normals, tangents, fixture % 7 == 0 ? 0.0 : 0.5);
}

// Independently solve the four-variable dual QP by enumerating inequality
// active sets. The reference contains no patch projection or Newton curvature.
static bool referencePatch(const Problem& problem, Vector& best)
{
	const Eigen::MatrixXd jacobian = Eigen::MatrixXd(problem.jacobian);
	Eigen::MatrixXd quadratic = jacobian * jacobian.transpose();
	quadratic.diagonal() += problem.regularization;
	Eigen::Matrix<double, 8, 4> inequalities;
	inequalities.setZero();
	Eigen::Matrix<double, 8, 1> bounds;
	bounds.setZero();
	int count = 0;
	for(int row = 0; row < 2; ++row)
	{
		inequalities(count++, row) = -1.0;
		const CompactContact& contact = problem.contacts[row];
		const double cap = contact.hasScalarBounds() ? problem.bounds(contact).upper : std::numeric_limits<double>::infinity();
		if(std::isfinite(cap))
		{
			inequalities(count, row) = 1.0;
			bounds[count++] = cap;
		}
	}
	for(int row = 2; row < 4; ++row)
		for(int sign = -1; sign <= 1; sign += 2)
		{
			inequalities(count, 0) = inequalities(count, 1) = -problem.patches[0].friction;
			inequalities(count++, row) = double(sign);
		}
	double bestCost = std::numeric_limits<double>::infinity();
	for(int mask = 0; mask < (1 << count); ++mask)
	{
		int active[4], size = 0;
		for(int row = 0; row < count; ++row)
			if(mask & (1 << row))
			{
				if(size < 4)
					active[size] = row;
				++size;
			}
		if(size > 4)
			continue;
		Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(4 + size, 4 + size);
		matrix.topLeftCorner(4, 4) = quadratic;
		Vector rhs(4 + size);
		rhs.head(4) = -problem.freeVelocity;
		for(int row = 0; row < size; ++row)
		{
			matrix.block(4 + row, 0, 1, 4) = inequalities.row(active[row]);
			matrix.block(0, 4 + row, 4, 1) = inequalities.row(active[row]).transpose();
			rhs[4 + row] = bounds[active[row]];
		}
		Eigen::FullPivLU<Eigen::MatrixXd> factor(matrix);
		if(!factor.isInvertible())
			continue;
		const Vector candidate = factor.solve(rhs);
		if((inequalities.topRows(count) * candidate.head(4) - bounds.head(count)).maxCoeff() > 1.0e-9 ||
			(size && candidate.tail(size).minCoeff() < -1.0e-9))
			continue;
		const double cost = 0.5 * candidate.head(4).dot(quadratic * candidate.head(4)) + problem.freeVelocity.dot(candidate.head(4));
		if(cost < bestCost)
		{
			bestCost = cost;
			best = candidate.head(4);
		}
	}
	return std::isfinite(bestCost);
}

int main()
{
	Problem problem;
	Workspace workspace;
	Result result;
	Settings settings;
	settings.tolerance = 1.0e-13;
	settings.checkFactor = true;
	double maximumDifference = 0.0, maximumFactorError = 0.0, maximumResidual = 0.0;
	int updates = 0;
	for(int fixture = 0; fixture < 48; ++fixture)
	{
		preparePatch(problem, fixture, fixture % 3 == 1 ? 1 : 2);
		if(prepareProblem(problem) != SolveStatus::eSUCCESS)
			return 1;
		Vector reference;
		if(!referencePatch(problem, reference) || solveNewton(problem, settings, result, workspace) != SolveStatus::eSUCCESS)
		{
			std::printf("PATCH_FAILED fixture=%d status=%d\n", fixture, int(result.status));
			return 1;
		}
		const double difference = (reference - result.impulse).lpNorm<Eigen::Infinity>();
		maximumDifference = std::max(maximumDifference, difference);
		maximumFactorError = std::max(maximumFactorError, result.factorError);
		maximumResidual = std::max(maximumResidual, computeResidual(problem, result.impulse));
		updates += result.rankUpdates;
		if(difference > 1.0e-8 || maximumResidual > 1.0e-8)
		{
			std::printf("PATCH_MISMATCH fixture=%d difference=%.12g residual=%.12g\n", fixture, difference, maximumResidual);
			return 1;
		}
		// Friction changes are an allowed phase update without reconstructing J.
		problem.patches[0].friction *= 0.7;
		if(!referencePatch(problem, reference) || solveNewton(problem, settings, result, workspace, &result) != SolveStatus::eSUCCESS ||
			(reference - result.impulse).lpNorm<Eigen::Infinity>() > 1.0e-8)
			return 1;
	}
	// Larger/changing groups exercise the two-vector Hessian and sparse updates;
	// every Newton direction is independently checked against sparse J^T*H*J.
	for(int fixture = 0; fixture < 48; ++fixture)
	{
		preparePatch(problem, fixture, 12, 3 + fixture % 7, 4);
		if(prepareProblem(problem) != SolveStatus::eSUCCESS || solveNewton(problem, settings, result, workspace) != SolveStatus::eSUCCESS ||
			computeResidual(problem, result.impulse) > 1.0e-8)
		{
			std::printf("LARGE_PATCH_FAILED fixture=%d status=%d residual=%.12g factor=%.12g\n", fixture, int(result.status),
				computeResidual(problem, result.impulse), result.factorError);
			return 1;
		}
		maximumFactorError = std::max(maximumFactorError, result.factorError);
		updates += result.rankUpdates;
	}
	std::printf("PATCH_VALIDATION,small=96,large=48,max_difference=%.12g,max_residual=%.12g,max_factor_error=%.12g,updates=%d\n",
		maximumDifference, maximumResidual, maximumFactorError, updates);
	return 0;
}
