#include "NewtonSolver.h"
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

static void initialize(newton::Problem& problem, int bodies)
{
	problem.timestep = 0.01;
	problem.inverseMass.assign(bodies, 1.0);
	problem.massDiagonal.resize(6 * bodies);
	problem.massDiagonal.setOnes();
	problem.clearContacts();
}

// Independent dual active-set enumeration: solve every free/fixed combination.
// This uses Q=J*J^T+R and never calls the production projection or line search.
static bool referenceBounds(const newton::Problem& problem, newton::Vector& impulse)
{
	const int count = problem.rowCount();
	Eigen::MatrixXd matrix = Eigen::MatrixXd(problem.jacobian * problem.jacobian.transpose());
	matrix.diagonal() += problem.regularization;
	newton::Vector lower(count), upper(count);
	for(int i = 0; i < count; ++i)
	{
		const newton::CompactContact& contact = problem.contacts[i];
		lower[i] = contact.hasScalarBounds() ? problem.bounds(contact).lower : 0.0;
		upper[i] = contact.hasScalarBounds() ? problem.bounds(contact).upper : std::numeric_limits<double>::infinity();
	}
	int combinations = 1;
	for(int i = 0; i < count; ++i)
		combinations *= 3;
	for(int candidate = 0; candidate < combinations; ++candidate)
	{
		int code = candidate;
		std::vector<int> freeRows;
		newton::Vector trial = newton::Vector::Zero(count);
		bool valid = true;
		for(int row = 0; row < count; ++row)
		{
			const int state = code % 3;
			code /= 3;
			if(state == 0)
				freeRows.push_back(row);
			else
			{
				trial[row] = state == 1 ? lower[row] : upper[row];
				valid = valid && std::isfinite(trial[row]);
			}
		}
		if(!valid)
			continue;
		const int freeCount = int(freeRows.size());
		Eigen::MatrixXd reduced(freeCount, freeCount);
		newton::Vector rhs(freeCount);
		for(int i = 0; i < freeCount; ++i)
		{
			rhs[i] = -problem.freeVelocity[freeRows[i]] - matrix.row(freeRows[i]).dot(trial);
			for(int j = 0; j < freeCount; ++j)
				reduced(i,j) = matrix(freeRows[i],freeRows[j]);
		}
		if(freeCount)
		{
			const newton::Vector solved = reduced.ldlt().solve(rhs);
			for(int i = 0; i < freeCount; ++i)
				trial[freeRows[i]] = solved[i];
		}
		const newton::Vector gradient = matrix * trial + problem.freeVelocity;
		for(int row = 0; row < count; ++row)
		{
			if(trial[row] < lower[row] - 1.0e-10 || trial[row] > upper[row] + 1.0e-10)
				valid = false;
			if(lower[row] == upper[row])
				continue;
			if(trial[row] <= lower[row] + 1.0e-10)
				valid = valid && gradient[row] >= -1.0e-9;
			else if(trial[row] >= upper[row] - 1.0e-10)
				valid = valid && gradient[row] <= 1.0e-9;
			else
				valid = valid && std::abs(gradient[row]) < 1.0e-9;
		}
		if(valid)
		{
			impulse = trial;
			return true;
		}
	}
	return false;
}

static bool validateBounds(newton::Workspace& workspace)
{
	const double infinity = std::numeric_limits<double>::infinity();
	const double lower[] = {0.0, -infinity, -0.2, 0.04, -infinity, 0.07};
	const double upper[] = {infinity, infinity, 0.3, 0.21, 0.0, 0.07};
	for(int fixture = 0; fixture < 16; ++fixture)
	{
		newton::Problem problem;
		initialize(problem, fixture % 2 + 1);
		for(int row = 0; row < 6; ++row)
		{
			newton::CompactContact contact;
			contact.body[0] = 0;
			contact.body[1] = problem.bodyCount() == 2 ? 1 : -1;
			contact.freeVelocity = 2.0 * std::sin(0.31 * (fixture + 1) * (row + 2));
			contact.regularization = 0.02 + 0.01 * (row % 3);
			for(int side = 0; side < 2; ++side)
				for(int axis = 0; axis < 6; ++axis)
					contact.jacobian[side][axis] = contact.body[side] < 0 ? 0.0 :
						std::sin(0.23 * (1 + fixture * 3 + row * 7 + side * 5 + axis));
			// Repeated Jacobians exercise independently bounded forces on one axis.
			if(row == 2)
			{
				contact.jacobian[0] = problem.contacts[1].jacobian[0];
				contact.jacobian[1] = problem.contacts[1].jacobian[1];
			}
			problem.addScalarContact(contact, lower[row], upper[row]);
		}
		if(newton::prepareProblem(problem) != newton::SolveStatus::eSUCCESS)
			return false;
		newton::Vector expected;
		if(!referenceBounds(problem, expected))
			return false;
		newton::Settings settings;
		settings.tolerance = 1.0e-14;
		settings.checkFactor = true;
		newton::Result result;
		if(newton::solveNewton(problem, settings, result, workspace) != newton::SolveStatus::eSUCCESS)
			return false;
		const double error = (result.impulse - expected).lpNorm<Eigen::Infinity>();
		if(error > 1.0e-8 || newton::computeResidual(problem, result.impulse) > 1.0e-8)
		{
			std::printf("BOUND_FAILURE,%d,%.17g\n", fixture, error);
			return false;
		}
		if(newton::solveNewton(problem, settings, result, workspace, &result) != newton::SolveStatus::eSUCCESS || result.iterations != 0)
			return false;
		// Change finite/equality bounds in place: no re-preparation and the warm
		// seed may now be worse than free motion. Check the new exact dual optimum.
		for(int phase = 0; phase < 3; ++phase)
		{
			if(!problem.setScalarBounds(1, phase == 1 ? -infinity : -0.01, phase == 1 ? infinity : 0.015) ||
				!problem.setScalarBounds(3, phase == 2 ? -infinity : 0.02, phase == 2 ? infinity : 0.04) ||
				!referenceBounds(problem, expected) ||
				newton::solveNewton(problem, settings, result, workspace, &result) != newton::SolveStatus::eSUCCESS ||
				(result.impulse - expected).lpNorm<Eigen::Infinity>() > 1.0e-8)
				return false;
		}

	}
	return true;
}

// Independent constrained dual solve for the six planar faces of a capped pyramid.
static bool referenceCone(const Eigen::Matrix3d& matrix, const newton::Vec3& free,
	double friction, double cap, newton::Vec3& impulse)
{
	Eigen::Matrix<double,6,3> bounds;
	bounds << 0,0,-1, 0,0,1, 1,0,-friction, -1,0,-friction, 0,1,-friction, 0,-1,-friction;
	Eigen::Matrix<double,6,1> limit;
	limit << 0,cap,0,0,0,0;
	for(int mask = 0; mask < 64; ++mask)
	{
		std::vector<int> active;
		for(int i = 0; i < 6; ++i)
			if(mask & (1 << i))
				active.push_back(i);
		if(active.size() > 3)
			continue;
		const int count = int(active.size());
		Eigen::MatrixXd kkt = Eigen::MatrixXd::Zero(3 + count, 3 + count);
		kkt.topLeftCorner<3,3>() = matrix;
		newton::Vector rhs(3 + count);
		rhs.head<3>() = -free;
		for(int i = 0; i < count; ++i)
		{
			kkt.block<1,3>(3+i,0) = bounds.row(active[i]);
			kkt.block<3,1>(0,3+i) = bounds.row(active[i]).transpose();
			rhs[3+i] = limit[active[i]];
		}
		Eigen::FullPivLU<Eigen::MatrixXd> factor(kkt);
		if(!factor.isInvertible())
			continue;
		const newton::Vector answer = factor.solve(rhs);
		if((bounds * answer.head<3>() - limit).maxCoeff() > 1.0e-9)
			continue;
		if(count && answer.tail(count).minCoeff() < -1.0e-9)
			continue;
		impulse = answer.head<3>();
		return true;
	}
	return false;
}

static bool validateCappedCones(newton::Workspace& workspace)
{
	for(int fixture = 0; fixture < 18; ++fixture)
	{
		newton::Problem problem;
		initialize(problem, 1);
		newton::Contact contact;
		contact.body[0] = 0;
		contact.body[1] = -1;
		contact.friction = 0.6;
		contact.maxNormalImpulse = fixture % 3 == 0 ? 0.0 : 0.05 + 0.025 * fixture;
		contact.regularization = newton::Vec3(0.04,0.04,0.02);
		contact.freeVelocity = newton::Vec3(0.1 * fixture - 0.8, 0.2 - 0.13 * fixture, -1.0);
		contact.jacobian[1].setZero();
		for(int row = 0; row < 3; ++row)
			for(int axis = 0; axis < 6; ++axis)
				contact.jacobian[0](row,axis) = std::sin(0.31*(1+row*6+axis+fixture));
		problem.addContact(contact);
		if(newton::prepareProblem(problem) != newton::SolveStatus::eSUCCESS)
			return false;
		Eigen::Matrix3d matrix = contact.jacobian[0] * contact.jacobian[0].transpose();
		matrix.diagonal() += contact.regularization;
		newton::Vec3 expected;
		if(!referenceCone(matrix, contact.freeVelocity, contact.friction, contact.maxNormalImpulse, expected))
			return false;
		newton::Settings settings;
		settings.tolerance = 1.0e-14;
		settings.checkFactor = true;
		newton::Result result;
		if(newton::solveNewton(problem, settings, result, workspace) != newton::SolveStatus::eSUCCESS)
			return false;
		const double error = (result.impulse - expected).lpNorm<Eigen::Infinity>();
		if(error > 1.0e-8 || newton::computeResidual(problem,result.impulse) > 1.0e-8)
		{
			std::printf("CAP_FAILURE,%d,%.17g\n",fixture,error);
			return false;
		}
	}
	return true;
}

int main()
{
	static_assert(sizeof(newton::CompactContact) == 128, "Scalar contact storage must remain compact");
	newton::Workspace workspace;
	if(!validateBounds(workspace) || !validateCappedCones(workspace))
		return 1;
	newton::Workspace moved(std::move(workspace));
	if(!validateBounds(moved))
		return 1;
	newton::Problem empty;
	initialize(empty,0);
	newton::Settings settings;
	newton::Result result;
	if(newton::prepareProblem(empty) != newton::SolveStatus::eSUCCESS ||
		newton::solveNewton(empty,settings,result,workspace) != newton::SolveStatus::eSUCCESS)
		return 1;
	empty.timestep = 0.0;
	if(newton::solveNewton(empty,settings,result,workspace) != newton::SolveStatus::eINVALID_INPUT)
		return 1;
	std::printf("VALIDATED,scalar_active_sets=128,capped_cones=18,workspace_move=1,status=1,scalar_bytes=%zu\n",sizeof(newton::CompactContact));
	return 0;
}
