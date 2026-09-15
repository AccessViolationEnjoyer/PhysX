#include "NewtonSolver.h"
#include <atomic>
#include "NewtonPatchProjection.h"
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <new>
#include <limits>

#include "EigenStorageKernels.h"
#include "BlockCholesky.h"

namespace newton
{
typedef Eigen::Triplet<double> Triplet;
typedef Eigen::SimplicialLDLT<Sparse> Factor;

static double clampValue(double value, double lower, double upper)
{
	return value < lower ? lower : (value > upper ? upper : value);
}

static Clock::time_point profileStart(bool enabled)
{
	return enabled ? Clock::now() : Clock::time_point();
}

static double profileElapsed(bool enabled, Clock::time_point start)
{
	return enabled ? elapsed(start) : 0.0;
}

static Vec3 projectPyramid(const Vec3& value, double friction, Mat3* derivative = NULL)
{

	const double a = std::abs(value[0]), b = std::abs(value[1]);
	if(std::max(a, b) <= friction * value[2])
	{
		if(derivative)
			derivative->setIdentity();
		return value;
	}
	if(value[2] <= -friction * (a + b))
	{
		if(derivative)
			derivative->setZero();
		return Vec3::Zero();
	}
	double normal = (value[2] + friction * (a + b)) / (1.0 + 2.0 * friction * friction);
	if(friction * normal > std::min(a, b))
		normal = (value[2] + friction * std::max(a, b)) / (1.0 + friction * friction);
	Vec3 result = value;
	result[2] = normal;
	result.head<2>() = value.head<2>().cwiseMax(-friction * normal).cwiseMin(friction * normal);
	if(derivative)
	{
		Vec3 direction(0.0, 0.0, 1.0);
		derivative->setZero();
		for(int i = 0; i < 2; ++i)
			if(std::abs(value[i]) > friction * normal)
				direction[i] = std::copysign(friction, value[i]);
			else
				(*derivative)(i, i) = 1.0;
		*derivative += direction * direction.transpose() / direction.squaredNorm();
	}
	return result;
}

static Vec3 weightedImpulse(const Vec3& drive, const Vec3& compliance, double friction, Mat3* derivative = NULL, bool bilateral = false, double maxNormalImpulse = std::numeric_limits<double>::infinity())
{
	if(bilateral)
	{
		if(derivative)
			*derivative = compliance.cwiseInverse().asDiagonal();
		return drive.cwiseQuotient(compliance);
	}

	const Vec3 inverseSqrt = compliance.cwiseSqrt().cwiseInverse();
	const double scaledFriction = friction * std::sqrt(compliance[0] / compliance[2]);
	Mat3 projectionDerivative;
	Vec3 impulse = inverseSqrt.cwiseProduct(projectPyramid(inverseSqrt.cwiseProduct(drive),
		scaledFriction, derivative ? &projectionDerivative : NULL));
	if(derivative)
		*derivative = inverseSqrt.asDiagonal() * projectionDerivative * inverseSqrt.asDiagonal();
	if(impulse[2] > maxNormalImpulse)
	{
		impulse[2] = maxNormalImpulse;
		const double tangentLimit = friction * maxNormalImpulse;
		if(derivative)
			derivative->setZero();
		for(int axis = 0; axis < 2; ++axis)
		{
			const double unconstrained = drive[axis] / compliance[axis];
			impulse[axis] = clampValue(unconstrained, -tangentLimit, tangentLimit);
			if(derivative && unconstrained > -tangentLimit && unconstrained < tangentLimit)
				(*derivative)(axis, axis) = 1.0 / compliance[axis];
		}
	}
	return impulse;
}

// Most constraint Hessians are diagonal. Only three-component friction
// projections need a dense block; scalar pyramid edges never allocate one.
struct PatchCurvature
{
	Vec6 normal[2], tangent[2];
	double normalCoefficient, crossCoefficient, tangentCoefficient;
	double tangentCoupling[4];
};

struct PatchScratch
{
	VectorStorage cap, velocity, impulse, diagonal;
	double friction[4], coupling[4];
	void resize(const Problem& problem)
	{
		int normals = 0, rows = 0;
		for(size_t i = 0; i < problem.patches.size(); ++i)
		{
			normals = std::max(normals, problem.patches[i].normalCount);
			rows = std::max(rows, problem.patches[i].normalCount + problem.patches[i].tangentCount);
		}
		cap.resize(normals);
		velocity.resize(rows);
		impulse.resize(rows);
		diagonal.resize(rows);
	}
};

static bool projectGroup(const Problem& problem, const Patch& patch, const double* velocity,
	double* impulse, double* diagonal, double* coupling, PatchScratch& scratch, PatchProjectionResult& result)
{
	const int firstRow = problem.contacts[patch.firstContact].row;
	for(int i = 0; i < patch.normalCount; ++i)
	{
		const CompactContact& contact = problem.contacts[patch.firstContact + i];
		scratch.cap[i] = contact.hasScalarBounds() ? problem.bounds(contact).upper : std::numeric_limits<double>::infinity();
	}
	for(int i = 0; i < patch.tangentCount; ++i)
		scratch.friction[i] = patch.friction;
	const PatchProjectionInput input = { patch.normalCount, patch.tangentCount, velocity,
		problem.regularization.data() + firstRow, scratch.cap.data(), velocity + patch.normalCount,
		problem.regularization.data() + firstRow + patch.normalCount, scratch.friction };
	const PatchProjectionOutput output = { impulse, impulse + patch.normalCount, diagonal,
		diagonal ? diagonal + patch.normalCount : NULL, coupling };
	return projectPatchUnchecked(input, output, result);
}

struct Curvature
{
	VectorStorage diagonal;
	std::vector<Mat3> coupled;
	std::vector<PatchCurvature> patches;
	Curvature() {}
	explicit Curvature(const Problem& problem) { resize(problem); }
	void resize(const Problem& problem)
	{
		diagonal.setZero(problem.rowCount());
		coupled.resize(problem.coupledContacts.size());
		reserveStorage(patches, problem.patches.size());
		patches.resize(problem.patches.size());
	}
};

// Distinguish every successful preparation, including a new Problem constructed
// at a previously used address. One relaxed increment per preparation avoids
// scanning immutable coefficients during same-problem continuation.
static std::uint64_t nextPreparationGeneration()
{
	static std::atomic<std::uint64_t> generation(0);
	return generation.fetch_add(1, std::memory_order_relaxed) + 1;
}

int Problem::addScalarContact(const CompactContact& input, double lowerImpulse, double upperImpulse)
{
	CompactContact contact = input;
	contact.row = 0;
	contact.block = 0;
	if(lowerImpulse != 0.0 || upperImpulse != std::numeric_limits<double>::infinity())
	{
		const ScalarBounds limits = { lowerImpulse, upperImpulse };
		contact.block = CompactContact::SCALAR_BOUNDS_TAG - int(scalarBounds.size());
		reserveStorage(scalarBounds, scalarBounds.size() + 1);
		scalarBounds.push_back(limits);
	}
	reserveStorage(contacts, contacts.size() + 1);
	const int index = int(contacts.size());
	contacts.push_back(contact);
	prepared = false;
	return index;
}

bool Problem::setScalarBounds(int contactIndex, double lowerImpulse, double upperImpulse) noexcept
{
	const double infinity = std::numeric_limits<double>::infinity();
	if(contactIndex < 0 || contactIndex >= int(contacts.size()) || !(lowerImpulse <= upperImpulse) ||
		lowerImpulse == infinity || upperImpulse == -infinity || !contacts[contactIndex].hasScalarBounds())
		return false;
	for(size_t i = 0; i < patches.size(); ++i)
		if(contactIndex >= patches[i].firstContact &&
			contactIndex < patches[i].firstContact + patches[i].normalCount + patches[i].tangentCount)
			return false;
	ScalarBounds& limits = scalarBounds[size_t(CompactContact::SCALAR_BOUNDS_TAG - contacts[contactIndex].block)];
	if(limits.lower == lowerImpulse && limits.upper == upperImpulse)
		return true;
	const bool wasEquality = limits.lower == -infinity && limits.upper == infinity;
	const bool isEquality = lowerImpulse == -infinity && upperImpulse == infinity;
	limits.lower = lowerImpulse;
	limits.upper = upperImpulse;
	if(prepared)
	{
		equalityRows += int(isEquality) - int(wasEquality);
		preparationGeneration = nextPreparationGeneration();
	}
	if(!isEquality)
		hasFiniteBounds = true;
	return true;
}

int Problem::addPatch(int firstContact, int normalCount, int tangentCount, double friction)
{
	const Patch patch = { firstContact, normalCount, tangentCount, friction };
	reserveStorage(patches, patches.size() + 1);
	const int index = int(patches.size());
	patches.push_back(patch);
	prepared = false;
	return index;
}

void Problem::addContact(const Contact& input)
{
	prepared = false;
	CompactContact contact;
	contact.body[0] = input.body[0];
	contact.body[1] = input.body[1];
	contact.jacobian[0] = input.jacobian[0].row(2).transpose();
	contact.jacobian[1] = input.jacobian[1].row(2).transpose();
	contact.freeVelocity = input.freeVelocity[2];
	contact.regularization = input.regularization[2];
	if(input.rowCount() == 3)
	{
		ContactBlock additional;
		additional.tangentJacobian[0] = input.jacobian[0].topRows<2>();
		additional.tangentJacobian[1] = input.jacobian[1].topRows<2>();
		additional.freeVelocity = input.freeVelocity.head<2>();
		additional.regularization = input.regularization.head<2>();
		additional.friction = input.friction;
		additional.maxNormalImpulse = input.maxNormalImpulse;
		contact.block = (input.bilateral ? -1 : 1) * (int(contactBlocks.size()) + 1);
		reserveStorage(contactBlocks, contactBlocks.size() + 1);
		contactBlocks.push_back(additional);
	}
	if(input.rowCount() == 1 && input.maxNormalImpulse != std::numeric_limits<double>::infinity())
	{
		addScalarContact(contact, 0.0, input.maxNormalImpulse);
		return;
	}
	reserveStorage(contacts, contacts.size() + 1);
	contacts.push_back(contact);
}

static bool validContact(const Problem& problem, const CompactContact& contact)
{
	if(contact.body[0] < -1 || contact.body[1] < -1 || contact.body[0] >= problem.bodyCount() ||
		contact.body[1] >= problem.bodyCount() || (contact.body[0] >= 0 && contact.body[0] == contact.body[1]))
		return false;
	if(contact.hasScalarBounds())
	{
		const size_t index = size_t(CompactContact::SCALAR_BOUNDS_TAG - contact.block);
		if(index >= problem.scalarBounds.size())
			return false;
		const ScalarBounds& limits = problem.scalarBounds[index];
		return limits.lower <= limits.upper && limits.lower < std::numeric_limits<double>::infinity() &&
			limits.upper > -std::numeric_limits<double>::infinity();
	}
	if(contact.block)
	{
		const size_t index = size_t(contact.block > 0 ? contact.block - 1 : -contact.block - 1);
		if(index >= problem.contactBlocks.size())
			return false;
		const ContactBlock& block = problem.contactBlocks[index];
		if(contact.block > 0 && (!(block.friction >= 0.0) || !std::isfinite(block.friction) ||
			!(block.maxNormalImpulse >= 0.0) || block.regularization[0] != block.regularization[1]))
			return false;
	}
	return true;
}

template<bool CountEntries>
static bool prepareProblemInternal(Problem& problem)
{
	// Every three-row contact owns one block; scalar contacts own none.
	const int rows = int(problem.contacts.size()) + 2 * int(problem.contactBlocks.size());
	std::vector<int>& columnCounts = problem.columnCursors;
	problem.coupledContacts.clear();
	problem.equalityRows = 0;
	problem.hasFiniteBounds = !problem.patches.empty();
	problem.prepared = false;
	if(problem.massDiagonal.size() != problem.bodyCount() * 6 || !problem.massDiagonal.allFinite() ||
		(problem.massDiagonal.size() && problem.massDiagonal.minCoeff() <= 0.0))
		return false;
	int previousPatchEnd = 0;
	for(size_t i = 0; i < problem.patches.size(); ++i)
	{
		const Patch& patch = problem.patches[i];
		if(patch.normalCount <= 0 || patch.tangentCount < 0 || patch.tangentCount > 4 ||
			patch.firstContact < previousPatchEnd || patch.firstContact > int(problem.contacts.size()) ||
			patch.normalCount > int(problem.contacts.size()) - patch.firstContact - patch.tangentCount ||
			!std::isfinite(patch.friction) || patch.friction < 0.0)
			return false;
		previousPatchEnd = patch.firstContact + patch.normalCount + patch.tangentCount;
	}
	if(CountEntries)
	{
		columnCounts.assign(problem.bodyCount() * 6, 0);
		for(int i = 0; i < int(problem.contacts.size()); ++i)
		{
			const CompactContact& contact = problem.contacts[i];
			if(!validContact(problem, contact))
				return false;
			const int first = contact.rowCount() == 1 ? 2 : 0;
			for(int end = 0; end < 2; ++end)
				if(contact.body[end] >= 0)
					for(int column = 0; column < 6; ++column)
						for(int axis = first; axis < 3; ++axis)
							columnCounts[6 * contact.body[end] + column] += problem.contactEntry(contact, end, axis, column) != 0.0;
		}
	}
	if(columnCounts.size() != size_t(problem.bodyCount() * 6))
		return false;
	problem.freeVelocity.resize(rows);
	problem.regularization.resize(rows);
	reserveStorage(problem.rowContact, size_t(rows));
	problem.rowContact.resize(rows);
	// Rows are emitted in order and each constraint has distinct body endpoints.
	// Exact column capacities let us fill CSC storage without triplets, duplicate
	// reduction or an intermediate transpose.
	problem.jacobian.resize(rows, problem.bodyCount() * 6);
	int entries = 0;
	for(int column = 0; column < int(columnCounts.size()); ++column)
	{
		problem.jacobian.outerIndexPtr()[column] = entries;
		const int count = columnCounts[column];
		if(count < 0 || count > std::numeric_limits<int>::max() - entries)
			return false;
		columnCounts[column] = entries;
		entries += count;
	}
	problem.jacobian.outerIndexPtr()[columnCounts.size()] = entries;
	problem.jacobian.resizeNonZeros(entries);
	int nextRow = 0;
	size_t nextPatch = 0;
	for(int i = 0; i < int(problem.contacts.size()); ++i)
	{
		CompactContact& contact = problem.contacts[i];
		if(!CountEntries)
			if(!validContact(problem, contact))
				return false;
		while(nextPatch < problem.patches.size() && i >= problem.patches[nextPatch].firstContact +
			problem.patches[nextPatch].normalCount + problem.patches[nextPatch].tangentCount)
			++nextPatch;
		const Patch* patch = nextPatch < problem.patches.size() && i >= problem.patches[nextPatch].firstContact ?
			&problem.patches[nextPatch] : NULL;
		if(patch)
		{
			const CompactContact& first = problem.contacts[patch->firstContact];
			if(contact.rowCount() != 1 || contact.body[0] != first.body[0] || contact.body[1] != first.body[1])
				return false;
			if(i < patch->firstContact + patch->normalCount)
			{
				if(contact.hasScalarBounds() && (problem.bounds(contact).lower != 0.0 || problem.bounds(contact).upper < 0.0))
					return false;
			}
			else if(!contact.hasScalarBounds() || problem.bounds(contact).lower != -std::numeric_limits<double>::infinity() ||
				problem.bounds(contact).upper != std::numeric_limits<double>::infinity())
				return false;
		}
		contact.row = nextRow;
		nextRow += contact.rowCount();
		if(nextRow > rows)
			return false;
		if(contact.hasScalarBounds())
		{
			const ScalarBounds& limits = problem.bounds(contact);
			if(!patch && limits.lower == -std::numeric_limits<double>::infinity() &&
				limits.upper == std::numeric_limits<double>::infinity())
				++problem.equalityRows;
			else if(std::isfinite(limits.lower) || std::isfinite(limits.upper))
				problem.hasFiniteBounds = true;
		}
		else if(contact.block)
		{
			problem.block(contact).coupled = -1;
			if(contact.block < 0)
				problem.equalityRows += 3;
			else if(std::isfinite(problem.block(contact).maxNormalImpulse))
				problem.hasFiniteBounds = true;
		}
		if(contact.block > 0)
		{
			problem.block(contact).coupled = int(problem.coupledContacts.size());
			problem.coupledContacts.push_back(i);
		}
		const int begin = contact.rowCount() == 1 ? 2 : 0;
		for(int axis = begin; axis < 3; ++axis)
		{
			const int row = contact.row + axis - begin;
			problem.rowContact[row] = i;
			problem.freeVelocity[row] = axis == 2 ? contact.freeVelocity : problem.block(contact).freeVelocity[axis];
			problem.regularization[row] = axis == 2 ? contact.regularization : problem.block(contact).regularization[axis];
			if(!(problem.regularization[row] > 0.0) || !std::isfinite(problem.regularization[row]) ||
				!std::isfinite(problem.freeVelocity[row]))
				return false;
			for(int end = 0; end < 2; ++end)
				if(contact.body[end] >= 0)
					for(int column = 0; column < 6; ++column)
						if(problem.contactEntry(contact, end, axis, column) != 0.0)
						{
							const int entry = columnCounts[6 * contact.body[end] + column]++;
							problem.jacobian.innerIndexPtr()[entry] = row;
							problem.jacobian.valuePtr()[entry] = problem.contactEntry(contact, end, axis, column);
						}
		}
	}
	problem.prepared = nextRow == rows;
	if(problem.prepared)
		problem.preparationGeneration = nextPreparationGeneration();
	return problem.prepared;
}

template<bool CountEntries>
static SolveStatus::Enum prepareProblemSafe(Problem& problem) noexcept
{
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS)
	try
	{
#endif
		return prepareProblemInternal<CountEntries>(problem) ? SolveStatus::eSUCCESS : SolveStatus::eINVALID_INPUT;
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS)
	}
	catch(const std::bad_alloc&)
	{
		problem.prepared = false;
		return SolveStatus::eOUT_OF_MEMORY;
	}
	catch(...)
	{
		problem.prepared = false;
		return SolveStatus::eNUMERICAL_FAILURE;
	}
#endif
}

SolveStatus::Enum prepareProblem(Problem& problem) noexcept
{
	return prepareProblemSafe<true>(problem);
}

SolveStatus::Enum prepareProblemFromColumnCounts(Problem& problem) noexcept
{
	return prepareProblemSafe<false>(problem);
}

double computeResidual(const Problem& problem, ConstVector impulse)
{
	const Vector velocity = problem.jacobian * (problem.jacobian.transpose() * impulse) +
		problem.freeVelocity + problem.regularization.cwiseProduct(impulse);
	double error = 0.0;
	PatchScratch scratch;
	scratch.resize(problem);
	VectorStorage unitRegularization(scratch.velocity.size());
	unitRegularization.setOnes();
	size_t nextPatch = 0;
	for(int i = 0; i < int(problem.contacts.size()); ++i)
	{
		if(nextPatch < problem.patches.size() && i == problem.patches[nextPatch].firstContact)
		{
			const Patch& patch = problem.patches[nextPatch];
			const int count = patch.normalCount + patch.tangentCount;
			const int first = problem.contacts[i].row;
			double response = 0.0;
			for(int row = 0; row < count; ++row)
			{
				const CompactContact& contact = problem.contacts[i + row];
				for(int end = 0; end < 2; ++end)
					if(contact.body[end] >= 0)
						response += contact.jacobian[end].squaredNorm();
				if(row < patch.normalCount)
					scratch.cap[row] = contact.hasScalarBounds() ? problem.bounds(contact).upper : std::numeric_limits<double>::infinity();
			}
			response = std::max(response / count, 1.0e-12);
			for(int row = 0; row < count; ++row)
				scratch.velocity[row] = velocity[first + row] / response - impulse[first + row];
			for(int row = 0; row < patch.tangentCount; ++row)
				scratch.friction[row] = patch.friction;
			const PatchProjectionInput input = { patch.normalCount, patch.tangentCount, scratch.velocity.data(),
				unitRegularization.data(), scratch.cap.data(), scratch.velocity.data() + patch.normalCount,
				unitRegularization.data() + patch.normalCount, scratch.friction };
			const PatchProjectionOutput output = { scratch.impulse.data(), scratch.impulse.data() + patch.normalCount, NULL, NULL, NULL };
			PatchProjectionResult projection;
			if(!projectPatchUnchecked(input, output, projection))
				return std::numeric_limits<double>::infinity();
			for(int row = 0; row < count; ++row)
				error = std::max(error, response * std::abs(impulse[first + row] - scratch.impulse[row]));
			i += count - 1;
			++nextPatch;
			continue;
		}
		const CompactContact& c = problem.contacts[i];
		double jacobianNorm = 0.0;
		for(int end = 0; end < 2; ++end)
			if(c.body[end] >= 0)
				jacobianNorm += c.rowCount() == 1 ? c.jacobian[end].squaredNorm() : problem.contactJacobian(c, end).squaredNorm();
		const double response = std::max(jacobianNorm, 1.0e-12) / 3.0;
		if(c.rowCount() == 1)
		{
			const double trial = impulse[c.row] - velocity[c.row] / response;
			const double projected = c.hasScalarBounds() ?
				clampValue(trial, problem.bounds(c).lower, problem.bounds(c).upper) : std::max(0.0, trial);
			error = std::max(error, response * std::abs(impulse[c.row] - projected));
		}
		else
		{
			const Vec3 trial = impulse.segment<3>(c.row) - velocity.segment<3>(c.row) / response;
			const Vec3 projected = c.block < 0 ? trial : weightedImpulse(trial, Vec3::Ones(),
				problem.block(c).friction, NULL, false, problem.block(c).maxNormalImpulse);
			error = std::max(error, response * (impulse.segment<3>(c.row) - projected).cwiseAbs().maxCoeff());
		}
	}
	return error;
}

typedef std::pair<int, int> BodyPair;

typedef Eigen::Matrix<double, 6, 6> BodyBlock;

struct HessianStorage
{
	SparseStorage matrix;
	std::vector<BodyPair> pairs;

	std::vector<BodyBlock> blocks;
	std::vector<int> diagonalBlocks;
};

// Jacobian rows often contain exact zeros. Skip zero columns while keeping
// each six-value destination column contiguous for vectorized accumulation.
static void addOuterProduct(BodyBlock& block, const Vec6& left, const Vec6& right, double weight)
{
	for(int axis = 0; axis < 6; ++axis)
		if(right[axis] != 0.0)
			block.col(axis) += (weight * right[axis]) * left;
}

static const Sparse& makeHessian(const Problem& problem, const Curvature& weights, HessianStorage& storage)
{
	const int bodies = problem.bodyCount(), size = bodies * 6;
	std::vector<BodyPair>& pairs = storage.pairs;
	pairs.clear();
	reserveStorage(pairs, bodies + problem.contacts.size());
	for(int body = 0; body < bodies; ++body)
		pairs.push_back(BodyPair(body, body));
	for(size_t i = 0; i < problem.contacts.size(); ++i)
	{
		const CompactContact& contact = problem.contacts[i];
		if(contact.body[0] >= 0 && contact.body[1] >= 0)
		{
			const BodyPair pair(std::min(contact.body[0], contact.body[1]), std::max(contact.body[0], contact.body[1]));
			if(pairs.back() != pair)
				pairs.push_back(pair);
		}
	}
	std::sort(pairs.begin(), pairs.end());
	pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
	reserveStorage(storage.blocks, pairs.size());
	storage.blocks.resize(pairs.size());
	storage.diagonalBlocks.resize(bodies);
	for(size_t pair = 0; pair < pairs.size(); ++pair)
	{
		BodyBlock& block = storage.blocks[pair];
		if(pairs[pair].first == pairs[pair].second)
		{
			block.setIdentity();
			storage.diagonalBlocks[pairs[pair].first] = int(pair);
		}
		else
			block.setZero();
	}
	BodyPair previous(-1, -1);
	int cross = -1;
	for(size_t i = 0; i < problem.contacts.size(); ++i)
	{
		const CompactContact& contact = problem.contacts[i];
		const int a = contact.body[0], b = contact.body[1];
		if(a >= 0 && b >= 0)
		{
			const BodyPair pair(std::min(a, b), std::max(a, b));
			if(pair != previous)
			{
				cross = int(std::lower_bound(pairs.begin(), pairs.end(), pair) - pairs.begin());
				previous = pair;
			}
		}
		if(contact.block > 0)
		{
			const Mat3& weight = weights.coupled[problem.coupledIndex(contact)];
			if(weight.isZero(0.0))
				continue;
			// Reconstruct each used endpoint once, then reuse it for diagonal
			// and cross contributions. Scalar rows need no reconstruction.
			Jacobian jacobian[2];
			for(int end = 0; end < 2; ++end)
				if(contact.body[end] >= 0)
				{
					jacobian[end] = problem.contactJacobian(contact, end);
					storage.blocks[storage.diagonalBlocks[contact.body[end]]].noalias() +=
						jacobian[end].transpose() * weight * jacobian[end];
				}
			if(a >= 0 && b >= 0)
			{
				const int low = a < b ? 0 : 1;
				storage.blocks[cross].noalias() += jacobian[1 - low].transpose() * weight * jacobian[low];
			}
		}
		else
		{
			const int first = contact.rowCount() == 1 ? 2 : 0;
			for(int axis = first; axis < 3; ++axis)
			{
				const double weight = weights.diagonal[contact.row + axis - first];
				if(weight == 0.0)
					continue;
				Vec6 jacobian[2];
				for(int end = 0; end < 2; ++end)
					if(contact.body[end] >= 0)
					{
						jacobian[end] = problem.contactRow(contact, end, axis);
						addOuterProduct(storage.blocks[storage.diagonalBlocks[contact.body[end]]],
							jacobian[end], jacobian[end], weight);
					}
				if(a >= 0 && b >= 0)
				{
					const int low = a < b ? 0 : 1;
					addOuterProduct(storage.blocks[cross], jacobian[1 - low], jacobian[low], weight);
				}
			}
		}
	}
	for(size_t i = 0; i < problem.patches.size(); ++i)
	{
		const CompactContact& contact = problem.contacts[problem.patches[i].firstContact];
		const PatchCurvature& patch = weights.patches[i];
		for(int left = 0; left < 2; ++left)
			for(int right = 0; right < 2; ++right)
			{
				const int a = contact.body[left], b = contact.body[right];
				if(a < 0 || b < 0 || a < b)
					continue;
				const int index = a == b ? storage.diagonalBlocks[a] :
					int(std::lower_bound(pairs.begin(), pairs.end(), BodyPair(b, a)) - pairs.begin());
				BodyBlock& block = storage.blocks[index];
				addOuterProduct(block, patch.normal[left], patch.normal[right], patch.normalCoefficient);
				addOuterProduct(block, patch.normal[left], patch.tangent[right], patch.crossCoefficient);
				addOuterProduct(block, patch.tangent[left], patch.normal[right], patch.crossCoefficient);
				addOuterProduct(block, patch.tangent[left], patch.tangent[right], patch.tangentCoefficient);
			}
	}
	SparseStorage& matrix = storage.matrix;
	matrix.resize(size, size);
	// Sorted body pairs emit 21 entries per diagonal block and 36 per
	// off-diagonal block, including structural zeros required by rank updates.
	matrix.resizeNonZeros(int(pairs.size()) * 36 - bodies * 15);
	int* outer = matrix.outerIndexPtr();
	int* inner = matrix.innerIndexPtr();
	double* values = matrix.valuePtr();
	int entry = 0;
	size_t begin = 0;
	for(int body = 0; body < bodies; ++body)
	{
		size_t end = begin;
		while(end < pairs.size() && pairs[end].first == body)
			++end;
		for(int axis = 0; axis < 6; ++axis)
		{
			outer[6 * body + axis] = entry;
			for(size_t pair = begin; pair < end; ++pair)
			{
				const int first = pairs[pair].second == body ? axis : 0;
				for(int row = first; row < 6; ++row)
				{
					inner[entry] = pairs[pair].second * 6 + row;
					values[entry++] = storage.blocks[pair](row, axis);
				}
			}
		}
		begin = end;
	}
	outer[size] = entry;
	return matrix;
}

// Independent assembly for factor audits. This deliberately uses sparse
// products rather than the production body's block-address accumulation.
static Sparse makeReferenceHessian(const Problem& problem, const Curvature& weights)
{
	std::vector<Triplet> entries;
	for(int row = 0; row < problem.rowCount(); ++row)
		entries.push_back(Triplet(row, row, weights.diagonal[row]));
	for(int block = 0; block < int(weights.coupled.size()); ++block)
	{
		const int first = problem.contacts[problem.coupledContacts[block]].row;
		for(int row = 0; row < 3; ++row)
			for(int column = 0; column < 3; ++column)
				entries.push_back(Triplet(first + row, first + column, weights.coupled[block](row, column)));
	}
	for(size_t i = 0; i < problem.patches.size(); ++i)
	{
		const Patch& group = problem.patches[i];
		const PatchCurvature& patch = weights.patches[i];
		const int first = problem.contacts[group.firstContact].row;
		const int count = group.normalCount + group.tangentCount;
		for(int row = 0; row < count; ++row)
			for(int column = 0; column < count; ++column)
			{
				const bool normalRow = row < group.normalCount, normalColumn = column < group.normalCount;
				const double left = normalRow ? weights.diagonal[first + row] : patch.tangentCoupling[row - group.normalCount];
				const double right = normalColumn ? weights.diagonal[first + column] : patch.tangentCoupling[column - group.normalCount];
				const double coefficient = normalRow && normalColumn ? patch.normalCoefficient :
					(!normalRow && !normalColumn ? patch.tangentCoefficient : patch.crossCoefficient);
				entries.push_back(Triplet(first + row, first + column, coefficient * left * right));
			}
	}
	Sparse curvature(problem.rowCount(), problem.rowCount());
	curvature.setFromTriplets(entries.begin(), entries.end());
	Sparse identity(problem.bodyCount() * 6, problem.bodyCount() * 6);
	identity.setIdentity();
	return problem.jacobian.transpose() * curvature * problem.jacobian + identity;
}

static bool evaluateImpulses(const Problem& problem, ConstVector contactVelocity, ConstVector inverseRoot,
	MutableVector impulse, Curvature* weights, PatchScratch& scratch)
{
	ConstVector compliance = problem.regularization;
	if(problem.isUnilateral())
	{
		impulse = inverseRoot.cwiseProduct(inverseRoot.cwiseProduct(-contactVelocity).cwiseMax(0.0));
		if(weights)
			weights->diagonal.array() = (contactVelocity.array() < 0.0).select(inverseRoot.array().square(), 0.0);
		return true;
	}
	size_t nextPatch = 0;
	for(int i = 0; i < int(problem.contacts.size()); ++i)
	{
		if(nextPatch < problem.patches.size() && i == problem.patches[nextPatch].firstContact)
		{
			const Patch& group = problem.patches[nextPatch];
			const int first = problem.contacts[i].row;
			PatchProjectionResult projection;
			if(!projectGroup(problem, group, contactVelocity.data() + first, impulse.data() + first,
				weights ? weights->diagonal.data() + first : NULL, weights ? scratch.coupling : NULL, scratch, projection))
				return false;
			if(weights)
			{
				PatchCurvature& patch = weights->patches[nextPatch];
				patch.normalCoefficient = -projection.boundedTangentRegularization * projection.inverseCoupling;
				patch.crossCoefficient = -projection.inverseCoupling;
				patch.tangentCoefficient = projection.normalInverseRegularization * projection.inverseCoupling;
				for(int end = 0; end < 2; ++end)
				{
					patch.normal[end].setZero();
					patch.tangent[end].setZero();
					if(problem.contacts[i].body[end] < 0)
						continue;
					for(int row = 0; row < group.normalCount; ++row)
						patch.normal[end] += weights->diagonal[first + row] * problem.contacts[i + row].jacobian[end];
					for(int row = 0; row < group.tangentCount; ++row)
						patch.tangent[end] += scratch.coupling[row] * problem.contacts[i + group.normalCount + row].jacobian[end];
				}
				for(int row = 0; row < group.tangentCount; ++row)
					patch.tangentCoupling[row] = scratch.coupling[row];
			}
			i += group.normalCount + group.tangentCount - 1;
			++nextPatch;
			continue;
		}
		const CompactContact& contact = problem.contacts[i];
		const int row = contact.row;
		if(contact.rowCount() == 1)
		{
			const double value = -contactVelocity[row];
			if(contact.hasScalarBounds())
			{
				const ScalarBounds& limits = problem.bounds(contact);
				const double unconstrained = value / compliance[row];
				impulse[row] = clampValue(unconstrained, limits.lower, limits.upper);
				if(weights)
					weights->diagonal[row] = unconstrained > limits.lower && unconstrained < limits.upper ?
						1.0 / compliance[row] : 0.0;
			}
			else
			{
				impulse[row] = inverseRoot[row] * std::max(0.0, inverseRoot[row] * value);
				if(weights)
					weights->diagonal[row] = value > 0.0 ? inverseRoot[row] * inverseRoot[row] : 0.0;
			}
		}
		else if(contact.block < 0)
		{
			impulse.segment<3>(row) = (-contactVelocity.segment<3>(row)).cwiseQuotient(compliance.segment<3>(row));
			if(weights)
				weights->diagonal.segment<3>(row) = compliance.segment<3>(row).cwiseInverse();
		}
		else
			impulse.segment<3>(row) = weightedImpulse(-contactVelocity.segment<3>(row),
				compliance.segment<3>(row), problem.block(contact).friction, weights ? &weights->coupled[problem.coupledIndex(contact)] : NULL, false, problem.block(contact).maxNormalImpulse);
	}
	return true;
}

// These calls use distinct solver-owned input/output buffers. The prepared
// Jacobian is compressed CSC with strictly increasing row indices per column.
static void multiplyJacobianCsc(const Problem& problem, ConstVector vector, MutableVector product)
{
	const double* EIGEN_RESTRICT values = problem.jacobian.valuePtr();
	const int* EIGEN_RESTRICT indices = problem.jacobian.innerIndexPtr();
	const int* EIGEN_RESTRICT offsets = problem.jacobian.outerIndexPtr();
	const double* EIGEN_RESTRICT input = vector.data();
	double* EIGEN_RESTRICT output = product.data();
	std::fill_n(output, product.size(), 0.0);
	for(int column = 0; column < problem.jacobian.cols(); ++column)
	{
		const double scale = input[column];
		const int end = offsets[column + 1];
		int entry = offsets[column];
		for(; entry < end; ++entry)
			output[indices[entry]] += values[entry] * scale;
	}
}

static void evaluatePrimalGradientCsc(const Problem& problem, ConstVector velocity,
	ConstVector impulse, MutableVector gradient)
{
	const double* EIGEN_RESTRICT values = problem.jacobian.valuePtr();
	const int* EIGEN_RESTRICT indices = problem.jacobian.innerIndexPtr();
	const int* EIGEN_RESTRICT offsets = problem.jacobian.outerIndexPtr();
	const double* EIGEN_RESTRICT input = impulse.data();
	const double* EIGEN_RESTRICT primal = velocity.data();
	double* EIGEN_RESTRICT output = gradient.data();
	for(int column = 0; column < problem.jacobian.cols(); ++column)
	{
		double value = 0.0;
		for(int entry = offsets[column]; entry < offsets[column + 1]; ++entry)
			value += values[entry] * input[indices[entry]];
		// Fuse only the final store, retaining the complete ordered dot product.
		output[column] = primal[column] - value;
	}
}

static bool evaluatePrimal(const Problem& problem, ConstVector velocity, ConstVector inverseRoot,
	MutableVector contactVelocity, MutableVector impulse, MutableVector gradient, Curvature* weights, PatchScratch& scratch)
{
	multiplyJacobianCsc(problem, velocity, contactVelocity);
	contactVelocity += problem.freeVelocity;
	if(!evaluateImpulses(problem, contactVelocity, inverseRoot, impulse, weights, scratch))
		return false;
	evaluatePrimalGradientCsc(problem, velocity, impulse, gradient);
	return true;
}

#include "IncrementalCholesky.h"

static double primalCost(const Problem& problem, ConstVector velocity, ConstVector contactVelocity, ConstVector impulse)
{
	const double quadratic = impulse.dot(problem.regularization.cwiseProduct(impulse));
	if(!problem.hasFiniteBounds)
		return 0.5 * (velocity.squaredNorm() + quadratic);
	return 0.5 * velocity.squaredNorm() - contactVelocity.dot(impulse) - 0.5 * quadratic;
}

struct NewtonStopping
{
	double tolerance;
	double lineTolerance;
	double inertiaSum;
	double costScale;

	NewtonStopping(const Problem& problem, const Settings& settings)
		: tolerance(settings.tolerance), lineTolerance(settings.lineTolerance),
		inertiaSum(problem.massDiagonal.sum()),
		costScale(1.0 / (inertiaSum * problem.timestep * problem.timestep)) {}

	double gradientNorm(const Problem& problem, ConstVector gradient) const
	{
		return std::sqrt(gradient.array().square().matrix().dot(problem.massDiagonal)) /
			(inertiaSum * problem.timestep);
	}

	double lineThreshold(const Problem& problem, ConstVector direction) const
	{
		return tolerance * lineTolerance * problem.timestep * inertiaSum *
			std::sqrt((direction.array().square() / problem.massDiagonal.array()).sum());
	}
};

struct ConvexLineValue
{
	double slope;
	double curvature;
};

static ConvexLineValue evaluateConvexLine(const Problem& problem, ConstVector contactVelocity, ConstVector contactDirection,
	ConstVector compliance, ConstVector inverseRoot, double velocitySlope, double directionNorm, double alpha, PatchScratch& scratch)
{
	ConvexLineValue result = { velocitySlope + alpha * directionNorm, directionNorm };
	if(problem.isUnilateral())
	{
		for(int row = 0; row < problem.rowCount(); ++row)
		{
			const double direction = contactDirection[row];
			const double drive = -contactVelocity[row] - alpha * direction;
			if(drive > 0.0)
			{
				const double root = inverseRoot[row];
				result.slope -= direction * (root * (root * drive));
				result.curvature += direction * (root * root * direction);
			}
		}
		return result;
	}
	size_t nextPatch = 0;
	for(int i = 0; i < int(problem.contacts.size()); ++i)
	{
		if(nextPatch < problem.patches.size() && i == problem.patches[nextPatch].firstContact)
		{
			const Patch& group = problem.patches[nextPatch];
			const int first = problem.contacts[i].row;
			const int count = group.normalCount + group.tangentCount;
			for(int row = 0; row < count; ++row)
				scratch.velocity[row] = contactVelocity[first + row] + alpha * contactDirection[first + row];
			PatchProjectionResult projection;
			if(!projectGroup(problem, group, scratch.velocity.data(), scratch.impulse.data(), scratch.diagonal.data(),
				scratch.coupling, scratch, projection))
			{
				result.slope = result.curvature = std::numeric_limits<double>::quiet_NaN();
				return result;
			}
			double normalDirection = 0.0, tangentDirection = 0.0;
			for(int row = 0; row < count; ++row)
			{
				const double direction = contactDirection[first + row];
				result.slope -= direction * scratch.impulse[row];
				result.curvature += direction * direction * scratch.diagonal[row];
				if(row < group.normalCount)
					normalDirection += direction * scratch.diagonal[row];
				else
					tangentDirection += direction * scratch.coupling[row - group.normalCount];
			}
			result.curvature += projection.inverseCoupling *
				(-projection.boundedTangentRegularization * normalDirection * normalDirection -
				2.0 * normalDirection * tangentDirection + projection.normalInverseRegularization * tangentDirection * tangentDirection);
			i += count - 1;
			++nextPatch;
			continue;
		}
		const CompactContact& contact = problem.contacts[i];
		const int row = contact.row;
		if(contact.rowCount() == 1)
		{
			const double direction = contactDirection[row];
			const double drive = -contactVelocity[row] - alpha * direction;
			if(contact.hasScalarBounds())
			{
				const ScalarBounds& limits = problem.bounds(contact);
				const double unconstrained = drive / compliance[row];
				result.slope -= direction * clampValue(unconstrained, limits.lower, limits.upper);
				if(unconstrained > limits.lower && unconstrained < limits.upper)
					result.curvature += direction * direction / compliance[row];
			}
			else if(drive > 0.0)
			{
				const double root = inverseRoot[row];
				result.slope -= direction * (root * (root * drive));
				result.curvature += direction * (root * root * direction);
			}
		}
		else
		{
			const Vec3 direction = contactDirection.segment<3>(row);
			Mat3 derivative;
			const Vec3 impulse = weightedImpulse(-contactVelocity.segment<3>(row) - alpha * direction,
				compliance.segment<3>(row), problem.block(contact).friction, &derivative, contact.block < 0, problem.block(contact).maxNormalImpulse);
			result.slope -= direction.dot(impulse);
			result.curvature += direction.dot(derivative * direction);
		}
	}

	return result;
}

static double searchConvex(const Problem& problem, ConstVector velocity, ConstVector direction,
	ConstVector contactVelocity, ConstVector inverseRoot, MutableVector contactDirection, int& evaluations, PatchScratch& scratch, double slopeTolerance = 0.0)
{
	// Reuse the contact velocity from the current objective/gradient evaluation.
	ConstVector compliance = problem.regularization;
	multiplyJacobianCsc(problem, direction, contactDirection);
	const double velocitySlope = velocity.dot(direction), directionNorm = direction.squaredNorm();
	const ConvexLineValue initial = evaluateConvexLine(problem, contactVelocity, contactDirection, compliance, inverseRoot,
		velocitySlope, directionNorm, 0.0, scratch);
	++evaluations;
	if(!std::isfinite(initial.slope))
		return initial.slope;
	if(initial.slope >= 0.0)
		return 0.0;
	double lower = 0.0, upper = 1.0;
	ConvexLineValue atUpper = evaluateConvexLine(problem, contactVelocity, contactDirection, compliance, inverseRoot,
		velocitySlope, directionNorm, upper, scratch);
	++evaluations;
	if(!std::isfinite(atUpper.slope))
		return atUpper.slope;
	for(int i = 0; atUpper.slope < 0.0 && i < 50; ++i)
	{
		lower = upper;
		upper *= 2.0;
		atUpper = evaluateConvexLine(problem, contactVelocity, contactDirection, compliance, inverseRoot,
			velocitySlope, directionNorm, upper, scratch);
		++evaluations;
		if(!std::isfinite(atUpper.slope))
			return atUpper.slope;
	}
	double alpha = clampValue(-initial.slope / initial.curvature, lower, upper);
	for(int i = 0; i < 50; ++i)
	{
		const ConvexLineValue value = evaluateConvexLine(problem, contactVelocity, contactDirection, compliance, inverseRoot,
			velocitySlope, directionNorm, alpha, scratch);
		++evaluations;
		if(!std::isfinite(value.slope))
			return value.slope;
		if(std::abs(value.slope) <= (slopeTolerance > 0.0 ? slopeTolerance :
			1.0e-10 * std::max(1.0, std::abs(initial.slope))))
			return alpha;
		if(value.slope < 0.0)
			lower = alpha;
		else
			upper = alpha;
		const double candidate = alpha - value.slope / value.curvature;
		alpha = candidate > lower && candidate < upper ? candidate : 0.5 * (lower + upper);
	}
	return alpha;
}

struct WorkspaceData
{
	VectorStorage velocity, impulse, gradient, contactVelocity, inverseRoot, coldImpulse, nextVelocity, direction, contactDirection;
	Curvature weights;
	PatchScratch patchScratch;
	IncrementalCholesky factor;
	const Problem* continuationProblem = NULL;
	std::uint64_t continuationGeneration = 0;
};

static SolveStatus::Enum solveNewtonInternal(const Problem& problem, const Settings& settings, Result& result,
	WorkspaceData& workspace, const Result* previous, bool continuation)
{
	static_cast<SolverStatistics&>(result) = SolverStatistics();
	const int expectedSize = problem.bodyCount() * 6;
	if(!problem.prepared || !(problem.timestep > 0.0) || !std::isfinite(problem.timestep) ||
		settings.iterations < 0 || !(settings.tolerance >= 0.0) || !std::isfinite(settings.tolerance) ||
		!(settings.lineTolerance >= 0.0) || !std::isfinite(settings.lineTolerance) ||
		problem.massDiagonal.size() != expectedSize || problem.jacobian.cols() != expectedSize ||
		problem.jacobian.rows() != problem.rowCount() || problem.regularization.size() != problem.rowCount() ||
		(previous && previous->primal.size() != expectedSize))
		return SolveStatus::eINVALID_INPUT;
	if(expectedSize == 0)
	{
		if(problem.rowCount() != 0)
			return SolveStatus::eINVALID_INPUT;
		result.primal.resize(0);
		result.impulse.resize(0);
		result.stopReason = 1;
		return SolveStatus::eSUCCESS;
	}
	ConstVector compliance = problem.regularization;
	const NewtonStopping stopping(problem, settings);
	// Reevaluate curvature each call; an explicit continuation may update the retained factor.
	VectorStorage& velocity = workspace.velocity;
	VectorStorage& impulse = workspace.impulse;
	VectorStorage& gradient = workspace.gradient;
	VectorStorage& contactVelocity = workspace.contactVelocity;
	VectorStorage& inverseRoot = workspace.inverseRoot;
	VectorStorage& nextVelocity = workspace.nextVelocity;
	VectorStorage& direction = workspace.direction;
	Curvature& weights = workspace.weights;
	IncrementalCholesky& factor = workspace.factor;
	const int bodies = problem.bodyCount() * 6, rows = problem.rowCount();
	velocity.resize(bodies);
	gradient.resize(bodies);
	nextVelocity.resize(bodies);
	direction.resize(bodies);
	contactVelocity.resize(rows);
	workspace.contactDirection.resize(rows);
	if(previous)
		velocity = previous->primal;
	else
		velocity.setZero();
	impulse.resize(problem.rowCount());
	weights.resize(problem);
	workspace.patchScratch.resize(problem);
	inverseRoot = compliance.cwiseSqrt().cwiseInverse();
	factor.beginSolve(settings.profile, continuation, settings.parallelExecutor);
	Clock::time_point evaluationStart = profileStart(settings.profile);
	if(!evaluatePrimal(problem, velocity, inverseRoot, contactVelocity, impulse, gradient, &weights, workspace.patchScratch))
		return SolveStatus::eNUMERICAL_FAILURE;
	if(previous)
	{
		// A cached solution can be worse than free motion after contacts or loads change.
		VectorStorage& coldImpulse = workspace.coldImpulse;
		coldImpulse.resize(problem.rowCount());
		// Comparing warm and free motion needs only the cold impulses and cost.
		if(!evaluateImpulses(problem, problem.freeVelocity, inverseRoot, coldImpulse, NULL, workspace.patchScratch))
			return SolveStatus::eNUMERICAL_FAILURE;
		double warmCost = velocity.squaredNorm() + impulse.dot(compliance.cwiseProduct(impulse));
		double coldCost = coldImpulse.dot(compliance.cwiseProduct(coldImpulse));
		if(problem.hasFiniteBounds)
		{
			warmCost = 2.0 * primalCost(problem, velocity, contactVelocity, impulse);
			coldCost = -2.0 * problem.freeVelocity.dot(coldImpulse) - coldCost;
		}
		if(warmCost > coldCost)
		{
			velocity.setZero();
			if(!evaluatePrimal(problem, velocity, inverseRoot, contactVelocity, impulse, gradient, &weights, workspace.patchScratch))
				return SolveStatus::eNUMERICAL_FAILURE;
		}
	}
	result.evaluationMs += profileElapsed(settings.profile, evaluationStart);
	double cost = primalCost(problem, velocity, contactVelocity, impulse);
	if(!std::isfinite(cost) || !gradient.allFinite())
		return SolveStatus::eNUMERICAL_FAILURE;
	for(int iteration = 0; iteration < settings.iterations; ++iteration)
	{
		result.scaledGradient = stopping.gradientNorm(problem, gradient);
		if(result.scaledGradient < stopping.tolerance)
		{
			result.stopReason = 1;
			break;
		}
		if(!factor.factor(problem, weights, result))
			return SolveStatus::eFACTORIZATION_FAILED;
		const Clock::time_point solveStart = profileStart(settings.profile);
		factor.solveDirection(gradient, direction);
		if(!direction.allFinite())
			return SolveStatus::eNUMERICAL_FAILURE;
		result.backsolveMs += profileElapsed(settings.profile, solveStart);
		if(settings.checkFactor)
		{
			// This independent check is outside performance runs. It catches errors
			// in signed updates and permutation by rebuilding the current Hessian.
			const Sparse matrix = makeReferenceHessian(problem, weights);
			Factor reference;
			reference.compute(matrix);
			const Vector expected = reference.solve(-gradient);
			const double error = (direction - expected).lpNorm<Eigen::Infinity>() /
				std::max(1.0, expected.lpNorm<Eigen::Infinity>());
			result.factorError = std::max(result.factorError, error);
			if(!std::isfinite(error) || error > 1.0e-6)
				return SolveStatus::eNUMERICAL_FAILURE;
		}
		// Very stiff constraints can leave a cancellation residual after motion
		// has converged. Stop when the computed correction reaches roundoff.
		if(direction.lpNorm<Eigen::Infinity>() <= 1.0e-14 * std::max(1.0, velocity.lpNorm<Eigen::Infinity>()))
		{
			result.stopReason = 5;
			break;
		}
		if(gradient.dot(direction) >= 0.0)
		{
			direction = -gradient;
		}
		const Clock::time_point lineStart = profileStart(settings.profile);
		const double alpha = searchConvex(problem, velocity, direction, contactVelocity, inverseRoot, workspace.contactDirection, result.lineSearchEvaluations, workspace.patchScratch,
			stopping.lineThreshold(problem, direction));
		result.lineSearchMs += profileElapsed(settings.profile, lineStart);
		if(!std::isfinite(alpha))
			return SolveStatus::eNUMERICAL_FAILURE;
		nextVelocity = velocity + alpha * direction;
		if((nextVelocity.array() == velocity.array()).all())
		{
			result.stopReason = 6;
			break;
		}
		velocity.swap(nextVelocity);
		result.iterations = iteration + 1;
		evaluationStart = profileStart(settings.profile);
		if(!evaluatePrimal(problem, velocity, inverseRoot, contactVelocity, impulse, gradient, &weights, workspace.patchScratch))
			return SolveStatus::eNUMERICAL_FAILURE;
		result.evaluationMs += profileElapsed(settings.profile, evaluationStart);
		const double nextCost = primalCost(problem, velocity, contactVelocity, impulse);
		if(!std::isfinite(nextCost) || !gradient.allFinite())
			return SolveStatus::eNUMERICAL_FAILURE;
		result.scaledImprovement = stopping.costScale * (cost - nextCost);
		cost = nextCost;
		if(result.scaledImprovement > 0.0 && result.scaledImprovement < stopping.tolerance)
		{
			result.stopReason = 2;
			break;
		}
	}
	result.impulse = impulse;
	result.primal = velocity;
	result.gradientResidual = gradient.lpNorm<Eigen::Infinity>();
	result.scaledGradient = stopping.gradientNorm(problem, gradient);
	if(!std::isfinite(result.scaledGradient))
		return SolveStatus::eNUMERICAL_FAILURE;
	return result.stopReason == 0 ? SolveStatus::eITERATION_LIMIT : SolveStatus::eSUCCESS;
}

Workspace::Workspace() noexcept : m_data(NULL) {}
Workspace::~Workspace() { delete m_data; }
Workspace::Workspace(Workspace&& other) noexcept : m_data(other.m_data) { other.m_data = NULL; }
Workspace& Workspace::operator=(Workspace&& other) noexcept
{
	if(this != &other)
	{
		delete m_data;
		m_data = other.m_data;
		other.m_data = NULL;
	}
	return *this;
}

static SolveStatus::Enum solveNewtonSafe(const Problem& problem, const Settings& settings, Result& result,
	WorkspaceData*& data, const Result* previous, bool continuation) noexcept
{
	const Clock::time_point start = Clock::now();
	result.status = SolveStatus::eINVALID_INPUT;
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS)
	try
	{
#endif
		if(!data)
			data = new(std::nothrow) WorkspaceData;
		if(data)
		{
			const bool reuse = continuation && data->continuationProblem == &problem &&
				data->continuationGeneration == problem.preparationGeneration;
			// Invalidate before any potentially failing operation. An error may
			// leave a partial rank update/factor, which must never be continued.
			data->continuationProblem = NULL;
			data->continuationGeneration = 0;
			result.status = solveNewtonInternal(problem, settings, result, *data, previous, reuse);
			if(result.status == SolveStatus::eSUCCESS || result.status == SolveStatus::eITERATION_LIMIT)
			{
				data->continuationProblem = &problem;
				data->continuationGeneration = problem.preparationGeneration;
			}
		}
		else
			result.status = SolveStatus::eOUT_OF_MEMORY;
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS)
	}
	catch(const std::bad_alloc&)
	{
		result.status = SolveStatus::eOUT_OF_MEMORY;
	}
	catch(...)
	{
		result.status = SolveStatus::eNUMERICAL_FAILURE;
	}
#endif
	result.elapsedMs = elapsed(start);
	return result.status;
}

SolveStatus::Enum solveNewton(const Problem& problem, const Settings& settings, Result& result,
	Workspace& workspace, const Result* previous) noexcept
{
	return solveNewtonSafe(problem, settings, result, workspace.m_data, previous, false);
}

SolveStatus::Enum continueNewton(const Problem& problem, const Settings& settings, Result& result,
	Workspace& workspace, const Result* previous) noexcept
{
	return solveNewtonSafe(problem, settings, result, workspace.m_data, previous, true);
}

}
