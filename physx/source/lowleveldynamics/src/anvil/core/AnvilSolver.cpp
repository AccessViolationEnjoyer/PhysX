#include "AnvilSolver.h"
#include <atomic>
#include <cassert>
#include "AnvilPatchProjection.h"
#include <algorithm>
#include <cmath>
#include <limits>

#include "StorageCholesky.h"
#include "BlockCholesky.h"

namespace anvil
{
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
		{
			derivative->setIdentity();
		}
		return value;
	}
	if(value[2] <= -friction * (a + b))
	{
		if(derivative)
		{
			derivative->setZero();
		}
		return Vec3::Zero();
	}
	double normal = (value[2] + friction * (a + b)) / (1.0 + 2.0 * friction * friction);
	if(friction * normal > std::min(a, b))
	{
		normal = (value[2] + friction * std::max(a, b)) / (1.0 + friction * friction);
	}
	Vec3 result = value;
	result[2] = normal;
	for(int axis = 0; axis < 2; ++axis)
	{
		result[axis] = clampValue(value[axis], -friction * normal, friction * normal);
	}
	if(derivative)
	{
		Vec3 direction(0.0, 0.0, 1.0);
		derivative->setZero();
		for(int i = 0; i < 2; ++i)
		{
			if(std::abs(value[i]) > friction * normal)
			{
				direction[i] = std::copysign(friction, value[i]);
			}
			else
			{
				(*derivative)(i, i) = 1.0;
			}
		}
		*derivative += direction * direction.transpose() / direction.squaredNorm();
	}
	return result;
}

static Vec3 weightedImpulse(const Vec3& drive, const double* ANVIL_RESTRICT compliance, const double* ANVIL_RESTRICT inverseRootValues, double friction, Mat3* derivative = NULL, bool bilateral = false, double maxNormalImpulse = MAX_IMPULSE)
{
	const Vec3 inverseRoot = loadVector<3>(inverseRootValues);
	if(bilateral)
	{
		if(derivative)
		{
			derivative->setZero();
			for(int axis = 0; axis < 3; ++axis)
			{
				(*derivative)(axis, axis) = inverseRoot[axis] * inverseRoot[axis];
			}
		}
		return drive.cwiseProduct(inverseRoot).cwiseProduct(inverseRoot);
	}

	const double scaledFriction = friction * compliance[0] * inverseRoot[0] * inverseRoot[2];
	Mat3 projectionDerivative;
	Vec3 impulse = inverseRoot.cwiseProduct(projectPyramid(inverseRoot.cwiseProduct(drive), scaledFriction, derivative ? &projectionDerivative : NULL));
	if(derivative)
	{
		for(int column = 0; column < 3; ++column)
		{
			for(int row = 0; row < 3; ++row)
			{
				(*derivative)(row, column) = inverseRoot[row] * projectionDerivative(row, column) * inverseRoot[column];
			}
		}
	}
	if(impulse[2] > maxNormalImpulse)
	{
		impulse[2] = maxNormalImpulse;
		const double tangentLimit = friction * maxNormalImpulse;
		if(derivative)
		{
			derivative->setZero();
		}
		for(int axis = 0; axis < 2; ++axis)
		{
			const double inverseCompliance = inverseRoot[axis] * inverseRoot[axis];
			const double unconstrained = drive[axis] * inverseCompliance;
			impulse[axis] = clampValue(unconstrained, -tangentLimit, tangentLimit);
			if(derivative && unconstrained > -tangentLimit && unconstrained < tangentLimit)
			{
				(*derivative)(axis, axis) = inverseCompliance;
			}
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
		const std::uint32_t patchCount = std::uint32_t(problem.patches.size());
		for(std::uint32_t i = 0; i < patchCount; ++i)
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

static bool projectGroup(const Problem& problem, const Patch& patch, const double* velocity, double* impulse, double* diagonal, double* coupling, PatchScratch& scratch, PatchProjectionResult& result)
{
	const int firstRow = problem.contacts[patch.firstContact].row;
	for(int i = 0; i < patch.normalCount; ++i)
	{
		const CompactContact& contact = problem.contacts[patch.firstContact + i];
		scratch.cap[i] = contact.hasScalarBounds() ? problem.bounds(contact).upper : MAX_IMPULSE;
	}
	for(int i = 0; i < patch.tangentCount; ++i)
	{
		scratch.friction[i] = patch.friction;
	}
	const PatchProjectionInput input = { patch.normalCount, patch.tangentCount, velocity, problem.regularization.data() + firstRow, scratch.cap.data(), velocity + patch.normalCount, problem.regularization.data() + firstRow + patch.normalCount, scratch.friction };
	const PatchProjectionOutput output = { impulse, impulse + patch.normalCount, diagonal, diagonal ? diagonal + patch.normalCount : NULL, coupling };
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
		reserveStorage(patches, std::uint32_t(problem.patches.size()));
		patches.resize(problem.patches.size());
	}
	void swap(Curvature& other) noexcept
	{
		diagonal.swap(other.diagonal);
		coupled.swap(other.coupled);
		patches.swap(other.patches);
	}
};

// Distinguish every successful preparation, including a new Problem constructed
// at a previously used address. Each Problem takes a unique base once and then
// counts its own preparations, so concurrent islands share no counter; this avoids
// scanning immutable coefficients during same-problem continuation.
std::uint64_t Problem::uniquePreparationBase() noexcept
{
	static std::atomic<std::uint64_t> base(0);
	return (base.fetch_add(1, std::memory_order_relaxed) + 1) << 32;
}

void Problem::setScalarBounds(int contactIndex, double lowerImpulse, double upperImpulse) noexcept
{
	assert(contactIndex >= 0 && contactIndex < int(contacts.size()));
	assert(lowerImpulse <= upperImpulse && lowerImpulse != MAX_IMPULSE && upperImpulse != -MAX_IMPULSE);
	assert(contacts[contactIndex].hasScalarBounds());
#ifndef NDEBUG
	const std::uint32_t patchCount = std::uint32_t(patches.size());
	for(std::uint32_t i = 0; i < patchCount; ++i)
	{
		assert(!(contactIndex >= patches[i].firstContact && contactIndex < patches[i].firstContact + patches[i].normalCount + patches[i].tangentCount));
	}
#endif
	ScalarBounds& limits = scalarBounds[std::uint32_t(CompactContact::SCALAR_BOUNDS_TAG - contacts[contactIndex].block)];
	if(limits.lower == lowerImpulse && limits.upper == upperImpulse)
	{
		return;
	}
	const bool wasEquality = limits.lower == -MAX_IMPULSE && limits.upper == MAX_IMPULSE;
	const bool isEquality = lowerImpulse == -MAX_IMPULSE && upperImpulse == MAX_IMPULSE;
	limits.lower = lowerImpulse;
	limits.upper = upperImpulse;
	if(prepared)
	{
		equalityRows += int(isEquality) - int(wasEquality);
		++preparationGeneration;
	}
	if(!isEquality)
	{
		hasFiniteBounds = true;
	}
}

int Problem::addPatch(int firstContact, int normalCount, int tangentCount, double friction)
{
	const Patch patch = { firstContact, normalCount, tangentCount, friction };
	reserveStorage(patches, std::uint32_t(patches.size()) + 1);
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
		for(int end = 0; end < 2; ++end)
		{
			for(int column = 0; column < 6; ++column)
			{
				additional.tangentJacobian[end](0, column) = input.jacobian[end](0, column);
				additional.tangentJacobian[end](1, column) = input.jacobian[end](1, column);
			}
		}
		additional.freeVelocity = input.freeVelocity.head<2>();
		additional.regularization = input.regularization.head<2>();
		additional.friction = input.friction;
		additional.maxNormalImpulse = input.maxNormalImpulse;
		contact.block = (input.bilateral ? -1 : 1) * (int(contactBlocks.size()) + 1);
		reserveStorage(contactBlocks, std::uint32_t(contactBlocks.size()) + 1);
		contactBlocks.push_back(additional);
	}
	if(input.rowCount() == 1 && input.maxNormalImpulse != MAX_IMPULSE)
	{
		addScalarContact(contact, 0.0, input.maxNormalImpulse);
		return;
	}
	reserveStorage(contacts, std::uint32_t(contacts.size()) + 1);
	contacts.push_back(contact);
}

#ifndef NDEBUG
static bool validContact(const Problem& problem, const CompactContact& contact)
{
	const bool validBodies = contact.body[0] >= -1 && contact.body[1] >= -1 && contact.body[0] < problem.bodyCount() && contact.body[1] < problem.bodyCount() && (contact.body[0] < 0 || contact.body[0] != contact.body[1]);
	const std::uint32_t scalarIndex = std::uint32_t(CompactContact::SCALAR_BOUNDS_TAG - contact.block);
	const bool validScalarBounds = !contact.hasScalarBounds() || (scalarIndex < problem.scalarBounds.size() && problem.scalarBounds[scalarIndex].lower <= problem.scalarBounds[scalarIndex].upper && problem.scalarBounds[scalarIndex].lower < MAX_IMPULSE && problem.scalarBounds[scalarIndex].upper > -MAX_IMPULSE);
	const std::uint32_t blockIndex = std::uint32_t(contact.block > 0 ? contact.block - 1 : -std::int64_t(contact.block) - 1);
	const bool validBlock = !contact.block || (blockIndex < problem.contactBlocks.size() && (contact.block < 0 || (problem.contactBlocks[blockIndex].friction >= 0.0 && problem.contactBlocks[blockIndex].maxNormalImpulse >= 0.0 && problem.contactBlocks[blockIndex].regularization[0] == problem.contactBlocks[blockIndex].regularization[1])));
	return validBodies && validScalarBounds && validBlock;
}
#endif

static std::uint64_t bodyPairKey(int body0, int body1)
{
	const std::uint32_t first = std::uint32_t(std::min(body0, body1));
	const std::uint32_t second = std::uint32_t(std::max(body0, body1));
	return (std::uint64_t(first) << 32) | second;
}

// A topology group is one run of rows sharing a body pair, or one row when ungrouped.
static ANVIL_FORCE_INLINE std::uint32_t topologyGroupFirst(const std::vector<ScalarContactRun>& runs, bool grouped, std::uint32_t group)
{
	return grouped ? std::uint32_t(runs[group].first) : group;
}

static ANVIL_FORCE_INLINE std::uint32_t topologyGroupEnd(const std::vector<ScalarContactRun>& runs, bool grouped, std::uint32_t group)
{
	return grouped ? std::uint32_t(runs[group].end) : group + 1;
}

static void prepareHessianTopology(Problem& problem)
{
	const std::uint32_t bodyCount = std::uint32_t(problem.bodyCount());
	const std::uint32_t contactCount = std::uint32_t(problem.contacts.size());
	std::vector<std::uint64_t>& pairs = problem.hessianPairs;
	pairs.clear();
	problem.hessianDiagonalBlocks.resize(bodyCount);
	problem.hessianContactBlocks.assign(contactCount, -1);
	// Unilateral problems group consecutive rows of one body pair into runs, so each run
	// needs one lookup rather than one per row; other problems treat every row alone.
	const std::vector<ScalarContactRun>& runs = problem.scalarContactRuns;
	const bool grouped = !runs.empty();
	const std::uint32_t groupCount = grouped ? std::uint32_t(runs.size()) : contactCount;
	// Dense contact graphs contain many rows for the same body pair. For small
	// bodies, a retained direct lookup avoids sorting all duplicates and doing a
	// binary search for every group. Sparse and large systems keep the compact path.
	const bool useDenseLookup = bodyCount <= 256 && contactCount >= 8 * bodyCount;
	if(useDenseLookup)
	{
		std::vector<int>& lookup = problem.hessianPairLookup;
		lookup.assign(std::size_t(bodyCount) * bodyCount, -1);
		for(std::uint32_t body = 0; body < bodyCount; ++body)
		{
			lookup[std::size_t(body) * bodyCount + body] = 0;
		}
		for(std::uint32_t group = 0; group < groupCount; ++group)
		{
			const CompactContact& contact = problem.contacts[topologyGroupFirst(runs, grouped, group)];
			if(contact.body[0] >= 0 && contact.body[1] >= 0)
			{
				const std::uint32_t first = std::uint32_t(std::min(contact.body[0], contact.body[1]));
				const std::uint32_t second = std::uint32_t(std::max(contact.body[0], contact.body[1]));
				lookup[std::size_t(first) * bodyCount + second] = 0;
			}
		}
		reserveStorage(pairs, bodyCount + groupCount);
		for(std::uint32_t first = 0; first < bodyCount; ++first)
		{
			for(std::uint32_t second = first; second < bodyCount; ++second)
			{
				int& pair = lookup[std::size_t(first) * bodyCount + second];
				if(pair >= 0)
				{
					pair = int(pairs.size());
					pairs.emplace_back((std::uint64_t(first) << 32) | second);
				}
			}
			problem.hessianDiagonalBlocks[first] = lookup[std::size_t(first) * bodyCount + first];
		}
		for(std::uint32_t group = 0; group < groupCount; ++group)
		{
			const std::uint32_t begin = topologyGroupFirst(runs, grouped, group), end = topologyGroupEnd(runs, grouped, group);
			const CompactContact& contact = problem.contacts[begin];
			if(contact.body[0] >= 0 && contact.body[1] >= 0)
			{
				const std::uint32_t first = std::uint32_t(std::min(contact.body[0], contact.body[1]));
				const std::uint32_t second = std::uint32_t(std::max(contact.body[0], contact.body[1]));
				std::fill(problem.hessianContactBlocks.begin() + begin, problem.hessianContactBlocks.begin() + end, lookup[std::size_t(first) * bodyCount + second]);
			}
		}
		return;
	}

	reserveStorage(pairs, bodyCount + groupCount);
	for(std::uint32_t body = 0; body < bodyCount; ++body)
	{
		pairs.emplace_back((std::uint64_t(body) << 32) | body);
	}
	for(std::uint32_t group = 0; group < groupCount; ++group)
	{
		const CompactContact& contact = problem.contacts[topologyGroupFirst(runs, grouped, group)];
		if(contact.body[0] >= 0 && contact.body[1] >= 0)
		{
			pairs.emplace_back(bodyPairKey(contact.body[0], contact.body[1]));
		}
	}
	std::sort(pairs.begin(), pairs.end());
	pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
	const std::uint32_t pairCount = std::uint32_t(pairs.size());
	for(std::uint32_t pair = 0; pair < pairCount; ++pair)
	{
		const std::uint32_t first = std::uint32_t(pairs[pair] >> 32);
		if(first == std::uint32_t(pairs[pair]))
		{
			problem.hessianDiagonalBlocks[first] = int(pair);
		}
	}
	for(std::uint32_t group = 0; group < groupCount; ++group)
	{
		const std::uint32_t begin = topologyGroupFirst(runs, grouped, group), end = topologyGroupEnd(runs, grouped, group);
		const CompactContact& contact = problem.contacts[begin];
		if(contact.body[0] >= 0 && contact.body[1] >= 0)
		{
			const std::uint64_t key = bodyPairKey(contact.body[0], contact.body[1]);
			const int block = int(std::lower_bound(pairs.begin(), pairs.end(), key) - pairs.begin());
			std::fill(problem.hessianContactBlocks.begin() + begin, problem.hessianContactBlocks.begin() + end, block);
		}
	}
}

template<bool CountEntries, bool BuildJacobian>
static void prepareProblemInternal(Problem& problem)
{
	// Every three-row contact owns one block; scalar contacts own none.
	const int rows = int(problem.contacts.size()) + 2 * int(problem.contactBlocks.size());
	std::vector<int>& columnCounts = problem.columnCursors;
	problem.coupledContacts.clear();
	problem.equalityRows = 0;
	problem.hasFiniteBounds = !problem.patches.empty();
	problem.prepared = false;
	problem.compactJacobian = !BuildJacobian;
	assert(problem.massDiagonal.size() == problem.bodyCount() * 6);
	assert(!problem.massDiagonal.size() || problem.massDiagonal.minCoeff() > 0.0);
	const int massCount = problem.massDiagonal.size();
	problem.inverseMassDiagonal.resize(massCount);
	for(int i = 0; i < massCount; ++i)
	{
		problem.inverseMassDiagonal[i] = 1.0 / problem.massDiagonal[i];
	}
	const std::uint32_t patchCount = std::uint32_t(problem.patches.size());
	const int contactCount = int(problem.contacts.size());
#ifndef NDEBUG
	int previousPatchEnd = 0;
	for(std::uint32_t i = 0; i < patchCount; ++i)
	{
		const Patch& patch = problem.patches[i];
		assert(patch.friction >= 0.0);
		assert(patch.normalCount > 0 && patch.tangentCount >= 0 && patch.tangentCount <= 4 && patch.firstContact >= previousPatchEnd && patch.firstContact <= contactCount && patch.normalCount <= contactCount - patch.firstContact - patch.tangentCount);
		previousPatchEnd = patch.firstContact + patch.normalCount + patch.tangentCount;
	}
#endif
	if(CountEntries)
	{
		columnCounts.assign(problem.bodyCount() * 6, 0);
		for(int i = 0; i < contactCount; ++i)
		{
			const CompactContact& contact = problem.contacts[i];
			assert(validContact(problem, contact));
			const int first = contact.rowCount() == 1 ? 2 : 0;
			for(int end = 0; end < 2; ++end)
			{
				if(contact.body[end] >= 0)
				{
					for(int column = 0; column < 6; ++column)
					{
						for(int axis = first; axis < 3; ++axis)
						{
							columnCounts[6 * contact.body[end] + column] += problem.contactEntry(contact, end, axis, column) != 0.0;
						}
					}
				}
			}
		}
	}
	assert(!BuildJacobian || columnCounts.size() == std::uint32_t(problem.bodyCount() * 6));
	problem.freeVelocity.resize(rows);
	problem.regularization.resize(rows);
	reserveStorage(problem.rowContact, std::uint32_t(rows));
	problem.rowContact.resize(rows);
	// Rows are emitted in order and each constraint has distinct body endpoints.
	// Exact column capacities let us fill CSC storage without triplets, duplicate
	// reduction or an intermediate transpose.
	problem.jacobian.resize(rows, problem.bodyCount() * 6);
	int entries = 0;
	const int columnCount = problem.bodyCount() * 6;
	if(BuildJacobian)
	{
		for(int column = 0; column < columnCount; ++column)
		{
			problem.jacobian.outerIndexPtr()[column] = entries;
			const int count = columnCounts[column];
			assert(count >= 0 && count <= std::numeric_limits<int>::max() - entries);
			columnCounts[column] = entries;
			entries += count;
		}
		problem.jacobian.outerIndexPtr()[columnCount] = entries;
		problem.jacobian.resizeNonZeros(entries);
	}
	else
	{
		std::fill_n(problem.jacobian.outerIndexPtr(), columnCount + 1, 0);
		problem.jacobian.resizeNonZeros(0);
	}
	int nextRow = 0;
	std::uint32_t nextPatch = 0;
	// Runs of rows sharing a body pair are gathered in the same pass over the contact
	// records, only while the problem can still be unilateral.
	std::vector<ScalarContactRun>& runs = problem.scalarContactRuns;
	runs.clear();
	bool detectRuns = problem.scalarBounds.empty() && problem.patches.empty();
	if(detectRuns)
	{
		reserveStorage(runs, std::uint32_t(contactCount));
	}
	for(int i = 0; i < contactCount; ++i)
	{
		CompactContact& contact = problem.contacts[i];
		if(!CountEntries)
		{
			assert(validContact(problem, contact));
		}
		detectRuns = detectRuns && !contact.block;
		if(detectRuns)
		{
			if(i && contact.body[0] == problem.contacts[i - 1].body[0] && contact.body[1] == problem.contacts[i - 1].body[1])
			{
				runs.back().end = i + 1;
			}
			else
			{
				const ScalarContactRun run = { i, i + 1 };
				runs.push_back(run);
			}
		}
		while(nextPatch < patchCount && i >= problem.patches[nextPatch].firstContact + problem.patches[nextPatch].normalCount + problem.patches[nextPatch].tangentCount)
		{
			++nextPatch;
		}
		const Patch* patch = nextPatch < patchCount && i >= problem.patches[nextPatch].firstContact ? &problem.patches[nextPatch] : NULL;
		if(patch)
		{
#ifndef NDEBUG
			const CompactContact& first = problem.contacts[patch->firstContact];
			assert(contact.rowCount() == 1 && contact.body[0] == first.body[0] && contact.body[1] == first.body[1]);
#endif
			if(i < patch->firstContact + patch->normalCount)
			{
				assert(!contact.hasScalarBounds() || (problem.bounds(contact).lower == 0.0 && problem.bounds(contact).upper >= 0.0));
			}
			else
			{
				assert(contact.hasScalarBounds() && problem.bounds(contact).lower == -MAX_IMPULSE && problem.bounds(contact).upper == MAX_IMPULSE);
			}
		}
		contact.row = nextRow;
		nextRow += contact.rowCount();
		assert(nextRow <= rows);
		if(contact.hasScalarBounds())
		{
			const ScalarBounds& limits = problem.bounds(contact);
			if(!patch && limits.lower == -MAX_IMPULSE && limits.upper == MAX_IMPULSE)
			{
				++problem.equalityRows;
			}
			else if(limits.lower != -MAX_IMPULSE || limits.upper != MAX_IMPULSE)
			{
				problem.hasFiniteBounds = true;
			}
		}
		else if(contact.block)
		{
			problem.block(contact).coupled = -1;
			if(contact.block < 0)
			{
				problem.equalityRows += 3;
			}
			else if(problem.block(contact).maxNormalImpulse != MAX_IMPULSE)
			{
				problem.hasFiniteBounds = true;
			}
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
			assert(problem.regularization[row] > 0.0);
			if(BuildJacobian)
			{
				for(int end = 0; end < 2; ++end)
				{
					if(contact.body[end] >= 0)
					{
						for(int column = 0; column < 6; ++column)
						{
							const double value = problem.contactEntry(contact, end, axis, column);
							if(value != 0.0)
							{
								const int entry = columnCounts[6 * contact.body[end] + column]++;
								problem.jacobian.innerIndexPtr()[entry] = row;
								problem.jacobian.valuePtr()[entry] = value;
							}
						}
					}
				}
			}
		}
	}
	assert(nextRow == rows);
	problem.contactRebuildWork = -1.0;
	if(!problem.isUnilateral())
	{
		runs.clear();
	}
	problem.prepared = true;
	prepareHessianTopology(problem);
	++problem.preparationGeneration;
}

void prepareProblem(Problem& problem) noexcept
{
	prepareProblemInternal<true, true>(problem);
}

void prepareProblemFromColumnCounts(Problem& problem) noexcept
{
	prepareProblemInternal<false, true>(problem);
}

void prepareCompactProblemFromColumnCounts(Problem& problem) noexcept
{
	bool compact = problem.contactBlocks.empty() && problem.scalarBounds.empty() && problem.patches.empty();
	const int contactCount = int(problem.contacts.size());
	for(int i = 0; compact && i < contactCount; ++i)
	{
		compact = problem.contacts[i].block == 0;
	}
	if(compact)
	{
		prepareProblemInternal<false, false>(problem);
	}
	else
	{
		prepareProblemInternal<true, true>(problem);
	}
}

double computeResidual(const Problem& problem, ConstVector impulse)
{
	const int columnCount = int(problem.jacobian.cols());
	const int rowCount = problem.rowCount();
	VectorStorage bodyVelocity(columnCount);
	bodyVelocity.setZero();
	if(problem.compactJacobian)
	{
		const int contactCount = int(problem.contacts.size());
		for(int i = 0; i < contactCount; ++i)
		{
			const CompactContact& contact = problem.contacts[i];
			const double scale = impulse[contact.row];
			for(int end = 0; end < 2; ++end)
			{
				if(contact.body[end] >= 0)
				{
					addScaled6(bodyVelocity.data() + 6 * contact.body[end], contact.jacobian[end].data(), scale);
				}
			}
		}
	}
	else
	{
		const int* outer = problem.jacobian.outerIndexPtr();
		const int* inner = problem.jacobian.innerIndexPtr();
		const double* values = problem.jacobian.valuePtr();
		for(int column = 0; column < columnCount; ++column)
		{
			for(int entry = outer[column]; entry < outer[column + 1]; ++entry)
			{
				bodyVelocity[column] += values[entry] * impulse[inner[entry]];
			}
		}
	}
	VectorStorage velocity(rowCount);
	for(int row = 0; row < rowCount; ++row)
	{
		velocity[row] = problem.freeVelocity[row] + problem.regularization[row] * impulse[row];
	}
	if(problem.compactJacobian)
	{
		const int contactCount = int(problem.contacts.size());
		for(int i = 0; i < contactCount; ++i)
		{
			const CompactContact& contact = problem.contacts[i];
			double value = 0.0;
			for(int end = 0; end < 2; ++end)
			{
				if(contact.body[end] >= 0)
				{
					value += dot6(contact.jacobian[end].data(), bodyVelocity.data() + 6 * contact.body[end]);
				}
			}
			velocity[contact.row] += value;
		}
	}
	else
	{
		const int* outer = problem.jacobian.outerIndexPtr();
		const int* inner = problem.jacobian.innerIndexPtr();
		const double* values = problem.jacobian.valuePtr();
		for(int column = 0; column < columnCount; ++column)
		{
			const double value = bodyVelocity[column];
			for(int entry = outer[column]; entry < outer[column + 1]; ++entry)
			{
				velocity[inner[entry]] += values[entry] * value;
			}
		}
	}
	double error = 0.0;
	PatchScratch scratch;
	scratch.resize(problem);
	VectorStorage unitRegularization(scratch.velocity.size());
	unitRegularization.setOnes();
	std::uint32_t nextPatch = 0;
	const std::uint32_t patchCount = std::uint32_t(problem.patches.size());
	const int contactCount = int(problem.contacts.size());
	for(int i = 0; i < contactCount; ++i)
	{
		if(nextPatch < patchCount && i == problem.patches[nextPatch].firstContact)
		{
			const Patch& patch = problem.patches[nextPatch];
			const int count = patch.normalCount + patch.tangentCount;
			const int first = problem.contacts[i].row;
			double response = 0.0;
			for(int row = 0; row < count; ++row)
			{
				const CompactContact& contact = problem.contacts[i + row];
				for(int end = 0; end < 2; ++end)
				{
					if(contact.body[end] >= 0)
					{
						response += contact.jacobian[end].squaredNorm();
					}
				}
				if(row < patch.normalCount)
				{
					scratch.cap[row] = contact.hasScalarBounds() ? problem.bounds(contact).upper : MAX_IMPULSE;
				}
			}
			response = std::max(response / count, 1.0e-12);
			for(int row = 0; row < count; ++row)
			{
				scratch.velocity[row] = velocity[first + row] / response - impulse[first + row];
			}
			for(int row = 0; row < patch.tangentCount; ++row)
			{
				scratch.friction[row] = patch.friction;
			}
			const PatchProjectionInput input = { patch.normalCount, patch.tangentCount, scratch.velocity.data(), unitRegularization.data(), scratch.cap.data(), scratch.velocity.data() + patch.normalCount, unitRegularization.data() + patch.normalCount, scratch.friction };
			const PatchProjectionOutput output = { scratch.impulse.data(), scratch.impulse.data() + patch.normalCount, NULL, NULL, NULL };
			PatchProjectionResult projection;
			if(!projectPatchUnchecked(input, output, projection))
			{
				return MAX_IMPULSE;
			}
			for(int row = 0; row < count; ++row)
			{
				error = std::max(error, response * std::abs(impulse[first + row] - scratch.impulse[row]));
			}
			i += count - 1;
			++nextPatch;
			continue;
		}
		const CompactContact& c = problem.contacts[i];
		double jacobianNorm = 0.0;
		for(int end = 0; end < 2; ++end)
		{
			if(c.body[end] >= 0)
			{
				jacobianNorm += c.rowCount() == 1 ? c.jacobian[end].squaredNorm() : problem.contactJacobian(c, end).squaredNorm();
			}
		}
		const double response = std::max(jacobianNorm, 1.0e-12) / 3.0;
		if(c.rowCount() == 1)
		{
			const double trial = impulse[c.row] - velocity[c.row] / response;
			const double projected = c.hasScalarBounds() ? clampValue(trial, problem.bounds(c).lower, problem.bounds(c).upper) : std::max(0.0, trial);
			error = std::max(error, response * std::abs(impulse[c.row] - projected));
		}
		else
		{
			const Vec3 trial = impulse.segment<3>(c.row) - velocity.segment<3>(c.row) / response;
			const double unit[3] = { 1.0, 1.0, 1.0 };
			const Vec3 projected = c.block < 0 ? trial : weightedImpulse(trial, unit, unit, problem.block(c).friction, NULL, false, problem.block(c).maxNormalImpulse);
			error = std::max(error, response * (impulse.segment<3>(c.row) - projected).cwiseAbs().maxCoeff());
		}
	}
	return error;
}

typedef Mat6 BodyBlock;

struct HessianStorage
{
	SparseStorage matrix;
	std::vector<BodyBlock> blocks;
	// Whether each block received any product; the block factor skips zero blocks.
	std::vector<unsigned char> nonzero;
	std::vector<std::uint64_t> pairs;
};

// Jacobian rows often contain exact zeros. Skip zero columns while keeping
// each six-value destination column contiguous for vectorized accumulation.
static void addOuterProduct(BodyBlock& block, const Vec6& left, const Vec6& right, double weight)
{
	for(int axis = 0; axis < 6; ++axis)
	{
		if(right[axis] != 0.0)
		{
			addScaled6(block.data() + 6 * axis, left.data(), weight * right[axis]);
		}
	}
}

static void addJacobianProduct(BodyBlock& block, const Jacobian& left, const Mat3& weight, const Jacobian& right)
{
	for(int column = 0; column < 6; ++column)
	{
		const double right0 = right(0, column), right1 = right(1, column), right2 = right(2, column);
		const double weighted0 = weight(0, 0) * right0 + weight(0, 1) * right1 + weight(0, 2) * right2;
		const double weighted1 = weight(1, 0) * right0 + weight(1, 1) * right1 + weight(1, 2) * right2;
		const double weighted2 = weight(2, 0) * right0 + weight(2, 1) * right1 + weight(2, 2) * right2;
#if defined(ANVIL_AVX2_FMA)
		const __m256d multiplier0 = _mm256_set1_pd(weighted0);
		const __m256d multiplier1 = _mm256_set1_pd(weighted1);
		const __m256d multiplier2 = _mm256_set1_pd(weighted2);
		__m256d value = _mm256_mul_pd(multiplier0, _mm256_set_pd(left(0, 3), left(0, 2), left(0, 1), left(0, 0)));
		value = _mm256_fmadd_pd(multiplier1, _mm256_set_pd(left(1, 3), left(1, 2), left(1, 1), left(1, 0)), value);
		value = _mm256_fmadd_pd(multiplier2, _mm256_set_pd(left(2, 3), left(2, 2), left(2, 1), left(2, 0)), value);
		double* destination = block.data() + 6 * column;
		_mm256_storeu_pd(destination, _mm256_add_pd(_mm256_loadu_pd(destination), value));
		const __m128d multiplier0End = _mm256_castpd256_pd128(multiplier0);
		const __m128d multiplier1End = _mm256_castpd256_pd128(multiplier1);
		const __m128d multiplier2End = _mm256_castpd256_pd128(multiplier2);
		__m128d valueEnd = _mm_mul_pd(multiplier0End, _mm_set_pd(left(0, 5), left(0, 4)));
		valueEnd = _mm_fmadd_pd(multiplier1End, _mm_set_pd(left(1, 5), left(1, 4)), valueEnd);
		valueEnd = _mm_fmadd_pd(multiplier2End, _mm_set_pd(left(2, 5), left(2, 4)), valueEnd);
		_mm_storeu_pd(destination + 4, _mm_add_pd(_mm_loadu_pd(destination + 4), valueEnd));
#elif defined(ANVIL_SIMD128)
		const simd::Double2 multiplier0 = simd::splat(weighted0);
		const simd::Double2 multiplier1 = simd::splat(weighted1);
		const simd::Double2 multiplier2 = simd::splat(weighted2);
		for(int row = 0; row < 6; row += 2)
		{
			simd::Double2 value = simd::multiply(multiplier0, simd::make(left(0, row), left(0, row + 1)));
			value = simd::multiplyAdd(multiplier1, simd::make(left(1, row), left(1, row + 1)), value);
			value = simd::multiplyAdd(multiplier2, simd::make(left(2, row), left(2, row + 1)), value);
			double* destination = block.data() + 6 * column + row;
			simd::store(destination, simd::add(simd::load(destination), value));
		}
#else
		for(int row = 0; row < 6; ++row)
		{
			block(row, column) += left(0, row) * weighted0 + left(1, row) * weighted1 + left(2, row) * weighted2;
		}
#endif
	}
}

// Adds sum_i weight_i * left_i * right_i' over one run of scalar rows sharing a body
// pair. The block stays in registers for the whole run, so each destination is read
// and written once rather than once per row.
// Runs are unilateral scalar contacts, so each contact's row is its index.
static void addRunOuterProducts(BodyBlock& block, const CompactContact* ANVIL_RESTRICT contacts, int first, int end, const double* ANVIL_RESTRICT weights, int leftEnd, int rightEnd)
{
#if defined(ANVIL_AVX2_FMA)
	__m256d head[6];
	__m128d tail[6];
	for(int column = 0; column < 6; ++column)
	{
		head[column] = _mm256_setzero_pd();
		tail[column] = _mm_setzero_pd();
	}
	// The weight scales the left vector, so every column multiplier is a broadcast
	// load rather than a scalar product followed by a register shuffle.
	for(int i = first; i < end; ++i)
	{
		const double weight = weights[i];
		if(weight == 0.0)
		{
			continue;
		}
		const double* left = contacts[i].jacobian[leftEnd].data();
		const double* right = contacts[i].jacobian[rightEnd].data();
		const __m256d leftHead = _mm256_mul_pd(_mm256_set1_pd(weight), _mm256_loadu_pd(left));
		const __m128d leftTail = _mm_mul_pd(_mm_set1_pd(weight), _mm_loadu_pd(left + 4));
		for(int column = 0; column < 6; ++column)
		{
			const __m256d scale = _mm256_broadcast_sd(right + column);
			head[column] = _mm256_fmadd_pd(scale, leftHead, head[column]);
			tail[column] = _mm_fmadd_pd(_mm256_castpd256_pd128(scale), leftTail, tail[column]);
		}
	}
	double* destination = block.data();
	for(int column = 0; column < 6; ++column)
	{
		_mm256_storeu_pd(destination + 6 * column, _mm256_add_pd(_mm256_loadu_pd(destination + 6 * column), head[column]));
		_mm_storeu_pd(destination + 6 * column + 4, _mm_add_pd(_mm_loadu_pd(destination + 6 * column + 4), tail[column]));
	}
#elif defined(ANVIL_SIMD128)
	// A full block needs 18 two-double accumulators, so the run is read twice:
	// three columns per pass keep nine accumulators in registers.
	double* destination = block.data();
	for(int half = 0; half < 6; half += 3)
	{
		simd::Double2 accumulated[9];
		for(int k = 0; k < 9; ++k)
		{
			accumulated[k] = simd::zero();
		}
		for(int i = first; i < end; ++i)
		{
			const double weight = weights[i];
			if(weight == 0.0)
			{
				continue;
			}
			const double* left = contacts[i].jacobian[leftEnd].data();
			const double* right = contacts[i].jacobian[rightEnd].data() + half;
			const simd::Double2 multiplier = simd::splat(weight);
			const simd::Double2 left0 = simd::multiply(multiplier, simd::load(left));
			const simd::Double2 left2 = simd::multiply(multiplier, simd::load(left + 2));
			const simd::Double2 left4 = simd::multiply(multiplier, simd::load(left + 4));
			for(int column = 0; column < 3; ++column)
			{
				const simd::Double2 scale = simd::splat(right[column]);
				accumulated[3 * column] = simd::multiplyAdd(scale, left0, accumulated[3 * column]);
				accumulated[3 * column + 1] = simd::multiplyAdd(scale, left2, accumulated[3 * column + 1]);
				accumulated[3 * column + 2] = simd::multiplyAdd(scale, left4, accumulated[3 * column + 2]);
			}
		}
		for(int column = 0; column < 3; ++column)
		{
			for(int pair = 0; pair < 3; ++pair)
			{
				double* target = destination + 6 * (half + column) + 2 * pair;
				simd::store(target, simd::add(simd::load(target), accumulated[3 * column + pair]));
			}
		}
	}
#else
	for(int i = first; i < end; ++i)
	{
		const double weight = weights[i];
		if(weight != 0.0)
		{
			addOuterProduct(block, contacts[i].jacobian[leftEnd], contacts[i].jacobian[rightEnd], weight);
		}
	}
#endif
}

// Adds sum_i weight_i * v_i * v_i' to a diagonal block for one run, where v_i is
// the rows' jacobian at one end. Only the lower triangle is accumulated: columns
// two and three share rows two to five, and columns four and five share rows four
// and five, so eight packed products replace twelve. Upper entries are never read.
static void addRunLowerOuterProducts(BodyBlock& block, const CompactContact* ANVIL_RESTRICT contacts, int first, int end, const double* ANVIL_RESTRICT weights, int side)
{
#if defined(ANVIL_AVX2_FMA)
	__m256d head0 = _mm256_setzero_pd(), head1 = _mm256_setzero_pd(), middle2 = _mm256_setzero_pd(), middle3 = _mm256_setzero_pd();
	__m128d tail0 = _mm_setzero_pd(), tail1 = _mm_setzero_pd(), tail4 = _mm_setzero_pd(), tail5 = _mm_setzero_pd();
	for(int i = first; i < end; ++i)
	{
		const double weight = weights[i];
		if(weight == 0.0)
		{
			continue;
		}
		const double* vector = contacts[i].jacobian[side].data();
		const __m256d multiplier = _mm256_set1_pd(weight);
		const __m256d vectorHead = _mm256_mul_pd(multiplier, _mm256_loadu_pd(vector));
		const __m256d vectorMiddle = _mm256_mul_pd(multiplier, _mm256_loadu_pd(vector + 2));
		const __m128d vectorTail = _mm_mul_pd(_mm256_castpd256_pd128(multiplier), _mm_loadu_pd(vector + 4));
		const __m256d scale0 = _mm256_broadcast_sd(vector), scale1 = _mm256_broadcast_sd(vector + 1);
		const __m256d scale2 = _mm256_broadcast_sd(vector + 2), scale3 = _mm256_broadcast_sd(vector + 3);
		const __m128d scale4 = _mm_loaddup_pd(vector + 4), scale5 = _mm_loaddup_pd(vector + 5);
		head0 = _mm256_fmadd_pd(scale0, vectorHead, head0);
		tail0 = _mm_fmadd_pd(_mm256_castpd256_pd128(scale0), vectorTail, tail0);
		head1 = _mm256_fmadd_pd(scale1, vectorHead, head1);
		tail1 = _mm_fmadd_pd(_mm256_castpd256_pd128(scale1), vectorTail, tail1);
		middle2 = _mm256_fmadd_pd(scale2, vectorMiddle, middle2);
		middle3 = _mm256_fmadd_pd(scale3, vectorMiddle, middle3);
		tail4 = _mm_fmadd_pd(scale4, vectorTail, tail4);
		tail5 = _mm_fmadd_pd(scale5, vectorTail, tail5);
	}
	double* destination = block.data();
	_mm256_storeu_pd(destination, _mm256_add_pd(_mm256_loadu_pd(destination), head0));
	_mm_storeu_pd(destination + 4, _mm_add_pd(_mm_loadu_pd(destination + 4), tail0));
	_mm256_storeu_pd(destination + 6, _mm256_add_pd(_mm256_loadu_pd(destination + 6), head1));
	_mm_storeu_pd(destination + 10, _mm_add_pd(_mm_loadu_pd(destination + 10), tail1));
	_mm256_storeu_pd(destination + 14, _mm256_add_pd(_mm256_loadu_pd(destination + 14), middle2));
	_mm256_storeu_pd(destination + 20, _mm256_add_pd(_mm256_loadu_pd(destination + 20), middle3));
	_mm_storeu_pd(destination + 28, _mm_add_pd(_mm_loadu_pd(destination + 28), tail4));
	_mm_storeu_pd(destination + 34, _mm_add_pd(_mm_loadu_pd(destination + 34), tail5));
#elif defined(ANVIL_SIMD128)
	// Two passes of six accumulators: columns 0 and 1 over rows 0-5, then columns 2 and 3
	// over rows 2-5 and columns 4 and 5 over rows 4 and 5. Column 1's row 0 and column 3's
	// row 2 are unread upper entries.
	double* destination = block.data();
	simd::Double2 lowColumns[6], highColumns[6];
	for(int k = 0; k < 6; ++k)
	{
		lowColumns[k] = simd::zero();
		highColumns[k] = simd::zero();
	}
	for(int i = first; i < end; ++i)
	{
		const double weight = weights[i];
		if(weight == 0.0)
		{
			continue;
		}
		const double* vector = contacts[i].jacobian[side].data();
		const simd::Double2 multiplier = simd::splat(weight);
		const simd::Double2 vector0 = simd::multiply(multiplier, simd::load(vector));
		const simd::Double2 vector2 = simd::multiply(multiplier, simd::load(vector + 2));
		const simd::Double2 vector4 = simd::multiply(multiplier, simd::load(vector + 4));
		const simd::Double2 scale0 = simd::splat(vector[0]), scale1 = simd::splat(vector[1]);
		lowColumns[0] = simd::multiplyAdd(scale0, vector0, lowColumns[0]);
		lowColumns[1] = simd::multiplyAdd(scale0, vector2, lowColumns[1]);
		lowColumns[2] = simd::multiplyAdd(scale0, vector4, lowColumns[2]);
		lowColumns[3] = simd::multiplyAdd(scale1, vector0, lowColumns[3]);
		lowColumns[4] = simd::multiplyAdd(scale1, vector2, lowColumns[4]);
		lowColumns[5] = simd::multiplyAdd(scale1, vector4, lowColumns[5]);
	}
	for(int i = first; i < end; ++i)
	{
		const double weight = weights[i];
		if(weight == 0.0)
		{
			continue;
		}
		const double* vector = contacts[i].jacobian[side].data();
		const simd::Double2 multiplier = simd::splat(weight);
		const simd::Double2 vector2 = simd::multiply(multiplier, simd::load(vector + 2));
		const simd::Double2 vector4 = simd::multiply(multiplier, simd::load(vector + 4));
		const simd::Double2 scale2 = simd::splat(vector[2]), scale3 = simd::splat(vector[3]);
		highColumns[0] = simd::multiplyAdd(scale2, vector2, highColumns[0]);
		highColumns[1] = simd::multiplyAdd(scale2, vector4, highColumns[1]);
		highColumns[2] = simd::multiplyAdd(scale3, vector2, highColumns[2]);
		highColumns[3] = simd::multiplyAdd(scale3, vector4, highColumns[3]);
		highColumns[4] = simd::multiplyAdd(simd::splat(vector[4]), vector4, highColumns[4]);
		highColumns[5] = simd::multiplyAdd(simd::splat(vector[5]), vector4, highColumns[5]);
	}
	const int firstOffsets[6] = { 0, 2, 4, 6, 8, 10 };
	const int secondOffsets[6] = { 14, 16, 20, 22, 28, 34 };
	for(int k = 0; k < 6; ++k)
	{
		simd::store(destination + firstOffsets[k], simd::add(simd::load(destination + firstOffsets[k]), lowColumns[k]));
		simd::store(destination + secondOffsets[k], simd::add(simd::load(destination + secondOffsets[k]), highColumns[k]));
	}
#else
	addRunOuterProducts(block, contacts, first, end, weights, side, side);
#endif
}

// Accumulate the Hessian's 6x6 body-pair blocks. Diagonal blocks are read only
// through their lower triangles.
static void assembleHessianBlocks(const Problem& problem, const Curvature& weights, HessianStorage& storage)
{
	const std::vector<std::uint64_t>& pairs = problem.hessianPairs;
	const std::uint32_t pairCount = std::uint32_t(pairs.size());
	reserveStorage(storage.blocks, pairCount);
	storage.blocks.resize(pairCount);
	reserveStorage(storage.nonzero, pairCount);
	storage.nonzero.resize(pairCount);
	for(std::uint32_t pair = 0; pair < pairCount; ++pair)
	{
		BodyBlock& block = storage.blocks[pair];
		const std::uint32_t first = std::uint32_t(pairs[pair] >> 32);
		const std::uint32_t second = std::uint32_t(pairs[pair]);
		storage.nonzero[pair] = first == second;
		if(first == second)
		{
			block.setIdentity();
		}
		else
		{
			block.setZero();
		}
	}
	const std::uint32_t contactCount = std::uint32_t(problem.contacts.size());
	if(problem.isUnilateral())
	{
		// Runs share one ordered body pair, so their three blocks accumulate together.
		const CompactContact* contacts = problem.contacts.data();
		const double* diagonal = weights.diagonal.data();
		const std::uint32_t runCount = std::uint32_t(problem.scalarContactRuns.size());
		for(std::uint32_t run = 0; run < runCount; ++run)
		{
			const int first = problem.scalarContactRuns[run].first, end = problem.scalarContactRuns[run].end;
			const int a = contacts[first].body[0], b = contacts[first].body[1];
			assert(contacts[first].row == first);
			// Runs without curvature, such as untouched neighbours, add nothing.
			bool curved = false;
			for(int i = first; i < end; ++i)
			{
				curved = curved | (diagonal[i] != 0.0);
			}
			if(!curved)
			{
				continue;
			}
			for(int side = 0; side < 2; ++side)
			{
				if(contacts[first].body[side] >= 0)
				{
					addRunLowerOuterProducts(storage.blocks[problem.hessianDiagonalBlocks[contacts[first].body[side]]], contacts, first, end, diagonal, side);
				}
			}
			if(a >= 0 && b >= 0)
			{
				const int low = a < b ? 0 : 1;
				addRunOuterProducts(storage.blocks[problem.hessianContactBlocks[first]], contacts, first, end, diagonal, 1 - low, low);
				storage.nonzero[problem.hessianContactBlocks[first]] = 1;
			}
		}
	}
	else
	{
		for(std::uint32_t i = 0; i < contactCount; ++i)
		{
			const CompactContact& contact = problem.contacts[i];
			const int a = contact.body[0], b = contact.body[1];
			const int cross = problem.hessianContactBlocks[i];
			if(contact.block > 0)
			{
				const Mat3& weight = weights.coupled[problem.coupledIndex(contact)];
				if(weight.isZero(0.0))
				{
					continue;
				}
				Jacobian jacobian[2];
				for(int end = 0; end < 2; ++end)
				{
					if(contact.body[end] >= 0)
					{
						jacobian[end] = problem.contactJacobian(contact, end);
						addJacobianProduct(storage.blocks[problem.hessianDiagonalBlocks[contact.body[end]]], jacobian[end], weight, jacobian[end]);
					}
				}
				if(a >= 0 && b >= 0)
				{
					const int low = a < b ? 0 : 1;
					addJacobianProduct(storage.blocks[cross], jacobian[1 - low], weight, jacobian[low]);
					storage.nonzero[cross] = 1;
				}
			}
			else
			{
				const int first = contact.rowCount() == 1 ? 2 : 0;
				for(int axis = first; axis < 3; ++axis)
				{
					const double weight = weights.diagonal[contact.row + axis - first];
					if(weight == 0.0)
					{
						continue;
					}
					Vec6 jacobian[2];
					for(int end = 0; end < 2; ++end)
					{
						if(contact.body[end] >= 0)
						{
							jacobian[end] = problem.contactRow(contact, end, axis);
							addOuterProduct(storage.blocks[problem.hessianDiagonalBlocks[contact.body[end]]], jacobian[end], jacobian[end], weight);
						}
					}
					if(a >= 0 && b >= 0)
					{
						const int low = a < b ? 0 : 1;
						addOuterProduct(storage.blocks[cross], jacobian[1 - low], jacobian[low], weight);
						storage.nonzero[cross] = 1;
					}
				}
			}
		}
	}
	const std::uint32_t patchCount = std::uint32_t(problem.patches.size());
	for(std::uint32_t i = 0; i < patchCount; ++i)
	{
		const CompactContact& contact = problem.contacts[problem.patches[i].firstContact];
		const PatchCurvature& patch = weights.patches[i];
		for(int left = 0; left < 2; ++left)
		{
			for(int right = 0; right < 2; ++right)
			{
				const int a = contact.body[left], b = contact.body[right];
				if(a < 0 || b < 0 || a < b)
				{
					continue;
				}
				const int index = a == b ? problem.hessianDiagonalBlocks[a] :
					problem.hessianContactBlocks[problem.patches[i].firstContact];
				BodyBlock& block = storage.blocks[index];
				storage.nonzero[index] = 1;
				addOuterProduct(block, patch.normal[left], patch.normal[right], patch.normalCoefficient);
				addOuterProduct(block, patch.normal[left], patch.tangent[right], patch.crossCoefficient);
				addOuterProduct(block, patch.tangent[left], patch.normal[right], patch.crossCoefficient);
				addOuterProduct(block, patch.tangent[left], patch.tangent[right], patch.tangentCoefficient);
			}
		}
	}
}

// Export assembled blocks as the scalar upper-triangular CSC Hessian.
static const Sparse& exportHessian(const Problem& problem, HessianStorage& storage)
{
	const int bodies = problem.bodyCount(), size = bodies * 6;
	const std::vector<std::uint64_t>& pairs = problem.hessianPairs;
	const std::uint32_t pairCount = std::uint32_t(pairs.size());
	SparseStorage& matrix = storage.matrix;
	const bool sameStructure = storage.pairs == pairs;
	// Sorted body pairs emit 21 entries per diagonal block and 36 per
	// off-diagonal block, including structural zeros required by rank updates.
	if(!sameStructure)
	{
		matrix.resize(size, size);
		matrix.resizeNonZeros(int(pairCount) * 36 - bodies * 15);
		storage.pairs = pairs;
	}
	int* outer = matrix.outerIndexPtr();
	int* inner = matrix.innerIndexPtr();
	double* values = matrix.valuePtr();
	int entry = 0;
	std::uint32_t begin = 0;
	for(int body = 0; body < bodies; ++body)
	{
		std::uint32_t end = begin;
		while(end < pairCount && std::uint32_t(pairs[end] >> 32) == std::uint32_t(body))
		{
			++end;
		}
		for(int axis = 0; axis < 6; ++axis)
		{
			if(!sameStructure)
			{
				outer[6 * body + axis] = entry;
			}
			for(std::uint32_t pair = begin; pair < end; ++pair)
			{
				const int second = int(std::uint32_t(pairs[pair]));
				const int first = second == body ? axis : 0;
				for(int row = first; row < 6; ++row)
				{
					if(!sameStructure)
					{
						inner[entry] = second * 6 + row;
					}
					values[entry++] = storage.blocks[pair](row, axis);
				}
			}
		}
		begin = end;
	}
	if(!sameStructure)
	{
		outer[size] = entry;
	}
	return matrix;
}

static const Sparse& makeHessian(const Problem& problem, const Curvature& weights, HessianStorage& storage)
{
	assembleHessianBlocks(problem, weights, storage);
	return exportHessian(problem, storage);
}

static bool solveDenseReference(const Sparse& matrix, ConstVector gradient, MutableVector solution)
{
	const int size = matrix.cols();
	std::vector<double> lower(std::uint32_t(size) * std::uint32_t(size), 0.0);
	for(int column = 0; column < size; ++column)
	{
		for(Sparse::InnerIterator entry(matrix, column); entry; ++entry)
		{
			lower[std::uint32_t(entry.row()) * std::uint32_t(size) + std::uint32_t(column)] = entry.value();
			lower[std::uint32_t(column) * std::uint32_t(size) + std::uint32_t(entry.row())] = entry.value();
		}
	}
	for(int column = 0; column < size; ++column)
	{
		double diagonal = lower[std::uint32_t(column) * std::uint32_t(size) + std::uint32_t(column)];
		for(int inner = 0; inner < column; ++inner)
		{
			const double value = lower[std::uint32_t(column) * std::uint32_t(size) + std::uint32_t(inner)];
			diagonal -= value * value;
		}
		if(!(diagonal > 0.0))
		{
			return false;
		}
		const double root = std::sqrt(diagonal);
		lower[std::uint32_t(column) * std::uint32_t(size) + std::uint32_t(column)] = root;
		for(int row = column + 1; row < size; ++row)
		{
			double value = lower[std::uint32_t(row) * std::uint32_t(size) + std::uint32_t(column)];
			for(int inner = 0; inner < column; ++inner)
			{
				value -= lower[std::uint32_t(row) * std::uint32_t(size) + std::uint32_t(inner)] *
						 lower[std::uint32_t(column) * std::uint32_t(size) + std::uint32_t(inner)];
			}
			lower[std::uint32_t(row) * std::uint32_t(size) + std::uint32_t(column)] = value / root;
		}
	}
	solution.resize(size);
	for(int row = 0; row < size; ++row)
	{
		double value = -gradient[row];
		for(int column = 0; column < row; ++column)
		{
			value -= lower[std::uint32_t(row) * std::uint32_t(size) + std::uint32_t(column)] * solution[column];
		}
		solution[row] = value / lower[std::uint32_t(row) * std::uint32_t(size) + std::uint32_t(row)];
	}
	for(int row = size - 1; row >= 0; --row)
	{
		double value = solution[row];
		for(int column = row + 1; column < size; ++column)
		{
			value -= lower[std::uint32_t(column) * std::uint32_t(size) + std::uint32_t(row)] * solution[column];
		}
		solution[row] = value / lower[std::uint32_t(row) * std::uint32_t(size) + std::uint32_t(row)];
	}
	return true;
}

static void evaluateUnilateralImpulses(ConstVector contactVelocity, ConstVector inverseRoot, MutableVector impulse, Curvature* weights)
{
	const int rowCount = contactVelocity.size();
#if defined(ANVIL_AVX2_FMA)
	const __m256d zero = _mm256_setzero_pd();
	int row = 0;
	if(weights)
	{
		for(; row + 4 <= rowCount; row += 4)
		{
			const __m256d root = _mm256_loadu_pd(inverseRoot.data() + row);
			const __m256d velocity = _mm256_loadu_pd(contactVelocity.data() + row);
			const __m256d active = _mm256_cmp_pd(velocity, zero, _CMP_LT_OQ);
			const __m256d drive = _mm256_max_pd(zero, _mm256_mul_pd(_mm256_sub_pd(zero, root), velocity));
			_mm256_storeu_pd(impulse.data() + row, _mm256_mul_pd(root, drive));
			_mm256_storeu_pd(weights->diagonal.data() + row, _mm256_and_pd(active, _mm256_mul_pd(root, root)));
		}
	}
	else
	{
		for(; row + 4 <= rowCount; row += 4)
		{
			const __m256d root = _mm256_loadu_pd(inverseRoot.data() + row);
			const __m256d velocity = _mm256_loadu_pd(contactVelocity.data() + row);
			const __m256d drive = _mm256_max_pd(zero, _mm256_mul_pd(_mm256_sub_pd(zero, root), velocity));
			_mm256_storeu_pd(impulse.data() + row, _mm256_mul_pd(root, drive));
		}
	}
	for(; row < rowCount; ++row)
#elif defined(ANVIL_SIMD128)
	const simd::Double2 zero = simd::zero();
	int row = 0;
	for(; row + 2 <= rowCount; row += 2)
	{
		const simd::Double2 root = simd::load(inverseRoot.data() + row);
		const simd::Double2 velocity = simd::load(contactVelocity.data() + row);
		const simd::Double2 drive = simd::maximum(zero, simd::multiply(simd::subtract(zero, root), velocity));
		simd::store(impulse.data() + row, simd::multiply(root, drive));
		if(weights)
		{
			simd::store(weights->diagonal.data() + row, simd::bitAnd(simd::less(velocity, zero), simd::multiply(root, root)));
		}
	}
	for(; row < rowCount; ++row)
#else
	for(int row = 0; row < rowCount; ++row)
#endif
	{
		impulse[row] = inverseRoot[row] * std::max(0.0, -inverseRoot[row] * contactVelocity[row]);
		if(weights)
		{
			weights->diagonal[row] = contactVelocity[row] < 0.0 ? inverseRoot[row] * inverseRoot[row] : 0.0;
		}
	}
}

// After a line search, advance the contact velocity along the accepted step
// and reevaluate unilateral impulses and curvature in the same pass. Rows whose
// curvature differs from the retained factor are listed for its next update.
// Returns sum(R * lambda^2); for an active row R * lambda^2 is its squared drive.
static double evaluateUnilateralStep(int rowCount, double alpha, bool advance, double* ANVIL_RESTRICT contactVelocity, const double* ANVIL_RESTRICT contactDirection,
	const double* ANVIL_RESTRICT inverseRoot, double* ANVIL_RESTRICT impulse, double* ANVIL_RESTRICT diagonal, const double* ANVIL_RESTRICT factorDiagonal,
	int* ANVIL_RESTRICT changedRows, int& changedCount)
{
	int changes = 0;
	int row = 0;
	double quadratic = 0.0;
#if defined(ANVIL_AVX2_FMA)
	const __m256d zero = _mm256_setzero_pd();
	const __m256d step = _mm256_set1_pd(advance ? alpha : 0.0);
	__m256d quadraticSum = zero;
	for(; row + 4 <= rowCount; row += 4)
	{
		__m256d velocity = _mm256_loadu_pd(contactVelocity + row);
		if(advance)
		{
			velocity = _mm256_fmadd_pd(step, _mm256_loadu_pd(contactDirection + row), velocity);
			_mm256_storeu_pd(contactVelocity + row, velocity);
		}
		const __m256d root = _mm256_loadu_pd(inverseRoot + row);
		const __m256d active = _mm256_cmp_pd(velocity, zero, _CMP_LT_OQ);
		const __m256d drive = _mm256_max_pd(zero, _mm256_mul_pd(_mm256_sub_pd(zero, root), velocity));
		_mm256_storeu_pd(impulse + row, _mm256_mul_pd(root, drive));
		const __m256d weight = _mm256_and_pd(active, _mm256_mul_pd(root, root));
		_mm256_storeu_pd(diagonal + row, weight);
		quadraticSum = _mm256_fmadd_pd(drive, drive, quadraticSum);
		int changed = _mm256_movemask_pd(_mm256_cmp_pd(weight, _mm256_loadu_pd(factorDiagonal + row), _CMP_NEQ_UQ));
		while(changed)
		{
			const int lane = int(_tzcnt_u32(unsigned(changed)));
			changedRows[changes++] = row + lane;
			changed &= changed - 1;
		}
	}
	const __m128d quadraticHalves = _mm_add_pd(_mm256_castpd256_pd128(quadraticSum), _mm256_extractf128_pd(quadraticSum, 1));
	quadratic = _mm_cvtsd_f64(_mm_add_pd(quadraticHalves, _mm_unpackhi_pd(quadraticHalves, quadraticHalves)));
#elif defined(ANVIL_SIMD128)
	const simd::Double2 zero = simd::zero();
	const simd::Double2 step = simd::splat(advance ? alpha : 0.0);
	simd::Double2 quadraticSum = zero;
	for(; row + 2 <= rowCount; row += 2)
	{
		simd::Double2 velocity = simd::load(contactVelocity + row);
		if(advance)
		{
			velocity = simd::multiplyAdd(step, simd::load(contactDirection + row), velocity);
			simd::store(contactVelocity + row, velocity);
		}
		const simd::Double2 root = simd::load(inverseRoot + row);
		const simd::Double2 drive = simd::maximum(zero, simd::multiply(simd::subtract(zero, root), velocity));
		simd::store(impulse + row, simd::multiply(root, drive));
		const simd::Double2 weight = simd::bitAnd(simd::less(velocity, zero), simd::multiply(root, root));
		simd::store(diagonal + row, weight);
		quadraticSum = simd::multiplyAdd(drive, drive, quadraticSum);
		const int changed = simd::mask(simd::notEqual(weight, simd::load(factorDiagonal + row)));
		if(changed & 1)
		{
			changedRows[changes++] = row;
		}
		if(changed & 2)
		{
			changedRows[changes++] = row + 1;
		}
	}
	quadratic = simd::sum(quadraticSum);
#endif
	for(; row < rowCount; ++row)
	{
		if(advance)
		{
			contactVelocity[row] += alpha * contactDirection[row];
		}
		const double velocity = contactVelocity[row];
		const double drive = std::max(0.0, -inverseRoot[row] * velocity);
		impulse[row] = inverseRoot[row] * drive;
		diagonal[row] = velocity < 0.0 ? inverseRoot[row] * inverseRoot[row] : 0.0;
		quadratic += drive * drive;
		if(diagonal[row] != factorDiagonal[row])
		{
			changedRows[changes++] = row;
		}
	}
	changedCount = changes;
	return quadratic;
}

static bool evaluateImpulses(const Problem& problem, ConstVector contactVelocity, ConstVector inverseRoot, MutableVector impulse, Curvature* weights, PatchScratch& scratch)
{
	ConstVector compliance = problem.regularization;
	std::uint32_t nextPatch = 0;
	const std::uint32_t patchCount = std::uint32_t(problem.patches.size());
	const int contactCount = int(problem.contacts.size());
	for(int i = 0; i < contactCount; ++i)
	{
		if(nextPatch < patchCount && i == problem.patches[nextPatch].firstContact)
		{
			const Patch& group = problem.patches[nextPatch];
			const int first = problem.contacts[i].row;
			PatchProjectionResult projection;
			if(!projectGroup(problem, group, contactVelocity.data() + first, impulse.data() + first, weights ? weights->diagonal.data() + first : NULL, weights ? scratch.coupling : NULL, scratch, projection))
			{
				return false;
			}
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
					{
						continue;
					}
					for(int row = 0; row < group.normalCount; ++row)
					{
						patch.normal[end] += weights->diagonal[first + row] * problem.contacts[i + row].jacobian[end];
					}
					for(int row = 0; row < group.tangentCount; ++row)
					{
						patch.tangent[end] += scratch.coupling[row] * problem.contacts[i + group.normalCount + row].jacobian[end];
					}
				}
				for(int row = 0; row < group.tangentCount; ++row)
				{
					patch.tangentCoupling[row] = scratch.coupling[row];
				}
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
				const double inverseCompliance = 1.0 / compliance[row];
				const double unconstrained = value * inverseCompliance;
				impulse[row] = clampValue(unconstrained, limits.lower, limits.upper);
				if(weights)
				{
					weights->diagonal[row] = unconstrained > limits.lower && unconstrained < limits.upper ? inverseCompliance : 0.0;
				}
			}
			else
			{
				impulse[row] = inverseRoot[row] * std::max(0.0, inverseRoot[row] * value);
				if(weights)
				{
					weights->diagonal[row] = value > 0.0 ? inverseRoot[row] * inverseRoot[row] : 0.0;
				}
			}
		}
		else if(contact.block < 0)
		{
			for(int axis = 0; axis < 3; ++axis)
			{
				const double inverseCompliance = 1.0 / compliance[row + axis];
				impulse[row + axis] = -contactVelocity[row + axis] * inverseCompliance;
				if(weights)
				{
					weights->diagonal[row + axis] = inverseCompliance;
				}
			}
		}
		else
		{
			const Vec3 value = weightedImpulse(-contactVelocity.segment<3>(row), compliance.data() + row, inverseRoot.data() + row, problem.block(contact).friction, weights ? &weights->coupled[problem.coupledIndex(contact)] : NULL, false, problem.block(contact).maxNormalImpulse);
			storeVector<3>(impulse.data() + row, value);
		}
	}
	return true;
}

// These calls use distinct solver-owned input/output buffers. The prepared
// Jacobian is compressed CSC with strictly increasing row indices per column.
static void multiplyJacobianScalarContacts(const Problem& problem, ConstVector vector, MutableVector product)
{
	const double* ANVIL_RESTRICT input = vector.data();
	double* ANVIL_RESTRICT output = product.data();
	const CompactContact* ANVIL_RESTRICT contacts = problem.contacts.data();
	const int runCount = int(problem.scalarContactRuns.size());
	for(int runIndex = 0; runIndex < runCount; ++runIndex)
	{
		const ScalarContactRun& run = problem.scalarContactRuns[runIndex];
		const int body0 = contacts[run.first].body[0];
		const int body1 = contacts[run.first].body[1];
		if(body0 >= 0 && body1 >= 0)
		{
			const double* ANVIL_RESTRICT input0 = input + 6 * body0;
			const double* ANVIL_RESTRICT input1 = input + 6 * body1;
			for(int i = run.first; i < run.end; ++i)
			{
				const CompactContact& contact = contacts[i];
				assert(contact.row == i);
				output[i] = dot6Pair(contact.jacobian[0].data(), input0, contact.jacobian[1].data(), input1);
			}
		}
		else if(body0 >= 0)
		{
			const double* ANVIL_RESTRICT input0 = input + 6 * body0;
			for(int i = run.first; i < run.end; ++i)
			{
				const CompactContact& contact = contacts[i];
				assert(contact.row == i);
				output[i] = dot6(contact.jacobian[0].data(), input0);
			}
		}
		else
		{
			const double* ANVIL_RESTRICT input1 = input + 6 * body1;
			for(int i = run.first; i < run.end; ++i)
			{
				const CompactContact& contact = contacts[i];
				assert(contact.row == i);
				output[i] = dot6(contact.jacobian[1].data(), input1);
			}
		}
	}
}

static void multiplyJacobianCscSerial(const Problem& problem, ConstVector vector, MutableVector product)
{
	const double* ANVIL_RESTRICT values = problem.jacobian.valuePtr();
	const int* ANVIL_RESTRICT indices = problem.jacobian.innerIndexPtr();
	const int* ANVIL_RESTRICT offsets = problem.jacobian.outerIndexPtr();
	const double* ANVIL_RESTRICT input = vector.data();
	double* ANVIL_RESTRICT output = product.data();
	std::fill_n(output, product.size(), 0.0);
	const int columnCount = int(problem.jacobian.cols());
	for(int column = 0; column < columnCount; ++column)
	{
		const double scale = input[column];
		const int end = offsets[column + 1];
		for(int entry = offsets[column]; entry < end; ++entry)
		{
			output[indices[entry]] += values[entry] * scale;
		}
	}
}

struct ContactVelocityEvaluation
{
	const Problem* problem;
	const double* velocity;
	double* contactVelocity;
	int chunks;
};

static void evaluateContactVelocityChunk(void* context, int index)
{
	ContactVelocityEvaluation& evaluation = *static_cast<ContactVelocityEvaluation*>(context);
	const int count = int(evaluation.problem->contacts.size());
	const int firstContact = count * index / evaluation.chunks;
	const int lastContact = count * (index + 1) / evaluation.chunks;
	for(int i = firstContact; i < lastContact; ++i)
	{
		const CompactContact& contact = evaluation.problem->contacts[i];
		const int firstEnd = contact.body[1] >= 0 && (contact.body[0] < 0 || contact.body[1] < contact.body[0]) ? 1 : 0;
		if(contact.rowCount() == 1)
		{
			double value = 0.0;
			for(int endpoint = 0; endpoint < 2; ++endpoint)
			{
				const int end = endpoint == 0 ? firstEnd : 1 - firstEnd;
				if(contact.body[end] >= 0)
				{
					value += contact.jacobian[end].dot(loadVector<6>(evaluation.velocity + 6 * contact.body[end]));
				}
			}
			evaluation.contactVelocity[contact.row] = value;
		}
		else
		{
			Vec3 value = Vec3::Zero();
			const ContactBlock& block = evaluation.problem->block(contact);
			for(int endpoint = 0; endpoint < 2; ++endpoint)
			{
				const int end = endpoint == 0 ? firstEnd : 1 - firstEnd;
				if(contact.body[end] < 0)
				{
					continue;
				}
				const Vec6 velocity = loadVector<6>(evaluation.velocity + 6 * contact.body[end]);
				const Vec2 tangent = block.tangentJacobian[end] * velocity;
				value[0] += tangent[0];
				value[1] += tangent[1];
				value[2] += contact.jacobian[end].dot(velocity);
			}
			storeVector<3>(evaluation.contactVelocity + contact.row, value);
		}
	}
}

static void multiplyJacobianCsc(const Problem& problem, ConstVector vector, MutableVector product, ParallelExecutor* parallelExecutor)
{
	if(problem.isUnilateral())
	{
		multiplyJacobianScalarContacts(problem, vector, product);
		return;
	}
	if(parallelExecutor != NULL && problem.bodyCount() >= 500)
	{
		const int workers = parallelExecutor->acquireWorkerCount();
		if(workers > 1)
		{
			ContactVelocityEvaluation evaluation = { &problem, vector.data(), product.data(), 4 * workers };
			parallelExecutor->parallelFor(evaluation.chunks, evaluateContactVelocityChunk, &evaluation);
			return;
		}
	}
	multiplyJacobianCscSerial(problem, vector, product);
}

struct GradientEvaluation
{
	const Problem* problem;
	const double* impulse;
	const double* velocity;
	double* gradient;
	int chunks;
};

static void evaluateGradientColumns(const Problem& problem, const double* ANVIL_RESTRICT impulse, const double* ANVIL_RESTRICT velocity, double* ANVIL_RESTRICT gradient, int first, int last)
{
	const double* ANVIL_RESTRICT values = problem.jacobian.valuePtr();
	const int* ANVIL_RESTRICT indices = problem.jacobian.innerIndexPtr();
	const int* ANVIL_RESTRICT offsets = problem.jacobian.outerIndexPtr();
	for(int column = first; column < last; ++column)
	{
		double value = 0.0;
		const int end = offsets[column + 1];
		for(int entry = offsets[column]; entry < end; ++entry)
		{
			value += values[entry] * impulse[indices[entry]];
		}
		gradient[column] = velocity[column] - value;
	}
}

#if defined(ANVIL_AVX2_FMA)
template<bool HasBody0, bool HasBody1>
static ANVIL_FORCE_INLINE void accumulateGradientScalarRun(const Problem& problem, const double* ANVIL_RESTRICT impulse,
	int first, int end, int body0, int body1, double* ANVIL_RESTRICT gradient)
{
	__m256d gradient0First = _mm256_setzero_pd(), gradient1First = _mm256_setzero_pd();
	__m128d gradient0End = _mm_setzero_pd(), gradient1End = _mm_setzero_pd();
	if(HasBody0)
	{
		gradient0First = _mm256_loadu_pd(gradient + 6 * body0);
		gradient0End = _mm_loadu_pd(gradient + 6 * body0 + 4);
	}
	if(HasBody1)
	{
		gradient1First = _mm256_loadu_pd(gradient + 6 * body1);
		gradient1End = _mm_loadu_pd(gradient + 6 * body1 + 4);
	}
	for(int contactIndex = first; contactIndex < end; ++contactIndex)
	{
		// Inactive rows contribute nothing and dominate separating or sliding contacts.
		if(impulse[contactIndex] == 0.0)
		{
			continue;
		}
		const CompactContact& contact = problem.contacts[contactIndex];
		assert(contact.row == contactIndex);
		const __m256d scale = _mm256_set1_pd(impulse[contactIndex]);
		if(HasBody0)
		{
			gradient0First = _mm256_fmadd_pd(scale, _mm256_loadu_pd(contact.jacobian[0].data()), gradient0First);
			gradient0End = _mm_fmadd_pd(_mm256_castpd256_pd128(scale), _mm_loadu_pd(contact.jacobian[0].data() + 4), gradient0End);
		}
		if(HasBody1)
		{
			gradient1First = _mm256_fmadd_pd(scale, _mm256_loadu_pd(contact.jacobian[1].data()), gradient1First);
			gradient1End = _mm_fmadd_pd(_mm256_castpd256_pd128(scale), _mm_loadu_pd(contact.jacobian[1].data() + 4), gradient1End);
		}
	}
	if(HasBody0)
	{
		_mm256_storeu_pd(gradient + 6 * body0, gradient0First);
		_mm_storeu_pd(gradient + 6 * body0 + 4, gradient0End);
	}
	if(HasBody1)
	{
		_mm256_storeu_pd(gradient + 6 * body1, gradient1First);
		_mm_storeu_pd(gradient + 6 * body1 + 4, gradient1End);
	}
}
#elif defined(ANVIL_SIMD128)
template<bool HasBody0, bool HasBody1>
static ANVIL_FORCE_INLINE void accumulateGradientScalarRun(const Problem& problem, const double* ANVIL_RESTRICT impulse,
	int first, int end, int body0, int body1, double* ANVIL_RESTRICT gradient)
{
	simd::Double2 gradient0[3] = { simd::zero(), simd::zero(), simd::zero() };
	simd::Double2 gradient1[3] = { simd::zero(), simd::zero(), simd::zero() };
	for(int pair = 0; pair < 3; ++pair)
	{
		if(HasBody0)
		{
			gradient0[pair] = simd::load(gradient + 6 * body0 + 2 * pair);
		}
		if(HasBody1)
		{
			gradient1[pair] = simd::load(gradient + 6 * body1 + 2 * pair);
		}
	}
	for(int contactIndex = first; contactIndex < end; ++contactIndex)
	{
		// Inactive rows contribute nothing and dominate separating or sliding contacts.
		if(impulse[contactIndex] == 0.0)
		{
			continue;
		}
		const CompactContact& contact = problem.contacts[contactIndex];
		assert(contact.row == contactIndex);
		const simd::Double2 scale = simd::splat(impulse[contactIndex]);
		for(int pair = 0; pair < 3; ++pair)
		{
			if(HasBody0)
			{
				gradient0[pair] = simd::multiplyAdd(scale, simd::load(contact.jacobian[0].data() + 2 * pair), gradient0[pair]);
			}
			if(HasBody1)
			{
				gradient1[pair] = simd::multiplyAdd(scale, simd::load(contact.jacobian[1].data() + 2 * pair), gradient1[pair]);
			}
		}
	}
	for(int pair = 0; pair < 3; ++pair)
	{
		if(HasBody0)
		{
			simd::store(gradient + 6 * body0 + 2 * pair, gradient0[pair]);
		}
		if(HasBody1)
		{
			simd::store(gradient + 6 * body1 + 2 * pair, gradient1[pair]);
		}
	}
}
#endif

static void evaluateGradientScalarContacts(const Problem& problem, const double* ANVIL_RESTRICT impulse, const double* ANVIL_RESTRICT velocity, double* ANVIL_RESTRICT gradient)
{
	const int columnCount = int(problem.jacobian.cols());
	std::fill_n(gradient, columnCount, 0.0);
#if defined(ANVIL_AVX2_FMA) || defined(ANVIL_SIMD128)
	const int runCount = int(problem.scalarContactRuns.size());
	for(int runIndex = 0; runIndex < runCount; ++runIndex)
	{
		const ScalarContactRun& run = problem.scalarContactRuns[runIndex];
		const int i = run.first, end = run.end;
		const CompactContact& firstContact = problem.contacts[i];
		assert(firstContact.row == i);
		if(end - i > 1 && firstContact.body[0] != firstContact.body[1])
		{
			const int body0 = firstContact.body[0], body1 = firstContact.body[1];
			if(body0 >= 0)
			{
				if(body1 >= 0)
					accumulateGradientScalarRun<true, true>(problem, impulse, i, end, body0, body1, gradient);
				else
					accumulateGradientScalarRun<true, false>(problem, impulse, i, end, body0, body1, gradient);
			}
			else
			{
				assert(body1 >= 0);
				accumulateGradientScalarRun<false, true>(problem, impulse, i, end, body0, body1, gradient);
			}
			continue;
		}
		const double scale = impulse[i];
		if(firstContact.body[0] >= 0)
		{
			addScaled6(gradient + 6 * firstContact.body[0], firstContact.jacobian[0].data(), scale);
		}
		if(firstContact.body[1] >= 0)
		{
			addScaled6(gradient + 6 * firstContact.body[1], firstContact.jacobian[1].data(), scale);
		}
	}
#else
	const int contactCount = int(problem.contacts.size());
	for(int i = 0; i < contactCount; ++i)
	{
		const CompactContact& contact = problem.contacts[i];
		assert(contact.row == i);
		const double scale = impulse[i];
		if(contact.body[0] >= 0)
		{
			addScaled6(gradient + 6 * contact.body[0], contact.jacobian[0].data(), scale);
		}
		if(contact.body[1] >= 0)
		{
			addScaled6(gradient + 6 * contact.body[1], contact.jacobian[1].data(), scale);
		}
	}
#endif
	for(int column = 0; column < columnCount; ++column)
	{
		gradient[column] = velocity[column] - gradient[column];
	}
}

static void evaluateGradientChunk(void* context, int index)
{
	GradientEvaluation& evaluation = *static_cast<GradientEvaluation*>(context);
	const int columns = int(evaluation.problem->jacobian.cols());
	const int first = columns * index / evaluation.chunks;
	const int last = columns * (index + 1) / evaluation.chunks;
	evaluateGradientColumns(*evaluation.problem, evaluation.impulse, evaluation.velocity, evaluation.gradient, first, last);
}

static void evaluatePrimalGradientCsc(const Problem& problem, ConstVector velocity, ConstVector impulse, MutableVector gradient, ParallelExecutor* parallelExecutor)
{
	if(problem.isUnilateral())
	{
		evaluateGradientScalarContacts(problem, impulse.data(), velocity.data(), gradient.data());
		return;
	}
	if(parallelExecutor != NULL && problem.bodyCount() >= 500)
	{
		const int workers = parallelExecutor->acquireWorkerCount();
		if(workers > 1)
		{
			GradientEvaluation evaluation = { &problem, impulse.data(), velocity.data(), gradient.data(), 4 * workers };
			parallelExecutor->parallelFor(evaluation.chunks, evaluateGradientChunk, &evaluation);
			return;
		}
	}
	evaluateGradientColumns(problem, impulse.data(), velocity.data(), gradient.data(), 0, int(problem.jacobian.cols()));
}
static bool evaluatePrimalFromContactVelocity(const Problem& problem, ConstVector velocity, ConstVector inverseRoot, ConstVector contactVelocity, MutableVector impulse, MutableVector gradient, Curvature* weights, PatchScratch& scratch, ParallelExecutor* parallelExecutor)
{
	if(problem.isUnilateral())
	{
		evaluateUnilateralImpulses(contactVelocity, inverseRoot, impulse, weights);
	}
	else if(!evaluateImpulses(problem, contactVelocity, inverseRoot, impulse, weights, scratch))
	{
		return false;
	}
	evaluatePrimalGradientCsc(problem, velocity, impulse, gradient, parallelExecutor);
	return true;
}

static bool evaluatePrimal(const Problem& problem, ConstVector velocity, ConstVector inverseRoot, MutableVector contactVelocity, MutableVector impulse, MutableVector gradient, Curvature* weights, PatchScratch& scratch, ParallelExecutor* parallelExecutor)
{
	multiplyJacobianCsc(problem, velocity, contactVelocity, parallelExecutor);
	contactVelocity += problem.freeVelocity;
	return evaluatePrimalFromContactVelocity(problem, velocity, inverseRoot, contactVelocity, impulse, gradient, weights, scratch, parallelExecutor);
}

#include "IncrementalCholesky.h"

static double primalCost(const Problem& problem, ConstVector velocity, ConstVector contactVelocity, ConstVector impulse)
{
	double quadratic = 0.0;
	const int rowCount = impulse.size();
#if defined(ANVIL_AVX2_FMA)
	int row = 0;
	for(; row + 4 <= rowCount; row += 4)
	{
		const __m256d values = _mm256_loadu_pd(impulse.data() + row);
		const __m256d products = _mm256_mul_pd(_mm256_mul_pd(values, _mm256_loadu_pd(problem.regularization.data() + row)), values);
		const __m128d low = _mm256_castpd256_pd128(products);
		const __m128d high = _mm256_extractf128_pd(products, 1);
		quadratic += _mm_cvtsd_f64(low);
		quadratic += _mm_cvtsd_f64(_mm_unpackhi_pd(low, low));
		quadratic += _mm_cvtsd_f64(high);
		quadratic += _mm_cvtsd_f64(_mm_unpackhi_pd(high, high));
	}
	for(; row < rowCount; ++row)
#elif defined(ANVIL_SIMD128)
	int row = 0;
	for(; row + 2 <= rowCount; row += 2)
	{
		const simd::Double2 values = simd::load(impulse.data() + row);
		const simd::Double2 products = simd::multiply(simd::multiply(values, simd::load(problem.regularization.data() + row)), values);
		quadratic += simd::low(products);
		quadratic += simd::high(products);
	}
	for(; row < rowCount; ++row)
#else
	for(int row = 0; row < rowCount; ++row)
#endif
	{
		quadratic += impulse[row] * problem.regularization[row] * impulse[row];
	}
	if(!problem.hasFiniteBounds)
	{
		return 0.5 * (velocity.squaredNorm() + quadratic);
	}
	return 0.5 * velocity.squaredNorm() - contactVelocity.dot(impulse) - 0.5 * quadratic;
}

struct AnvilStopping
{
	double tolerance;
	double lineTolerance;
	double inertiaSum;
	double costScale;

	AnvilStopping(const Problem& problem, const Settings& settings) : tolerance(settings.tolerance), lineTolerance(settings.lineTolerance), inertiaSum(problem.massDiagonal.sum()), costScale(1.0 / (inertiaSum * problem.timestep * problem.timestep)) {}

	double gradientNorm(const Problem& problem, ConstVector gradient) const
	{
		double norm = 0.0;
		const int rowCount = gradient.size();
		for(int row = 0; row < rowCount; ++row)
		{
			norm += gradient[row] * gradient[row] * problem.massDiagonal[row];
		}
		return std::sqrt(norm) / (inertiaSum * problem.timestep);
	}

};

// Whether a direction changes no body's linear or angular velocity by more than the
// tolerances. The inverse mass diagonal converts mass-scaled rows to physical velocities.
static bool withinVelocityTolerance(ConstVector direction, ConstVector inverseMassDiagonal, double linear, double angular)
{
	const double limits[2] = { linear * linear, angular * angular };
	const int count = direction.size();
	for(int row = 0; row < count; ++row)
	{
		if(direction[row] * direction[row] * inverseMassDiagonal[row] > limits[row % 6 / 3])
		{
			return false;
		}
	}
	return true;
}

struct DirectionMetrics
{
	double gradient;
	double velocity;
	double squaredNorm;
	double weightedNorm;
	double velocityInfinity;
	double directionInfinity;
};

static DirectionMetrics evaluateDirectionMetrics(ConstVector gradient, ConstVector velocity, ConstVector direction, ConstVector inverseMassDiagonal)
{
	DirectionMetrics result = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	const int count = direction.size();
#if defined(ANVIL_AVX2_FMA)
	int row = 0;
	for(; row + 4 <= count; row += 4)
	{
		const __m256d directionValues = _mm256_loadu_pd(direction.data() + row);
		const __m256d velocityValues = _mm256_loadu_pd(velocity.data() + row);
		const __m256d gradientProducts = _mm256_mul_pd(_mm256_loadu_pd(gradient.data() + row), directionValues);
		const __m256d velocityProducts = _mm256_mul_pd(velocityValues, directionValues);
		const __m256d normProducts = _mm256_mul_pd(directionValues, directionValues);
		const __m256d weightedProducts = _mm256_mul_pd(normProducts, _mm256_loadu_pd(inverseMassDiagonal.data() + row));
		const __m256d signMask = _mm256_set1_pd(-0.0);
		const __m256d velocityAbsolute = _mm256_andnot_pd(signMask, velocityValues);
		const __m256d directionAbsolute = _mm256_andnot_pd(signMask, directionValues);
		const __m128d gradientLow = _mm256_castpd256_pd128(gradientProducts), gradientHigh = _mm256_extractf128_pd(gradientProducts, 1);
		const __m128d velocityLow = _mm256_castpd256_pd128(velocityProducts), velocityHigh = _mm256_extractf128_pd(velocityProducts, 1);
		const __m128d normLow = _mm256_castpd256_pd128(normProducts), normHigh = _mm256_extractf128_pd(normProducts, 1);
		const __m128d weightedLow = _mm256_castpd256_pd128(weightedProducts), weightedHigh = _mm256_extractf128_pd(weightedProducts, 1);
		result.gradient += _mm_cvtsd_f64(gradientLow);
		result.gradient += _mm_cvtsd_f64(_mm_unpackhi_pd(gradientLow, gradientLow));
		result.gradient += _mm_cvtsd_f64(gradientHigh);
		result.gradient += _mm_cvtsd_f64(_mm_unpackhi_pd(gradientHigh, gradientHigh));
		result.velocity += _mm_cvtsd_f64(velocityLow);
		result.velocity += _mm_cvtsd_f64(_mm_unpackhi_pd(velocityLow, velocityLow));
		result.velocity += _mm_cvtsd_f64(velocityHigh);
		result.velocity += _mm_cvtsd_f64(_mm_unpackhi_pd(velocityHigh, velocityHigh));
		result.squaredNorm += _mm_cvtsd_f64(normLow);
		result.squaredNorm += _mm_cvtsd_f64(_mm_unpackhi_pd(normLow, normLow));
		result.squaredNorm += _mm_cvtsd_f64(normHigh);
		result.squaredNorm += _mm_cvtsd_f64(_mm_unpackhi_pd(normHigh, normHigh));
		result.weightedNorm += _mm_cvtsd_f64(weightedLow);
		result.weightedNorm += _mm_cvtsd_f64(_mm_unpackhi_pd(weightedLow, weightedLow));
		result.weightedNorm += _mm_cvtsd_f64(weightedHigh);
		result.weightedNorm += _mm_cvtsd_f64(_mm_unpackhi_pd(weightedHigh, weightedHigh));
		const __m128d velocityAbsoluteLow = _mm256_castpd256_pd128(velocityAbsolute), velocityAbsoluteHigh = _mm256_extractf128_pd(velocityAbsolute, 1);
		const __m128d directionAbsoluteLow = _mm256_castpd256_pd128(directionAbsolute), directionAbsoluteHigh = _mm256_extractf128_pd(directionAbsolute, 1);
		result.velocityInfinity = std::max(result.velocityInfinity, _mm_cvtsd_f64(velocityAbsoluteLow));
		result.velocityInfinity = std::max(result.velocityInfinity, _mm_cvtsd_f64(_mm_unpackhi_pd(velocityAbsoluteLow, velocityAbsoluteLow)));
		result.velocityInfinity = std::max(result.velocityInfinity, _mm_cvtsd_f64(velocityAbsoluteHigh));
		result.velocityInfinity = std::max(result.velocityInfinity, _mm_cvtsd_f64(_mm_unpackhi_pd(velocityAbsoluteHigh, velocityAbsoluteHigh)));
		result.directionInfinity = std::max(result.directionInfinity, _mm_cvtsd_f64(directionAbsoluteLow));
		result.directionInfinity = std::max(result.directionInfinity, _mm_cvtsd_f64(_mm_unpackhi_pd(directionAbsoluteLow, directionAbsoluteLow)));
		result.directionInfinity = std::max(result.directionInfinity, _mm_cvtsd_f64(directionAbsoluteHigh));
		result.directionInfinity = std::max(result.directionInfinity, _mm_cvtsd_f64(_mm_unpackhi_pd(directionAbsoluteHigh, directionAbsoluteHigh)));
	}
	for(; row < count; ++row)
#elif defined(ANVIL_SIMD128)
	// Sums keep the scalar order; maxima are order-independent and stay packed.
	simd::Double2 velocityInfinity = simd::zero(), directionInfinity = simd::zero();
	int row = 0;
	for(; row + 2 <= count; row += 2)
	{
		const simd::Double2 directionValues = simd::load(direction.data() + row);
		const simd::Double2 velocityValues = simd::load(velocity.data() + row);
		const simd::Double2 gradientProducts = simd::multiply(simd::load(gradient.data() + row), directionValues);
		const simd::Double2 velocityProducts = simd::multiply(velocityValues, directionValues);
		const simd::Double2 normProducts = simd::multiply(directionValues, directionValues);
		const simd::Double2 weightedProducts = simd::multiply(normProducts, simd::load(inverseMassDiagonal.data() + row));
		result.gradient += simd::low(gradientProducts);
		result.gradient += simd::high(gradientProducts);
		result.velocity += simd::low(velocityProducts);
		result.velocity += simd::high(velocityProducts);
		result.squaredNorm += simd::low(normProducts);
		result.squaredNorm += simd::high(normProducts);
		result.weightedNorm += simd::low(weightedProducts);
		result.weightedNorm += simd::high(weightedProducts);
		velocityInfinity = simd::maximum(velocityInfinity, simd::absolute(velocityValues));
		directionInfinity = simd::maximum(directionInfinity, simd::absolute(directionValues));
	}
	result.velocityInfinity = std::max(simd::low(velocityInfinity), simd::high(velocityInfinity));
	result.directionInfinity = std::max(simd::low(directionInfinity), simd::high(directionInfinity));
	for(; row < count; ++row)
#else
	for(int row = 0; row < count; ++row)
#endif
	{
		result.gradient += gradient[row] * direction[row];
		result.velocity += velocity[row] * direction[row];
		result.squaredNorm += direction[row] * direction[row];
		result.weightedNorm += direction[row] * direction[row] * inverseMassDiagonal[row];
		result.velocityInfinity = std::max(result.velocityInfinity, std::abs(velocity[row]));
		result.directionInfinity = std::max(result.directionInfinity, std::abs(direction[row]));
	}
	return result;
}

struct ConvexLineValue
{
	double slope;
	double curvature;
	bool valid;
};

template<bool ComputeCurvature>
static ConvexLineValue evaluateUnilateralLine(int rowCount, ConstVector contactVelocity, ConstVector contactDirection, ConstVector inverseRoot, double velocitySlope, double directionNorm, double alpha)
{
	ConvexLineValue result = { velocitySlope + alpha * directionNorm, directionNorm, true };
#if defined(ANVIL_AVX2_FMA)
	const __m256d zero = _mm256_setzero_pd();
	const __m256d alphaVector = _mm256_set1_pd(alpha);
	__m256d slopeSum = zero;
	__m256d curvatureSum = zero;
	int row = 0;
	for(; row + 4 <= rowCount; row += 4)
	{
		const __m256d direction = _mm256_loadu_pd(contactDirection.data() + row);
		const __m256d velocity = _mm256_loadu_pd(contactVelocity.data() + row);
		const __m256d drive = _mm256_sub_pd(_mm256_sub_pd(zero, velocity), _mm256_mul_pd(alphaVector, direction));
		const __m256d active = _mm256_cmp_pd(drive, zero, _CMP_GT_OQ);
		if(_mm256_testz_pd(active, active))
		{
			continue;
		}
		const __m256d root = _mm256_loadu_pd(inverseRoot.data() + row);
		const __m256d slope = _mm256_and_pd(active, _mm256_mul_pd(direction, _mm256_mul_pd(root, _mm256_mul_pd(root, drive))));
		slopeSum = _mm256_add_pd(slopeSum, slope);
		if(ComputeCurvature)
		{
			const __m256d inverseCompliance = _mm256_mul_pd(root, root);
			const __m256d curvature = _mm256_and_pd(active, _mm256_mul_pd(direction, _mm256_mul_pd(inverseCompliance, direction)));
			curvatureSum = _mm256_add_pd(curvatureSum, curvature);
		}
	}
	const __m128d slopeLow = _mm256_castpd256_pd128(slopeSum);
	const __m128d slopeHigh = _mm256_extractf128_pd(slopeSum, 1);
	result.slope -= _mm_cvtsd_f64(slopeLow) + _mm_cvtsd_f64(_mm_unpackhi_pd(slopeLow, slopeLow)) +
		_mm_cvtsd_f64(slopeHigh) + _mm_cvtsd_f64(_mm_unpackhi_pd(slopeHigh, slopeHigh));
	if(ComputeCurvature)
	{
		const __m128d curvatureLow = _mm256_castpd256_pd128(curvatureSum);
		const __m128d curvatureHigh = _mm256_extractf128_pd(curvatureSum, 1);
		result.curvature += _mm_cvtsd_f64(curvatureLow) + _mm_cvtsd_f64(_mm_unpackhi_pd(curvatureLow, curvatureLow)) +
			_mm_cvtsd_f64(curvatureHigh) + _mm_cvtsd_f64(_mm_unpackhi_pd(curvatureHigh, curvatureHigh));
	}
	for(; row < rowCount; ++row)
#elif defined(ANVIL_SIMD128)
	const simd::Double2 zero = simd::zero();
	const simd::Double2 alphaVector = simd::splat(alpha);
	simd::Double2 slopeSum = zero, curvatureSum = zero;
	int row = 0;
	for(; row + 2 <= rowCount; row += 2)
	{
		const simd::Double2 direction = simd::load(contactDirection.data() + row);
		const simd::Double2 drive = simd::subtract(simd::subtract(zero, simd::load(contactVelocity.data() + row)), simd::multiply(alphaVector, direction));
		const simd::Double2 active = simd::greater(drive, zero);
		if(!simd::any(active))
		{
			continue;
		}
		const simd::Double2 root = simd::load(inverseRoot.data() + row);
		slopeSum = simd::add(slopeSum, simd::bitAnd(active, simd::multiply(direction, simd::multiply(root, simd::multiply(root, drive)))));
		if(ComputeCurvature)
		{
			const simd::Double2 inverseCompliance = simd::multiply(root, root);
			curvatureSum = simd::add(curvatureSum, simd::bitAnd(active, simd::multiply(direction, simd::multiply(inverseCompliance, direction))));
		}
	}
	result.slope -= simd::sum(slopeSum);
	if(ComputeCurvature)
	{
		result.curvature += simd::sum(curvatureSum);
	}
	for(; row < rowCount; ++row)
#else
	for(int row = 0; row < rowCount; ++row)
#endif
	{
		const double direction = contactDirection[row];
		const double drive = -contactVelocity[row] - alpha * direction;
		if(drive > 0.0)
		{
			const double root = inverseRoot[row];
			result.slope -= direction * (root * (root * drive));
			if(ComputeCurvature)
			{
				const double inverseCompliance = root * root;
				result.curvature += direction * (inverseCompliance * direction);
			}
		}
	}
	return result;
}

// The objective/gradient evaluation immediately preceding a line search has
// already projected the impulses and assembled the active curvature diagonal.
// Reusing those values at alpha == 0 avoids reconstructing both from contact
// velocity while retaining the same row order and arithmetic grouping.
static ConvexLineValue evaluateUnilateralInitialLine(int rowCount, ConstVector contactDirection, ConstVector impulse, ConstVector diagonal, double velocitySlope, double directionNorm)
{
	ConvexLineValue result = { velocitySlope, directionNorm, true };
#if defined(ANVIL_AVX2_FMA)
	__m256d slopeSum = _mm256_setzero_pd();
	__m256d curvatureSum = _mm256_setzero_pd();
	int row = 0;
	for(; row + 4 <= rowCount; row += 4)
	{
		const __m256d inverseCompliance = _mm256_loadu_pd(diagonal.data() + row);
		const __m256i inverseComplianceBits = _mm256_castpd_si256(inverseCompliance);
		if(_mm256_testz_si256(inverseComplianceBits, inverseComplianceBits))
		{
			continue;
		}
		const __m256d direction = _mm256_loadu_pd(contactDirection.data() + row);
		const __m256d slope = _mm256_mul_pd(direction, _mm256_loadu_pd(impulse.data() + row));
		slopeSum = _mm256_add_pd(slopeSum, slope);
		const __m256d curvature = _mm256_mul_pd(direction, _mm256_mul_pd(inverseCompliance, direction));
		curvatureSum = _mm256_add_pd(curvatureSum, curvature);
	}
	const __m128d slopeLow = _mm256_castpd256_pd128(slopeSum);
	const __m128d slopeHigh = _mm256_extractf128_pd(slopeSum, 1);
	result.slope -= _mm_cvtsd_f64(slopeLow) + _mm_cvtsd_f64(_mm_unpackhi_pd(slopeLow, slopeLow)) +
		_mm_cvtsd_f64(slopeHigh) + _mm_cvtsd_f64(_mm_unpackhi_pd(slopeHigh, slopeHigh));
	const __m128d curvatureLow = _mm256_castpd256_pd128(curvatureSum);
	const __m128d curvatureHigh = _mm256_extractf128_pd(curvatureSum, 1);
	result.curvature += _mm_cvtsd_f64(curvatureLow) + _mm_cvtsd_f64(_mm_unpackhi_pd(curvatureLow, curvatureLow)) +
		_mm_cvtsd_f64(curvatureHigh) + _mm_cvtsd_f64(_mm_unpackhi_pd(curvatureHigh, curvatureHigh));
	for(; row < rowCount; ++row)
#elif defined(ANVIL_SIMD128)
	const simd::Double2 zero = simd::zero();
	simd::Double2 slopeSum = zero, curvatureSum = zero;
	int row = 0;
	for(; row + 2 <= rowCount; row += 2)
	{
		const simd::Double2 inverseCompliance = simd::load(diagonal.data() + row);
		if(!simd::any(simd::notEqual(inverseCompliance, zero)))
		{
			continue;
		}
		const simd::Double2 direction = simd::load(contactDirection.data() + row);
		slopeSum = simd::multiplyAdd(direction, simd::load(impulse.data() + row), slopeSum);
		curvatureSum = simd::add(curvatureSum, simd::multiply(direction, simd::multiply(inverseCompliance, direction)));
	}
	result.slope -= simd::sum(slopeSum);
	result.curvature += simd::sum(curvatureSum);
	for(; row < rowCount; ++row)
#else
	for(int row = 0; row < rowCount; ++row)
#endif
	{
		const double direction = contactDirection[row];
		result.slope -= direction * impulse[row];
		result.curvature += direction * (diagonal[row] * direction);
	}
	return result;
}

static bool staysInUnilateralSegment(int rowCount, ConstVector contactVelocity, ConstVector contactDirection, double alpha)
{
#if defined(ANVIL_AVX2_FMA)
	const __m256d zero = _mm256_setzero_pd();
	const __m256d alphaVector = _mm256_set1_pd(alpha);
	int row = 0;
	for(; row + 4 <= rowCount; row += 4)
	{
		const __m256d velocity = _mm256_loadu_pd(contactVelocity.data() + row);
		const __m256d direction = _mm256_loadu_pd(contactDirection.data() + row);
		const __m256d candidateVelocity = _mm256_fmadd_pd(alphaVector, direction, velocity);
		const __m256d initiallyActive = _mm256_cmp_pd(velocity, zero, _CMP_LT_OQ);
		const __m256d initiallyInactive = _mm256_cmp_pd(velocity, zero, _CMP_GE_OQ);
		const __m256d crossedActive = _mm256_and_pd(initiallyActive, _mm256_cmp_pd(candidateVelocity, zero, _CMP_GT_OQ));
		const __m256d crossedInactive = _mm256_and_pd(initiallyInactive, _mm256_cmp_pd(candidateVelocity, zero, _CMP_LT_OQ));
		if(_mm256_movemask_pd(_mm256_or_pd(crossedActive, crossedInactive)) != 0)
		{
			return false;
		}
	}
	for(; row < rowCount; ++row)
#elif defined(ANVIL_SIMD128)
	const simd::Double2 zero = simd::zero();
	const simd::Double2 alphaVector = simd::splat(alpha);
	int row = 0;
	for(; row + 2 <= rowCount; row += 2)
	{
		const simd::Double2 velocity = simd::load(contactVelocity.data() + row);
		const simd::Double2 candidateVelocity = simd::multiplyAdd(alphaVector, simd::load(contactDirection.data() + row), velocity);
		const simd::Double2 crossedActive = simd::bitAnd(simd::less(velocity, zero), simd::greater(candidateVelocity, zero));
		const simd::Double2 crossedInactive = simd::bitAnd(simd::greaterEqual(velocity, zero), simd::less(candidateVelocity, zero));
		if(simd::any(simd::bitOr(crossedActive, crossedInactive)))
		{
			return false;
		}
	}
	for(; row < rowCount; ++row)
#else
	for(int row = 0; row < rowCount; ++row)
#endif
	{
		const double velocity = contactVelocity[row];
		const double candidateVelocity = velocity + alpha * contactDirection[row];
		if((velocity < 0.0 && candidateVelocity > 0.0) || (velocity >= 0.0 && candidateVelocity < 0.0))
		{
			return false;
		}
	}
	return true;
}

static ConvexLineValue evaluateConvexLine(const Problem& problem, ConstVector contactVelocity, ConstVector contactDirection, ConstVector compliance, ConstVector inverseRoot, double velocitySlope, double directionNorm, double alpha, PatchScratch& scratch)
{
	ConvexLineValue result = { velocitySlope + alpha * directionNorm, directionNorm, true };
	std::uint32_t nextPatch = 0;
	const std::uint32_t patchCount = std::uint32_t(problem.patches.size());
	const int contactCount = int(problem.contacts.size());
	for(int i = 0; i < contactCount; ++i)
	{
		if(nextPatch < patchCount && i == problem.patches[nextPatch].firstContact)
		{
			const Patch& group = problem.patches[nextPatch];
			const int first = problem.contacts[i].row;
			const int count = group.normalCount + group.tangentCount;
			for(int row = 0; row < count; ++row)
			{
				scratch.velocity[row] = contactVelocity[first + row] + alpha * contactDirection[first + row];
			}
			PatchProjectionResult projection;
			if(!projectGroup(problem, group, scratch.velocity.data(), scratch.impulse.data(), scratch.diagonal.data(), scratch.coupling, scratch, projection))
			{
				result.valid = false;
				return result;
			}
			double normalDirection = 0.0, tangentDirection = 0.0;
			for(int row = 0; row < count; ++row)
			{
				const double direction = contactDirection[first + row];
				result.slope -= direction * scratch.impulse[row];
				result.curvature += direction * direction * scratch.diagonal[row];
				if(row < group.normalCount)
				{
					normalDirection += direction * scratch.diagonal[row];
				}
				else
				{
					tangentDirection += direction * scratch.coupling[row - group.normalCount];
				}
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
				const double inverseCompliance = inverseRoot[row] * inverseRoot[row];
				const double unconstrained = drive * inverseCompliance;
				result.slope -= direction * clampValue(unconstrained, limits.lower, limits.upper);
				if(unconstrained > limits.lower && unconstrained < limits.upper)
				{
					result.curvature += direction * direction * inverseCompliance;
				}
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
			const Vec3 impulse = weightedImpulse(-contactVelocity.segment<3>(row) - alpha * direction, compliance.data() + row, inverseRoot.data() + row, problem.block(contact).friction, &derivative, contact.block < 0, problem.block(contact).maxNormalImpulse);
			result.slope -= direction.dot(impulse);
			result.curvature += direction.dot(derivative * direction);
		}
	}

	return result;
}

// Inputs of one line search; unilateral problems use the specialized scalar evaluation.
struct LineInput
{
	const Problem* problem;
	const VectorStorage* contactVelocity;
	const VectorStorage* contactDirection;
	const VectorStorage* compliance;
	const VectorStorage* inverseRoot;
	double velocitySlope;
	double directionNorm;
	PatchScratch* scratch;
	bool unilateral;
};

static ConvexLineValue evaluateLine(const LineInput& line, double alpha, bool computeCurvature)
{
	if(!line.unilateral)
	{
		return evaluateConvexLine(*line.problem, *line.contactVelocity, *line.contactDirection, *line.compliance, *line.inverseRoot, line.velocitySlope, line.directionNorm, alpha, *line.scratch);
	}
	const int rows = line.problem->rowCount();
	return computeCurvature ? evaluateUnilateralLine<true>(rows, *line.contactVelocity, *line.contactDirection, *line.inverseRoot, line.velocitySlope, line.directionNorm, alpha) :
		evaluateUnilateralLine<false>(rows, *line.contactVelocity, *line.contactDirection, *line.inverseRoot, line.velocitySlope, line.directionNorm, alpha);
}

static bool searchConvex(const Problem& problem, ConstVector direction, ConstVector contactVelocity, ConstVector impulse, const Curvature& weights, ConstVector inverseRoot, MutableVector contactDirection, int& evaluations, PatchScratch& scratch, ParallelExecutor* parallelExecutor, double velocitySlope, double directionNorm, double slopeTolerance, double& alphaResult)
{
	// Reuse the contact velocity from the current objective/gradient evaluation.
	ConstVector compliance = problem.regularization;
	multiplyJacobianCsc(problem, direction, contactDirection, parallelExecutor);
	const bool unilateral = problem.isUnilateral();
	const LineInput line = { &problem, &contactVelocity, &contactDirection, &compliance, &inverseRoot, velocitySlope, directionNorm, &scratch, unilateral };
	const ConvexLineValue initial = unilateral ? evaluateUnilateralInitialLine(problem.rowCount(), contactDirection, impulse, weights.diagonal, velocitySlope, directionNorm) : evaluateLine(line, 0.0, true);
	++evaluations;
	if(!initial.valid)
	{
		return false;
	}
	if(initial.slope >= 0.0)
	{
		alphaResult = 0.0;
		return true;
	}
	const double initialCandidate = -initial.slope / initial.curvature;
	if(unilateral && staysInUnilateralSegment(problem.rowCount(), contactVelocity, contactDirection, initialCandidate))
	{
		alphaResult = initialCandidate;
		return true;
	}
	double lower = 0.0, upper = 1.0;
	bool upperHasCurvature = unilateral && initialCandidate >= upper;
	ConvexLineValue atUpper = evaluateLine(line, upper, upperHasCurvature);
	++evaluations;
	if(!atUpper.valid)
	{
		return false;
	}
	for(int i = 0; atUpper.slope < 0.0 && i < 50; ++i)
	{
		lower = upper;
		upper *= 2.0;
		upperHasCurvature = unilateral && initialCandidate >= upper;
		atUpper = evaluateLine(line, upper, upperHasCurvature);
		++evaluations;
		if(!atUpper.valid)
		{
			return false;
		}
	}
	double alpha = clampValue(-initial.slope / initial.curvature, lower, upper);
	for(int i = 0; i < 50; ++i)
	{
		const bool reuseUpper = i == 0 && alpha == upper && upperHasCurvature;
		const ConvexLineValue value = reuseUpper ? atUpper : evaluateLine(line, alpha, true);
		evaluations += !reuseUpper;
		if(!value.valid)
		{
			return false;
		}
		if(std::abs(value.slope) <= (slopeTolerance > 0.0 ? slopeTolerance : 1.0e-10 * std::max(1.0, std::abs(initial.slope))))
		{
			alphaResult = alpha;
			return true;
		}
		if(value.slope < 0.0)
		{
			lower = alpha;
		}
		else
		{
			upper = alpha;
		}
		const double candidate = alpha - value.slope / value.curvature;
		alpha = candidate > lower && candidate < upper ? candidate : 0.5 * (lower + upper);
	}
	alphaResult = alpha;
	return true;
}

// Stiff unilateral rows make Anvil's piecewise-quadratic objective change curvature
// by 1/R at every kink, so an exact line search stops after a few active-set changes.
// When Anvil has not converged quickly, a primal-dual interior-point method follows
// the smooth central path instead. With multipliers lambda >= 0 and gaps w >= 0,
//   v = J' lambda,   w = J v + f + R lambda,   lambda_i w_i = mu,
// and each Mehrotra predictor-corrector step solves (I + J' D J) dv = rhs with
// D = lambda / (w + R lambda), the Anvil Hessian shape with weights in (0, 1/R).
// It returns once the stiff objective's scaled gradient is below exitGradient;
// ordinary Anvil iterations then finish from that point.
struct InteriorPointData
{
	VectorStorage lambda, slack, weight, inverseLambda, inverseSlack, primalResidual, scaled, target;
	VectorStorage affineLambda, affineSlack, stepLambda, stepSlack, stepContact;
	VectorStorage dualResidual, stepVelocity, rhs, zero;
	// The multipliers belong to the problem last solved; continuations may warm start.
	bool warm = false;
};

#if defined(ANVIL_AVX2_FMA)
static ANVIL_FORCE_INLINE double sum4(__m256d value)
{
	const __m128d halves = _mm_add_pd(_mm256_castpd256_pd128(value), _mm256_extractf128_pd(value, 1));
	return _mm_cvtsd_f64(_mm_add_pd(halves, _mm_unpackhi_pd(halves, halves)));
}

static ANVIL_FORCE_INLINE double max4(__m256d value)
{
	const __m128d halves = _mm_max_pd(_mm256_castpd256_pd128(value), _mm256_extractf128_pd(value, 1));
	return _mm_cvtsd_f64(_mm_max_pd(halves, _mm_unpackhi_pd(halves, halves)));
}
#endif

// Solve for the velocity step of the current `scaled` right-hand side, then recover
// multiplier and gap steps: dl = scaled - D J dv, dw = (target - w dl) / lambda.
// Returns the largest ratio max(-dl / lambda, -dw / w), i.e. the inverse step to
// the boundary, and accumulates sum(dl * dw).
static double interiorPointDirection(const Problem& problem, IncrementalCholesky& factor, InteriorPointData& data, ParallelExecutor* parallelExecutor,
	double* ANVIL_RESTRICT stepLambda, double* ANVIL_RESTRICT stepSlack, double& stepProduct)
{
	const int rows = problem.rowCount(), bodies = problem.bodyCount() * 6;
	// The factor solves H x = -gradient, so pass dualResidual - J' scaled.
	evaluatePrimalGradientCsc(problem, data.zero, data.scaled, data.rhs, parallelExecutor);
	for(int i = 0; i < bodies; ++i)
	{
		data.rhs[i] += data.dualResidual[i];
	}
	factor.solveDirection(data.rhs, data.stepVelocity);
	multiplyJacobianCsc(problem, data.stepVelocity, data.stepContact, parallelExecutor);
	const double* ANVIL_RESTRICT scaled = data.scaled.data();
	const double* ANVIL_RESTRICT weight = data.weight.data();
	const double* ANVIL_RESTRICT contact = data.stepContact.data();
	const double* ANVIL_RESTRICT target = data.target.data();
	const double* ANVIL_RESTRICT slack = data.slack.data();
	const double* ANVIL_RESTRICT inverseLambda = data.inverseLambda.data();
	const double* ANVIL_RESTRICT inverseSlack = data.inverseSlack.data();
	double ratio = 0.0, product = 0.0;
	int row = 0;
#if defined(ANVIL_AVX2_FMA)
	const __m256d zero = _mm256_setzero_pd();
	__m256d ratios = zero, products = zero;
	for(; row + 4 <= rows; row += 4)
	{
		const __m256d dl = _mm256_fnmadd_pd(_mm256_loadu_pd(weight + row), _mm256_loadu_pd(contact + row), _mm256_loadu_pd(scaled + row));
		const __m256d dw = _mm256_mul_pd(_mm256_fnmadd_pd(_mm256_loadu_pd(slack + row), dl, _mm256_loadu_pd(target + row)), _mm256_loadu_pd(inverseLambda + row));
		_mm256_storeu_pd(stepLambda + row, dl);
		_mm256_storeu_pd(stepSlack + row, dw);
		ratios = _mm256_max_pd(ratios, _mm256_max_pd(_mm256_mul_pd(_mm256_sub_pd(zero, dl), _mm256_loadu_pd(inverseLambda + row)),
			_mm256_mul_pd(_mm256_sub_pd(zero, dw), _mm256_loadu_pd(inverseSlack + row))));
		products = _mm256_fmadd_pd(dl, dw, products);
	}
	ratio = max4(ratios);
	product = sum4(products);
#elif defined(ANVIL_SIMD128)
	const simd::Double2 zero = simd::zero();
	simd::Double2 ratios = zero, products = zero;
	for(; row + 2 <= rows; row += 2)
	{
		const simd::Double2 dl = simd::negativeMultiplyAdd(simd::load(weight + row), simd::load(contact + row), simd::load(scaled + row));
		const simd::Double2 dw = simd::multiply(simd::negativeMultiplyAdd(simd::load(slack + row), dl, simd::load(target + row)), simd::load(inverseLambda + row));
		simd::store(stepLambda + row, dl);
		simd::store(stepSlack + row, dw);
		ratios = simd::maximum(ratios, simd::maximum(simd::multiply(simd::subtract(zero, dl), simd::load(inverseLambda + row)),
			simd::multiply(simd::subtract(zero, dw), simd::load(inverseSlack + row))));
		products = simd::multiplyAdd(dl, dw, products);
	}
	ratio = std::max(simd::low(ratios), simd::high(ratios));
	product = simd::sum(products);
#endif
	for(; row < rows; ++row)
	{
		const double dl = scaled[row] - weight[row] * contact[row];
		const double dw = (target[row] - slack[row] * dl) * inverseLambda[row];
		stepLambda[row] = dl;
		stepSlack[row] = dw;
		ratio = std::max(ratio, std::max(-dl * inverseLambda[row], -dw * inverseSlack[row]));
		product += dl * dw;
	}
	stepProduct = product;
	return ratio;
}

// velocity/contactVelocity are updated in place; impulse and gradient hold the stiff
// objective's projected impulses and gradient at the returned point.
static int solveInteriorPoint(const Problem& problem, const Settings& settings, const AnvilStopping& stopping, IncrementalCholesky& factor, Curvature& weights,
	InteriorPointData& data, VectorStorage& velocity, VectorStorage& contactVelocity, ConstVector inverseRoot, VectorStorage& impulse, VectorStorage& gradient,
	int maxSteps, double exitGradient, double warmShift, Result& result, bool& failed)
{
	failed = false;
	const bool warm = warmShift >= 0.0 && data.warm && data.lambda.size() == problem.rowCount();
	const int rows = problem.rowCount(), bodies = problem.bodyCount() * 6;
	VectorStorage* rowVectors[] = { &data.lambda, &data.slack, &data.weight, &data.inverseLambda, &data.inverseSlack, &data.primalResidual, &data.scaled, &data.target,
		&data.affineLambda, &data.affineSlack, &data.stepLambda, &data.stepSlack, &data.stepContact };
	for(VectorStorage* vector : rowVectors)
	{
		vector->resize(rows);
	}
	data.dualResidual.resize(bodies);
	data.stepVelocity.resize(bodies);
	data.rhs.resize(bodies);
	data.zero.setZero(bodies);
	double* ANVIL_RESTRICT lambda = data.lambda.data();
	double* ANVIL_RESTRICT slack = data.slack.data();
	double* ANVIL_RESTRICT weight = data.weight.data();
	double* ANVIL_RESTRICT inverseLambda = data.inverseLambda.data();
	double* ANVIL_RESTRICT inverseSlack = data.inverseSlack.data();
	double* ANVIL_RESTRICT primalResidual = data.primalResidual.data();
	double* ANVIL_RESTRICT scaled = data.scaled.data();
	double* ANVIL_RESTRICT target = data.target.data();
	double* ANVIL_RESTRICT contact = contactVelocity.data();
	const double* ANVIL_RESTRICT compliance = problem.regularization.data();
	// A cold start uses the penalty multipliers and gaps at the current velocity,
	// shifted by their mean magnitudes into a well-centered interior point. A warm
	// start keeps the previous multipliers, recomputes gaps for the changed free
	// velocities and shifts both by a small fraction of their means.
	multiplyJacobianCsc(problem, velocity, contactVelocity, settings.parallelExecutor);
	contactVelocity += problem.freeVelocity;
	double lambdaMean = 0.0, slackMean = 0.0;
	for(int row = 0; row < rows; ++row)
	{
		const double s = contact[row];
		if(!warm)
		{
			lambda[row] = s < 0.0 ? -s * inverseRoot[row] * inverseRoot[row] : 0.0;
		}
		slack[row] = std::max(0.0, s + compliance[row] * lambda[row]);
		lambdaMean += lambda[row];
		slackMean += slack[row];
	}
	const double shift = warm ? warmShift : 1.0;
	lambdaMean = std::max(shift * lambdaMean / rows, 1.0e-12);
	slackMean = std::max(shift * slackMean / rows, 1.0e-12);
	for(int row = 0; row < rows; ++row)
	{
		lambda[row] += lambdaMean;
		slack[row] += slackMean;
	}
	data.warm = true;
	double* ANVIL_RESTRICT diagonal = weights.diagonal.data();
	const double drop = settings.interiorPointDrop;
	int steps = 0;
	while(steps < maxSteps)
	{
		// dualResidual = v - J' lambda
		evaluatePrimalGradientCsc(problem, velocity, data.lambda, data.dualResidual, settings.parallelExecutor);
		// Residuals, weights and reciprocals; the predictor targets zero complementarity,
		// so its right-hand side is D (rp - w).
		double complementarity = 0.0;
		int row = 0;
#if defined(ANVIL_AVX2_FMA)
		{
			const __m256d one = _mm256_set1_pd(1.0), dropVector = _mm256_set1_pd(drop);
			__m256d sum = _mm256_setzero_pd();
			for(; row + 4 <= rows; row += 4)
			{
				const __m256d l = _mm256_loadu_pd(lambda + row), w = _mm256_loadu_pd(slack + row), r = _mm256_loadu_pd(compliance + row);
				const __m256d rp = _mm256_sub_pd(_mm256_sub_pd(w, _mm256_loadu_pd(contact + row)), _mm256_mul_pd(r, l));
				const __m256d lw = _mm256_mul_pd(l, w);
				sum = _mm256_add_pd(sum, lw);
				const __m256d d = _mm256_div_pd(l, _mm256_fmadd_pd(r, l, w));
				_mm256_storeu_pd(primalResidual + row, rp);
				_mm256_storeu_pd(weight + row, d);
				// Negligible curvature from clearly separated rows is omitted from the
				// factored Hessian only; the step equations keep every exact weight.
				_mm256_storeu_pd(diagonal + row, _mm256_and_pd(_mm256_cmp_pd(_mm256_mul_pd(d, r), dropVector, _CMP_GT_OQ), d));
				_mm256_storeu_pd(inverseLambda + row, _mm256_div_pd(one, l));
				_mm256_storeu_pd(inverseSlack + row, _mm256_div_pd(one, w));
				_mm256_storeu_pd(target + row, _mm256_sub_pd(_mm256_setzero_pd(), lw));
				_mm256_storeu_pd(scaled + row, _mm256_mul_pd(d, _mm256_sub_pd(rp, w)));
			}
			complementarity = sum4(sum);
		}
#elif defined(ANVIL_SIMD128)
		{
			const simd::Double2 one = simd::splat(1.0), dropVector = simd::splat(drop), zero = simd::zero();
			simd::Double2 sum = zero;
			for(; row + 2 <= rows; row += 2)
			{
				const simd::Double2 l = simd::load(lambda + row), w = simd::load(slack + row), r = simd::load(compliance + row);
				const simd::Double2 rp = simd::subtract(simd::subtract(w, simd::load(contact + row)), simd::multiply(r, l));
				const simd::Double2 lw = simd::multiply(l, w);
				sum = simd::add(sum, lw);
				const simd::Double2 d = simd::divide(l, simd::multiplyAdd(r, l, w));
				simd::store(primalResidual + row, rp);
				simd::store(weight + row, d);
				simd::store(diagonal + row, simd::bitAnd(simd::greater(simd::multiply(d, r), dropVector), d));
				simd::store(inverseLambda + row, simd::divide(one, l));
				simd::store(inverseSlack + row, simd::divide(one, w));
				simd::store(target + row, simd::subtract(zero, lw));
				simd::store(scaled + row, simd::multiply(d, simd::subtract(rp, w)));
			}
			complementarity = simd::sum(sum);
		}
#endif
		for(; row < rows; ++row)
		{
			const double rp = slack[row] - contact[row] - compliance[row] * lambda[row];
			const double d = lambda[row] / (slack[row] + compliance[row] * lambda[row]);
			complementarity += lambda[row] * slack[row];
			primalResidual[row] = rp;
			weight[row] = d;
			diagonal[row] = d * compliance[row] > drop ? d : 0.0;
			inverseLambda[row] = 1.0 / lambda[row];
			inverseSlack[row] = 1.0 / slack[row];
			target[row] = -lambda[row] * slack[row];
			scaled[row] = d * (rp - slack[row]);
		}
		const double mu = complementarity / rows;
		const bool factored = factor.factorFresh(problem, weights, result);
		if(!factored)
		{
			failed = true;
			return steps;
		}
		// The factor retains `weights`; its former storage is now this scratch diagonal.
		diagonal = weights.diagonal.data();
		const Clock::time_point solveStart = profileStart(settings.profile);
		double affineProduct;
		const double affineRatio = interiorPointDirection(problem, factor, data, settings.parallelExecutor, data.affineLambda.data(), data.affineSlack.data(), affineProduct);
		const double affineAlpha = affineRatio > 1.0 ? 1.0 / affineRatio : 1.0;
		// The predictor satisfies lambda dw + w dl = -lambda w exactly.
		const double affineMu = std::max(0.0, ((1.0 - affineAlpha) * complementarity + affineAlpha * affineAlpha * affineProduct) / rows);
		const double centeringRatio = affineMu / mu;
		const double centering = centeringRatio * centeringRatio * centeringRatio;
		// Corrector with Mehrotra's second-order term.
		const double* ANVIL_RESTRICT affineLambda = data.affineLambda.data();
		const double* ANVIL_RESTRICT affineSlack = data.affineSlack.data();
		const double centeredTarget = centering * mu;
		for(row = 0; row < rows; ++row)
		{
			const double value = centeredTarget + target[row] - affineLambda[row] * affineSlack[row];
			target[row] = value;
			scaled[row] = weight[row] * (value * inverseLambda[row] + primalResidual[row]);
		}
		double product;
		const double ratio = interiorPointDirection(problem, factor, data, settings.parallelExecutor, data.stepLambda.data(), data.stepSlack.data(), product);
		result.backsolveMs += profileElapsed(settings.profile, solveStart);
		const double alpha = ratio > 0.99 ? 0.99 / ratio : 1.0;
		for(int i = 0; i < bodies; ++i)
		{
			velocity[i] += alpha * data.stepVelocity[i];
		}
		const double* ANVIL_RESTRICT stepLambda = data.stepLambda.data();
		const double* ANVIL_RESTRICT stepSlack = data.stepSlack.data();
		const double* ANVIL_RESTRICT stepContact = data.stepContact.data();
		double* ANVIL_RESTRICT penalty = impulse.data();
		row = 0;
#if defined(ANVIL_AVX2_FMA)
		{
			const __m256d step = _mm256_set1_pd(alpha), zero = _mm256_setzero_pd();
			for(; row + 4 <= rows; row += 4)
			{
				_mm256_storeu_pd(lambda + row, _mm256_fmadd_pd(step, _mm256_loadu_pd(stepLambda + row), _mm256_loadu_pd(lambda + row)));
				_mm256_storeu_pd(slack + row, _mm256_fmadd_pd(step, _mm256_loadu_pd(stepSlack + row), _mm256_loadu_pd(slack + row)));
				const __m256d s = _mm256_fmadd_pd(step, _mm256_loadu_pd(stepContact + row), _mm256_loadu_pd(contact + row));
				_mm256_storeu_pd(contact + row, s);
				// Projected impulses of the stiff objective at the new velocity.
				const __m256d root = _mm256_loadu_pd(inverseRoot.data() + row);
				_mm256_storeu_pd(penalty + row, _mm256_mul_pd(root, _mm256_max_pd(zero, _mm256_mul_pd(_mm256_sub_pd(zero, root), s))));
			}
		}
#elif defined(ANVIL_SIMD128)
		{
			const simd::Double2 step = simd::splat(alpha), zero = simd::zero();
			for(; row + 2 <= rows; row += 2)
			{
				simd::store(lambda + row, simd::multiplyAdd(step, simd::load(stepLambda + row), simd::load(lambda + row)));
				simd::store(slack + row, simd::multiplyAdd(step, simd::load(stepSlack + row), simd::load(slack + row)));
				const simd::Double2 s = simd::multiplyAdd(step, simd::load(stepContact + row), simd::load(contact + row));
				simd::store(contact + row, s);
				// Projected impulses of the stiff objective at the new velocity.
				const simd::Double2 root = simd::load(inverseRoot.data() + row);
				simd::store(penalty + row, simd::multiply(root, simd::maximum(zero, simd::multiply(simd::subtract(zero, root), s))));
			}
		}
#endif
		for(; row < rows; ++row)
		{
			lambda[row] += alpha * stepLambda[row];
			slack[row] += alpha * stepSlack[row];
			contact[row] += alpha * stepContact[row];
			penalty[row] = inverseRoot[row] * std::max(0.0, -inverseRoot[row] * contact[row]);
		}
		++steps;
		evaluatePrimalGradientCsc(problem, velocity, impulse, gradient, settings.parallelExecutor);
		result.scaledGradient = stopping.gradientNorm(problem, gradient);
		if(result.scaledGradient < exitGradient)
		{
			break;
		}
	}
	return steps;
}

struct WorkspaceData
{
	VectorStorage velocity, impulse, gradient, contactVelocity, inverseRoot, coldImpulse, nextVelocity, direction, contactDirection;
	Curvature weights;
	PatchScratch patchScratch;
	IncrementalCholesky factor;
	std::vector<int> changedRows;
	InteriorPointData interiorPoint;
	const Problem* continuationProblem = NULL;
	std::uint64_t continuationGeneration = 0;
};

static SolveStatus::Enum solveAnvilInternal(const Problem& problem, const Settings& settings, Result& result, WorkspaceData& workspace, const Result* previous, bool continuation)
{
	static_cast<SolverStatistics&>(result) = SolverStatistics();
	const int expectedSize = problem.bodyCount() * 6;
	assert(settings.iterations >= 0 && settings.tolerance >= 0.0 && settings.lineTolerance >= 0.0);
	assert(problem.prepared && problem.timestep > 0.0 && problem.massDiagonal.size() == expectedSize && problem.jacobian.cols() == expectedSize && problem.jacobian.rows() == problem.rowCount() && problem.regularization.size() == problem.rowCount() && (!previous || previous->primal.size() == expectedSize));
	if(expectedSize == 0)
	{
		assert(problem.rowCount() == 0);
		result.primal.resize(0);
		result.impulse.resize(0);
		result.stopReason = 1;
		return SolveStatus::eSUCCESS;
	}
	ConstVector compliance = problem.regularization;
	const AnvilStopping stopping(problem, settings);
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
	inverseRoot.resize(rows);
	workspace.contactDirection.resize(rows);
	if(previous)
	{
		velocity = previous->primal;
	}
	else
	{
		velocity.setZero();
	}
	impulse.resize(problem.rowCount());
	weights.resize(problem);
	workspace.patchScratch.resize(problem);
	// Correctly rounded square roots and divisions match the scalar loop exactly.
	int rootRow = 0;
#if defined(ANVIL_AVX2_FMA)
	for(; rootRow + 4 <= rows; rootRow += 4)
	{
		_mm256_storeu_pd(inverseRoot.data() + rootRow, _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(_mm256_loadu_pd(compliance.data() + rootRow))));
	}
#elif defined(ANVIL_SIMD128)
	for(; rootRow + 2 <= rows; rootRow += 2)
	{
		simd::store(inverseRoot.data() + rootRow, simd::divide(simd::splat(1.0), simd::squareRoot(simd::load(compliance.data() + rootRow))));
	}
#endif
	for(; rootRow < rows; ++rootRow)
	{
		inverseRoot[rootRow] = 1.0 / std::sqrt(compliance[rootRow]);
	}
	factor.beginSolve(settings.profile, continuation, settings.parallelExecutor);
	// Continuations of an interior-point solve restart the interior point from its
	// previous multipliers instead of repeating the Anvil phase.
	const bool warmInteriorPoint = continuation && workspace.interiorPoint.warm && settings.interiorPointWarmShift >= 0.0;
	if(!continuation)
	{
		workspace.interiorPoint.warm = false;
	}
	Clock::time_point evaluationStart = profileStart(settings.profile);
	if(!evaluatePrimal(problem, velocity, inverseRoot, contactVelocity, impulse, gradient, &weights, workspace.patchScratch, settings.parallelExecutor))
	{
		return SolveStatus::eNUMERICAL_FAILURE;
	}
	if(previous)
	{
		// A cached solution can be worse than free motion after contacts or loads change.
		VectorStorage& coldImpulse = workspace.coldImpulse;
		coldImpulse.resize(problem.rowCount());
		// Comparing warm and free motion needs only the cold impulses and cost.
		if(problem.isUnilateral())
		{
			evaluateUnilateralImpulses(problem.freeVelocity, inverseRoot, coldImpulse, NULL);
		}
		else if(!evaluateImpulses(problem, problem.freeVelocity, inverseRoot, coldImpulse, NULL, workspace.patchScratch))
		{
			return SolveStatus::eNUMERICAL_FAILURE;
		}
		double warmCost = velocity.squaredNorm(), coldCost = 0.0;
		for(int row = 0; row < rows; ++row)
		{
			warmCost += impulse[row] * compliance[row] * impulse[row];
			coldCost += coldImpulse[row] * compliance[row] * coldImpulse[row];
		}
		if(problem.hasFiniteBounds)
		{
			warmCost = 2.0 * primalCost(problem, velocity, contactVelocity, impulse);
			coldCost = -2.0 * problem.freeVelocity.dot(coldImpulse) - coldCost;
		}
		if(warmCost > coldCost)
		{
			velocity.setZero();
			if(!evaluatePrimal(problem, velocity, inverseRoot, contactVelocity, impulse, gradient, &weights, workspace.patchScratch, settings.parallelExecutor))
			{
				return SolveStatus::eNUMERICAL_FAILURE;
			}
		}
	}
	result.evaluationMs += profileElapsed(settings.profile, evaluationStart);
	double cost = primalCost(problem, velocity, contactVelocity, impulse);
	const bool unilateral = problem.isUnilateral();
	std::vector<int>& changedRows = workspace.changedRows;
	if(unilateral)
	{
		reserveStorage(changedRows, std::uint32_t(rows));
		changedRows.resize(rows);
	}
	// Fused unilateral evaluations list the curvature rows that differ from the factor.
	int changedCount = -1;
	bool interiorPointUsed = false;
	// Consecutive short line-search steps indicate Anvil crossing kinks slowly.
	int shortSteps = 0;
	for(int iteration = 0; iteration < settings.iterations; ++iteration)
	{
		result.scaledGradient = stopping.gradientNorm(problem, gradient);
		if(result.scaledGradient < stopping.tolerance)
		{
			result.stopReason = 1;
			break;
		}
		if(unilateral && !interiorPointUsed && settings.interiorPointSwitch >= 0 && rows > 0 && (warmInteriorPoint ||
			(iteration >= settings.interiorPointSwitch && (shortSteps >= 3 || iteration >= 4 * settings.interiorPointSwitch))))
		{
			interiorPointUsed = true;
			const Clock::time_point interiorStart = profileStart(settings.profile);
			bool failed;
			const int steps = solveInteriorPoint(problem, settings, stopping, factor, weights, workspace.interiorPoint, velocity, contactVelocity, inverseRoot, impulse, gradient,
				settings.iterations - iteration, std::max(stopping.tolerance, settings.interiorPointExit), warmInteriorPoint ? settings.interiorPointWarmShift : -1.0, result, failed);
			result.interiorPointSteps += steps;
			result.evaluationMs += profileElapsed(settings.profile, interiorStart);
			if(failed)
			{
				return SolveStatus::eFACTORIZATION_FAILED;
			}
			// Resume Anvil from the interior-point velocity with exact curvature.
			if(!evaluatePrimal(problem, velocity, inverseRoot, contactVelocity, impulse, gradient, &weights, workspace.patchScratch, settings.parallelExecutor))
			{
				return SolveStatus::eNUMERICAL_FAILURE;
			}
			cost = primalCost(problem, velocity, contactVelocity, impulse);
			changedCount = -1;
			iteration += steps - 1;
			result.iterations = iteration + 1;
			continue;
		}
		if(!factor.factor(problem, weights, result, changedCount >= 0 ? changedRows.data() : NULL, changedCount))
		{
			return SolveStatus::eFACTORIZATION_FAILED;
		}
		const Clock::time_point solveStart = profileStart(settings.profile);
		factor.solveDirection(gradient, direction);
		result.backsolveMs += profileElapsed(settings.profile, solveStart);
		if(settings.checkFactor)
		{
			// This independent check is outside performance runs. It catches errors
			// in signed updates and permutation by rebuilding the current Hessian.
			HessianStorage referenceStorage;
			const Sparse& matrix = makeHessian(problem, factor.currentWeights(), referenceStorage);
			Vector expected;
			if(!solveDenseReference(matrix, gradient, expected))
			{
				return SolveStatus::eNUMERICAL_FAILURE;
			}
			double difference = 0.0;
			for(int row = 0; row < bodies; ++row)
			{
				difference = std::max(difference, std::abs(direction[row] - expected[row]));
			}
			const double error = difference / std::max(1.0, expected.infinityNorm());
			result.factorError = std::max(result.factorError, error);
			if(error > 1.0e-6)
			{
				return SolveStatus::eNUMERICAL_FAILURE;
			}
		}
		// Very stiff constraints can leave a cancellation residual after motion
		// has converged. Stop when the computed correction reaches roundoff.
		DirectionMetrics directionMetrics = evaluateDirectionMetrics(gradient, velocity, direction, problem.inverseMassDiagonal);
		if(directionMetrics.directionInfinity <= 1.0e-14 * std::max(1.0, directionMetrics.velocityInfinity))
		{
			result.stopReason = 5;
			break;
		}
		if(settings.velocityTolerance > 0.0 && withinVelocityTolerance(direction, problem.inverseMassDiagonal, settings.velocityTolerance, settings.angularVelocityTolerance))
		{
			result.stopReason = 7;
			break;
		}
		if(directionMetrics.gradient >= 0.0)
		{
			for(int row = 0; row < bodies; ++row)
			{
				direction[row] = -gradient[row];
			}
			directionMetrics = evaluateDirectionMetrics(gradient, velocity, direction, problem.inverseMassDiagonal);
		}
		const Clock::time_point lineStart = profileStart(settings.profile);
		double alpha;
		const double lineThreshold = stopping.tolerance * stopping.lineTolerance * problem.timestep * stopping.inertiaSum * std::sqrt(directionMetrics.weightedNorm);
		if(!searchConvex(problem, direction, contactVelocity, impulse, factor.currentWeights(), inverseRoot, workspace.contactDirection, result.lineSearchEvaluations, workspace.patchScratch, settings.parallelExecutor, directionMetrics.velocity, directionMetrics.squaredNorm, lineThreshold, alpha))
		{
			return SolveStatus::eNUMERICAL_FAILURE;
		}
		result.lineSearchMs += profileElapsed(settings.profile, lineStart);
		shortSteps = alpha < 0.5 ? shortSteps + 1 : 0;
		bool unchanged = true;
		for(int row = 0; row < bodies; ++row)
		{
			nextVelocity[row] = velocity[row] + alpha * direction[row];
			unchanged = unchanged && nextVelocity[row] == velocity[row];
		}
		if(unchanged)
		{
			result.stopReason = 6;
			break;
		}
		velocity.swap(nextVelocity);
		result.iterations = iteration + 1;
		evaluationStart = profileStart(settings.profile);
		// The line search has already computed J * direction. Reuse it for
		// J * (velocity + alpha * direction), periodically rebuilding from
		// velocity to bound accumulated floating-point error.
		const bool refreshContactVelocity = (iteration & 7) == 7;
		if(unilateral)
		{
			if(refreshContactVelocity)
			{
				multiplyJacobianCsc(problem, velocity, contactVelocity, settings.parallelExecutor);
				contactVelocity += problem.freeVelocity;
			}
			const double quadratic = evaluateUnilateralStep(rows, alpha, !refreshContactVelocity, contactVelocity.data(), workspace.contactDirection.data(), inverseRoot.data(),
				impulse.data(), weights.diagonal.data(), factor.currentWeights().diagonal.data(), changedRows.data(), changedCount);
			evaluatePrimalGradientCsc(problem, velocity, impulse, gradient, settings.parallelExecutor);
			result.evaluationMs += profileElapsed(settings.profile, evaluationStart);
			const double nextCost = 0.5 * (velocity.squaredNorm() + quadratic);
			result.scaledImprovement = stopping.costScale * (cost - nextCost);
			cost = nextCost;
			if(result.scaledImprovement > 0.0 && result.scaledImprovement < stopping.tolerance)
			{
				result.stopReason = 2;
				break;
			}
			continue;
		}
		if(!refreshContactVelocity)
		{
			for(int row = 0; row < rows; ++row)
			{
				contactVelocity[row] += alpha * workspace.contactDirection[row];
			}
		}
		if(!(refreshContactVelocity ?
			evaluatePrimal(problem, velocity, inverseRoot, contactVelocity, impulse, gradient, &weights, workspace.patchScratch, settings.parallelExecutor) :
			evaluatePrimalFromContactVelocity(problem, velocity, inverseRoot, contactVelocity, impulse, gradient, &weights, workspace.patchScratch, settings.parallelExecutor)))
		{
			return SolveStatus::eNUMERICAL_FAILURE;
		}
		result.evaluationMs += profileElapsed(settings.profile, evaluationStart);
		const double nextCost = primalCost(problem, velocity, contactVelocity, impulse);
		result.scaledImprovement = stopping.costScale * (cost - nextCost);
		cost = nextCost;
		if(result.scaledImprovement > 0.0 && result.scaledImprovement < stopping.tolerance)
		{
			result.stopReason = 2;
			break;
		}
	}
	// The workspace recomputes both vectors on its next solve, so exchange buffers.
	result.impulse.swap(impulse);
	result.primal.swap(velocity);
	result.gradientResidual = gradient.infinityNorm();
	result.scaledGradient = stopping.gradientNorm(problem, gradient);
	return result.stopReason == 0 ? SolveStatus::eITERATION_LIMIT : SolveStatus::eSUCCESS;
}

Workspace::Workspace() noexcept : m_data(new WorkspaceData) {}
Workspace::~Workspace() { delete m_data; }
Workspace::Workspace(Workspace&& other) noexcept : m_data(other.m_data) { other.m_data = new WorkspaceData; }
Workspace& Workspace::operator=(Workspace&& other) noexcept
{
	if(this != &other)
	{
		WorkspaceData* data = m_data;
		m_data = other.m_data;
		other.m_data = data;
	}
	return *this;
}

static SolveStatus::Enum solveAnvilWorkspace(const Problem& problem, const Settings& settings, Result& result, WorkspaceData& data, const Result* previous, bool continuation) noexcept
{
	const Clock::time_point start = profileStart(settings.timing);
	const bool reuse = continuation && data.continuationProblem == &problem && data.continuationGeneration == problem.preparationGeneration;
	data.continuationProblem = NULL;
	data.continuationGeneration = 0;
	result.status = solveAnvilInternal(problem, settings, result, data, previous, reuse);
	if(result.status == SolveStatus::eSUCCESS || result.status == SolveStatus::eITERATION_LIMIT)
	{
		data.continuationProblem = &problem;
		data.continuationGeneration = problem.preparationGeneration;
	}
	result.elapsedMs = profileElapsed(settings.timing, start);
	return result.status;
}

SolveStatus::Enum solveAnvil(const Problem& problem, const Settings& settings, Result& result, Workspace& workspace, const Result* previous) noexcept
{
	return solveAnvilWorkspace(problem, settings, result, *workspace.m_data, previous, false);
}

SolveStatus::Enum continueAnvil(const Problem& problem, const Settings& settings, Result& result, Workspace& workspace, const Result* previous) noexcept
{
	return solveAnvilWorkspace(problem, settings, result, *workspace.m_data, previous, true);
}

}
