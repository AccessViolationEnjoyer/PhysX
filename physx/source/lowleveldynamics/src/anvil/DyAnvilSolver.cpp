// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#include "DyAnvilSolver.h"
#include "DyAnvilConstraintPrep.h"
#include "DyAnvilContactPrep.h"
#include "DyDynamics.h"
#include "DyConstraint.h"
#include "CmFlushPool.h"
#include "CmTask.h"
#include "PxsRigidBody.h"
#include "PxsContactManager.h"
#include "PxSceneDesc.h"
#include "foundation/PxAtomic.h"
#include "foundation/PxIntrinsics.h"
#include "foundation/PxMutex.h"
#include "foundation/PxThread.h"
#include "foundation/PxUserAllocated.h"
#include "common/PxProfileZone.h"
#include "core/AnvilSolver.h"
#include <cmath>

namespace physx
{
namespace Dy
{
static volatile PxI32 gAnvilParallelTaskActive = 0;

class AnvilParallelExecutor;

class AnvilParallelTask : public Cm::Task
{
	AnvilParallelTask& operator=(const AnvilParallelTask&);
public:
	AnvilParallelTask(PxU64 contextId, AnvilParallelExecutor& executor, PxU32 worker, PxI32 generation) : Cm::Task(contextId), mExecutor(executor), mWorker(worker), mGeneration(generation)
	{
	}

	virtual void runInternal() PX_OVERRIDE;
	virtual const char* getName() const PX_OVERRIDE { return "PxsDynamics.anvilParallel"; }

private:
	AnvilParallelExecutor& mExecutor;
	PxU32 mWorker;
	PxI32 mGeneration;
};

class AnvilParallelExecutor : public anvil::ParallelExecutor
{
	AnvilParallelExecutor& operator=(const AnvilParallelExecutor&);
	enum { MAX_WORKERS = 8 };
	struct Counter
	{
		volatile PxI32 value;
		PxU8 padding[60];
	};
public:
	AnvilParallelExecutor(DynamicsContext& context, PxBaseTask* continuation, PxU32 workerCount) : mContext(context), mContinuation(continuation), mWorkerCount(PxMin(workerCount, PxU32(MAX_WORKERS))), mCount(0), mBatchSize(1), mFunction(NULL), mFunctionContext(NULL), mFinish(0), mAcquired(false), mStarted(false)
	{
		mGeneration.value = 0;
		mNext.value = 0;
		for(PxU32 worker = 0; worker < MAX_WORKERS; ++worker)
		{
			mProgress[worker].value = 0;
		}
	}

	virtual int workerCapacity() const PX_OVERRIDE
	{
		return int(mWorkerCount);
	}

	virtual int acquireWorkerCount() PX_OVERRIDE
	{
		if(mWorkerCount < 2)
		{
			return 1;
		}
		if(mAcquired)
		{
			return int(mWorkerCount);
		}
		// A waiting solver task consumes one dispatcher worker. Let only one island
		// recruit helpers so other large islands continue to make serial progress.
		if(PxAtomicCompareExchange(&gAnvilParallelTaskActive, 1, 0) != 0)
		{
			return 1;
		}
		mAcquired = true;
		return int(mWorkerCount);
	}

	virtual void parallelFor(int count, anvil::ParallelFunction function, void* context) PX_OVERRIDE
	{
		if(!mStarted)
		{
			startWorkers();
		}
		mCount = count;
		mBatchSize = PxMax(1, count / (int(mWorkerCount) * 8));
		mFunction = function;
		mFunctionContext = context;
		mNext.value = 0;
		PxMemoryBarrier();
		const PxI32 generation = PxAtomicIncrement(&mGeneration.value);
		runWork();
		for(PxU32 worker = 1; worker < mWorkerCount; ++worker)
		{
			waitFor(&mProgress[worker].value, generation);
		}
	}

	virtual void endParallelRegion() PX_OVERRIDE
	{
		if(!mStarted)
		{
			return;
		}
		mFinish = 1;
		PxMemoryBarrier();
		const PxI32 generation = PxAtomicIncrement(&mGeneration.value);
		for(PxU32 worker = 1; worker < mWorkerCount; ++worker)
		{
			waitFor(&mProgress[worker].value, generation);
		}
		mFinish = 0;
		mStarted = false;
	}

	void runWorker(PxU32 worker, PxI32 generation)
	{
		for(;;)
		{
			waitFor(&mGeneration.value, generation + 1);
			generation = mGeneration.value;
			PxMemoryBarrier();
			if(mFinish)
			{
				break;
			}
			runWork();
			PxMemoryBarrier();
			PxAtomicExchange(&mProgress[worker].value, generation);
		}
		PxMemoryBarrier();
		PxAtomicExchange(&mProgress[worker].value, generation);
	}

	void finish()
	{
		if(!mAcquired)
		{
			return;
		}
		endParallelRegion();
		PxAtomicExchange(&gAnvilParallelTaskActive, 0);
		mAcquired = false;
	}

private:
	static void waitFor(volatile PxI32* value, PxI32 target)
	{
		while(*value < target)
		{
			PxThread::yieldProcessor();
		}
	}

	void startWorkers()
	{
		mStarted = true;
		const PxI32 generation = mGeneration.value;
		for(PxU32 worker = 1; worker < mWorkerCount; ++worker)
		{
			void* memory = mContext.getTaskPool().allocate(sizeof(AnvilParallelTask));
			AnvilParallelTask* task = PX_PLACEMENT_NEW(memory, AnvilParallelTask)(mContext.getContextId(), *this, worker, generation);
			task->setContinuation(mContinuation);
			task->removeReference();
		}
	}

	void runWork()
	{
		PxI32 first = PxAtomicAdd(&mNext.value, mBatchSize) - mBatchSize;
		while(first < mCount)
		{
			const PxI32 last = PxMin(first + mBatchSize, mCount);
			for(PxI32 index = first; index < last; ++index)
			{
				mFunction(mFunctionContext, index);
			}
			first = PxAtomicAdd(&mNext.value, mBatchSize) - mBatchSize;
		}
	}

	DynamicsContext& mContext;
	PxBaseTask* mContinuation;
	PxU32 mWorkerCount;
	PxI32 mCount;
	PxI32 mBatchSize;
	anvil::ParallelFunction mFunction;
	void* mFunctionContext;
	Counter mGeneration;
	Counter mNext;
	Counter mProgress[MAX_WORKERS];
	volatile PxI32 mFinish;
	bool mAcquired;
	bool mStarted;
};

void AnvilParallelTask::runInternal()
{
	mExecutor.runWorker(mWorker, mGeneration);
}

struct AnvilBodySeed
{
	const PxsBodyCore* body;
	PxU64 update;
	PxTransform pose;
	PxVec3 linearCorrection;
	PxVec3 angularCorrection;
	PxVec3 inverseInertia;
	PxReal inverseMass;
	PxReal timestep;
	PxU8 lockFlags;

	AnvilBodySeed() : body(NULL), update(0), pose(PxIdentity), linearCorrection(0.0f), angularCorrection(0.0f), inverseInertia(0.0f), inverseMass(0.0f), timestep(0.0f), lockFlags(0) {}
};

// Contact rows of one chunk of a large island's descriptors, prepared in parallel and then
// merged in descriptor order, so the island's rows match serial preparation.
struct AnvilPrepareChunk : public PxUserAllocated
{
	anvil::Problem problem;
	AnvilContactRows contacts;
	PxContactBuffer buffer;
	// One friction state per descriptor, reserved before the parallel region; NULL if none.
	AnvilFrictionState* frictionStates;
	// The chunk's first records in the island's arrays.
	PxU32 firstContact;
	PxU32 firstBounds;
	PxU32 firstBlock;
	PxU32 firstPoint;
	PxU32 firstPair;
};

struct AnvilIslandWorkspace : public PxUserAllocated
{
	anvil::Problem problem;
	anvil::Result result;
	anvil::Result previous;
	anvil::Workspace numeric;
	AnvilJointRows joints;
	AnvilContactRows contacts;
	PxArray<PxU8> lockFlags;
	// A batch's constraint descriptors grouped by island, and each descriptor's island.
	PxArray<PxU32> descriptorOrder;
	PxArray<PxU32> descriptorIslands;
	// Retained staging for parallel row preparation.
	PxArray<AnvilPrepareChunk*> prepareChunks;

	AnvilIslandWorkspace() {}
	~AnvilIslandWorkspace()
	{
		for(PxU32 i = 0; i < prepareChunks.size(); ++i)
		{
			PX_DELETE(prepareChunks[i]);
		}
	}

private:
	AnvilIslandWorkspace(const AnvilIslandWorkspace&);
	AnvilIslandWorkspace& operator=(const AnvilIslandWorkspace&);
};

class AnvilSolver : public PxUserAllocated
{
	PX_NOCOPY(AnvilSolver)
public:
	explicit AnvilSolver(const PxSceneDesc& desc) : regularization(desc.anvilRegularization), jointRegularization(desc.anvilJointRegularization),
		surfaceRegularization(desc.anvilSurfaceRegularization), stiffeningDepth(desc.anvilStiffeningDepth), displacementTolerance(desc.anvilDisplacementTolerance), update(0), errorReported(0)
	{
		settings.iterations = int(desc.anvilMaxIterations);
		settings.tolerance = desc.anvilTolerance;
		// The SDK reports solve time through profile zones, not Result::elapsedMs.
		settings.timing = false;
		// Dispatcher workers and a synchronous caller bound concurrent island solves.
		const PxU32 workerCount = desc.cpuDispatcher ? desc.cpuDispatcher->getWorkerCount() : 0u;
		const PxU32 workspaceCount = PxMax(workerCount + 1, 1u);
		workspaces.reserve(workspaceCount);
		available.reserve(workspaceCount);
		for(PxU32 i = 0; i < workspaceCount; ++i)
		{
			void* memory = PxReflectionAllocator<AnvilIslandWorkspace>::allocate(sizeof(AnvilIslandWorkspace), PX_FL);
			AnvilIslandWorkspace* workspace = PX_PLACEMENT_NEW(memory, AnvilIslandWorkspace);
			workspaces.pushBack(workspace);
			available.pushBack(workspace);
		}
	}

	~AnvilSolver()
	{
		const PxU32 workspaceCount = workspaces.size();
		for(PxU32 i = 0; i < workspaceCount; ++i)
		{
			PX_DELETE(workspaces[i]);
		}
	}

	AnvilIslandWorkspace* acquire()
	{
		PxMutex::ScopedLock lock(mutex);
		PX_ASSERT(!available.empty());
		return available.popBack();
	}

	void release(AnvilIslandWorkspace* workspace)
	{
		PxMutex::ScopedLock lock(mutex);
		PX_ASSERT(available.size() < available.capacity());
		available.pushBack(workspace);
	}

	void report(const char* message)
	{
		if(PxAtomicCompareExchange(&errorReported, 1, 0) == 0)
		{
			PxGetFoundation().error(PxErrorCode::eINVALID_OPERATION, PX_FL, "%s Island integration was skipped.", message);
		}
	}

	anvil::Settings settings;
	PxReal regularization;
	PxReal jointRegularization;
	PxReal surfaceRegularization;
	PxReal stiffeningDepth;
	PxReal displacementTolerance;
	PxU64 update;
	// Settings every island of the current step shares; islands add their body arrays.
	AnvilJointSettings jointSettings;
	AnvilContactSettings contactSettings;
	PxArray<AnvilBodySeed> seeds;
	PxMutex mutex;
	PxArray<AnvilIslandWorkspace*> workspaces;
	PxArray<AnvilIslandWorkspace*> available;
	volatile PxI32 errorReported;
};

AnvilSolver* createAnvilSolver(const PxSceneDesc& desc)
{
	void* memory = PxReflectionAllocator<AnvilSolver>::allocate(sizeof(AnvilSolver), PX_FL);
	return PX_PLACEMENT_NEW(memory, AnvilSolver)(desc);
}

void destroyAnvilSolver(AnvilSolver* solver)
{
	PX_DELETE(solver);
}

void beginAnvilUpdate(AnvilSolver& solver, PxU32 nodeCount, const DynamicsContext& context, PxReal timestep)
{
	++solver.update;
	if(nodeCount > solver.seeds.size())
	{
		solver.seeds.resize(nodeCount);
	}
	AnvilJointSettings& jointSettings = solver.jointSettings;
	jointSettings.timestep = timestep;
	jointSettings.regularization = solver.jointRegularization;
	AnvilContactSettings& contactSettings = solver.contactSettings;
	contactSettings.timestep = timestep;
	contactSettings.regularization = solver.regularization;
	contactSettings.bounceThreshold = context.getBounceThreshold();
	contactSettings.ccdMaxSeparation = context.getCCDSeparationThreshold();
	contactSettings.impedance = 1.0 / (1.0 + double(contactSettings.regularization));
	const double timeConstant = std::max(0.02, 2.0 * double(timestep));
	contactSettings.damping = 2.0 / (contactSettings.impedance * timeConstant);
	contactSettings.stiffness = 1.0 / (contactSettings.impedance * contactSettings.impedance * timeConstant * timeConstant);
	contactSettings.surfaceRegularization = solver.surfaceRegularization;
	contactSettings.regularizationLogRange = std::log2(double(contactSettings.regularization) / contactSettings.surfaceRegularization);
	contactSettings.stiffeningDepth = solver.stiffeningDepth;
	contactSettings.slipLimit = 1.0e-3f * context.getLengthScale();
}

static bool matchesAnvilSeed(const AnvilBodySeed& seed, const PxsBodyCore& body, PxU64 update)
{
	return seed.update + 1 == update && seed.body == &body && seed.pose.p == body.body2World.p && seed.pose.q == body.body2World.q && seed.inverseMass == body.inverseMass && seed.inverseInertia == body.inverseInertia && seed.lockFlags == PxU8(body.lockFlags);
}

static void applyAnvilLocks(PxSolverBodyData& data, PxU8 lockFlags)
{
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		if(lockFlags & (1 << axis))
		{
			data.linearVelocity[axis] = 0.0f;
		}
		if(lockFlags & (1 << (axis + 3)))
		{
			data.angularVelocity[axis] = 0.0f;
		}
	}
	if(!(lockFlags & 0x38))
	{
		return;
	}

	// Restrict the physical inverse inertia to the unlocked world axes. A symmetric
	// square root keeps the Anvil response symmetric and makes integration's lock
	// clamp redundant; merely clamping the solved velocity would violate the rows.
	anvil::Mat3 inverseRoot;
	for(PxU32 column = 0; column < 3; ++column)
	{
		for(PxU32 row = 0; row < 3; ++row)
		{
			inverseRoot(row, column) = data.sqrtInvInertia[column][row];
		}
	}
	anvil::Mat3 response = inverseRoot * inverseRoot.transpose();
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		if(lockFlags & (1 << (axis + 3)))
		{
			response.row(axis).setZero();
			response.col(axis).setZero();
		}
	}
	anvil::Vec3 eigenvalues;
	anvil::Mat3 eigenvectors;
	anvil::symmetricEigen(response, eigenvalues, eigenvectors);
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		eigenvalues[axis] = std::sqrt(std::max(0.0, eigenvalues[axis]));
	}
	inverseRoot = eigenvectors * anvil::diagonalMatrix(eigenvalues) * eigenvectors.transpose();
	for(PxU32 column = 0; column < 3; ++column)
	{
		for(PxU32 row = 0; row < 3; ++row)
		{
			data.sqrtInvInertia[column][row] = (lockFlags & ((1 << (row + 3)) | (1 << (column + 3)))) ? 0.0f : PxReal(inverseRoot(row, column));
		}
	}
}

static void prepareAnvilBodies(AnvilSolver& solver, AnvilIslandWorkspace& workspace, PxsBodyCore* const* bodyCores, const PxU32* nodeIndices, PxSolverBodyData* bodyData, PxU32 bodyCount, PxReal timestep)
{
	anvil::Problem& problem = workspace.problem;
	problem.timestep = timestep;
	problem.inverseMass.resize(bodyCount);
	problem.massDiagonal.resize(6 * bodyCount);
	problem.freeBodyVelocity.resize(6 * bodyCount);
	workspace.previous.primal.setZero(6 * bodyCount);
	workspace.lockFlags.resize(bodyCount);

	for(PxU32 i = 0; i < bodyCount; ++i)
	{
		const PxsBodyCore& body = *bodyCores[i];
		PxSolverBodyData& data = bodyData[i];
		const PxU8 bodyLockFlags = PxU8(body.lockFlags);
		workspace.lockFlags[i] = bodyLockFlags;
		if(bodyLockFlags)
		{
			applyAnvilLocks(data, bodyLockFlags);
		}
		problem.inverseMass[i] = data.invMass;
		const PxU32 bodyOffset = 6 * i;
		PX_ASSERT(data.invMass >= 0.0f);
		PX_ASSERT(body.inverseInertia.x >= 0.0f && body.inverseInertia.y >= 0.0f && body.inverseInertia.z >= 0.0f);
		const double mass = data.invMass != 0.0f ? 1.0 / data.invMass : 1.0;
		const PxVec3 inertia(body.inverseInertia.x != 0.0f ? 1.0f / body.inverseInertia.x : 1.0f, body.inverseInertia.y != 0.0f ? 1.0f / body.inverseInertia.y : 1.0f, body.inverseInertia.z != 0.0f ? 1.0f / body.inverseInertia.z : 1.0f);
		const PxMat33 rotation(data.body2World.q);
		for(PxU32 axis = 0; axis < 3; ++axis)
		{
			problem.massDiagonal[bodyOffset + axis] = mass;
			// This diagonal is only the stopping metric. Row response uses the full
			// world inertia, including off-diagonal terms, throughout the solve.
			problem.massDiagonal[bodyOffset + axis + 3] =
				double(rotation.column0[axis]) * rotation.column0[axis] * inertia.x +
				double(rotation.column1[axis]) * rotation.column1[axis] * inertia.y +
				double(rotation.column2[axis]) * rotation.column2[axis] * inertia.z;
			problem.freeBodyVelocity[bodyOffset + axis] = data.linearVelocity[axis];
			problem.freeBodyVelocity[bodyOffset + axis + 3] = data.angularVelocity[axis];
		}
		const AnvilBodySeed& seed = solver.seeds[nodeIndices[i]];
		if(matchesAnvilSeed(seed, body, solver.update) && seed.timestep == timestep)
		{
			const double rootMass = std::sqrt(mass);
			// Transform a physical world-space correction into the current inertia
			// basis. Reusing last frame's whitened angular vector after rotation is wrong.
			const PxVec3 localAngular = data.body2World.q.rotateInv(seed.angularCorrection);
			PxVec3 localScaled(0.0f);
			for(PxU32 axis = 0; axis < 3; ++axis)
			{
				if(body.inverseInertia[axis] != 0.0f)
				{
					localScaled[axis] = PxReal(localAngular[axis] / std::sqrt(double(body.inverseInertia[axis])));
				}
			}
			const PxVec3 angular = data.body2World.q.rotate(localScaled);
			for(PxU32 axis = 0; axis < 3; ++axis)
			{
				if(!(bodyLockFlags & (1 << axis)) && data.invMass > 0.0f)
				{
					workspace.previous.primal[bodyOffset + axis] = seed.linearCorrection[axis] * rootMass;
				}
				if(!(bodyLockFlags & (1 << (axis + 3))))
				{
					workspace.previous.primal[bodyOffset + axis + 3] = angular[axis];
				}
			}
		}
	}
}

static PxI32 anvilBodyIndex(PxU32 dataIndex, PxU32 firstBodyIndex, PxU32 bodyCount)
{
	return dataIndex > firstBodyIndex && dataIndex <= firstBodyIndex + bodyCount ? PxI32(dataIndex - firstBodyIndex - 1) : -1;
}

// Islands with this many bodies prepare contact rows in parallel, in chunks of this many
// descriptors. A chunk's friction states fit one friction stream block.
static const PxU32 ANVIL_PARALLEL_PREPARE_BODIES = 500;
static const PxU32 ANVIL_PREPARE_CHUNK_DESCRIPTORS = 256;
PX_COMPILE_TIME_ASSERT(ANVIL_PREPARE_CHUNK_DESCRIPTORS * sizeof(AnvilFrictionState) <= PxcNpMemBlock::SIZE);

struct AnvilParallelPrepare
{
	AnvilIslandWorkspace* workspace;
	DynamicsContext* context;
	const PxSolverConstraintDesc* descriptorArray;
	const PxU32* descriptors;
	PxU32 descriptorCount;
	PxSolverBodyData* bodyData;
	PxU32 firstBodyIndex;
	PxU32 bodyCount;
	const AnvilContactSettings* settings;
};

static void prepareAnvilChunk(void* context, int index)
{
	const AnvilParallelPrepare& prepare = *static_cast<const AnvilParallelPrepare*>(context);
	AnvilPrepareChunk& chunk = *prepare.workspace->prepareChunks[PxU32(index)];
	chunk.problem.clearContacts();
	chunk.contacts.clear();
	const PxU32 first = PxU32(index) * ANVIL_PREPARE_CHUNK_DESCRIPTORS;
	const PxU32 end = PxMin(first + ANVIL_PREPARE_CHUNK_DESCRIPTORS, prepare.descriptorCount);
	for(PxU32 k = first; k < end; ++k)
	{
		const PxSolverConstraintDesc& desc = prepare.descriptorArray[prepare.descriptors ? prepare.descriptors[k] : k];
		PX_ASSERT(desc.constraintType == DY_SC_TYPE_RB_CONTACT);
		PxsContactManager& manager = *reinterpret_cast<PxsContactManager*>(desc.constraint);
		PxsContactManagerOutput& output = prepare.context->mOutputIterator.getContactManagerOutput(manager.getWorkUnit().mNpIndex);
		prepareAnvilContacts(manager, output, prepare.bodyData[desc.bodyADataIndex], prepare.bodyData[desc.bodyBDataIndex],
			anvilBodyIndex(desc.bodyADataIndex, prepare.firstBodyIndex, prepare.bodyCount), anvilBodyIndex(desc.bodyBDataIndex, prepare.firstBodyIndex, prepare.bodyCount),
			*prepare.settings, chunk.buffer, NULL, chunk.frictionStates ? chunk.frictionStates + (k - first) : NULL, chunk.problem, chunk.contacts);
	}
}

// Copy a chunk's records to its offsets in the island, rebasing their indices.
static void mergeAnvilChunk(void* context, int index)
{
	const AnvilParallelPrepare& prepare = *static_cast<const AnvilParallelPrepare*>(context);
	const AnvilPrepareChunk& chunk = *prepare.workspace->prepareChunks[PxU32(index)];
	anvil::Problem& problem = prepare.workspace->problem;
	AnvilContactRows& rows = prepare.workspace->contacts;
	const PxU32 contactCount = PxU32(chunk.problem.contacts.size());
	for(PxU32 i = 0; i < contactCount; ++i)
	{
		anvil::CompactContact& contact = problem.contacts[chunk.firstContact + i];
		contact = chunk.problem.contacts[i];
		if(contact.hasScalarBounds())
		{
			contact.block -= int(chunk.firstBounds);
		}
		else if(contact.block > 0)
		{
			contact.block += int(chunk.firstBlock);
		}
		else if(contact.block < 0)
		{
			contact.block -= int(chunk.firstBlock);
		}
	}
	for(PxU32 i = 0; i < PxU32(chunk.problem.scalarBounds.size()); ++i)
	{
		problem.scalarBounds[chunk.firstBounds + i] = chunk.problem.scalarBounds[i];
	}
	for(PxU32 i = 0; i < PxU32(chunk.problem.contactBlocks.size()); ++i)
	{
		problem.contactBlocks[chunk.firstBlock + i] = chunk.problem.contactBlocks[i];
	}
	for(PxU32 i = 0; i < chunk.contacts.points.size(); ++i)
	{
		AnvilContactPoint& point = rows.points[chunk.firstPoint + i];
		point = chunk.contacts.points[i];
		point.firstContact += chunk.firstContact;
	}
	for(PxU32 i = 0; i < chunk.contacts.pairs.size(); ++i)
	{
		AnvilContactPair& pair = rows.pairs[chunk.firstPair + i];
		pair = chunk.contacts.pairs[i];
		pair.firstPoint += chunk.firstPoint;
	}
}

// Prepares a contact-only island's rows in parallel; false leaves them to the serial path.
static bool prepareAnvilRowsParallel(AnvilIslandWorkspace& workspace, DynamicsContext& context, ThreadContext& threadContext, PxSolverBodyData* bodyData, PxU32 firstBodyIndex, PxU32 bodyCount,
	const PxU32* descriptors, PxU32 descriptorCount, const AnvilContactSettings& settings, anvil::ParallelExecutor* parallelExecutor)
{
	if(!parallelExecutor || bodyCount < ANVIL_PARALLEL_PREPARE_BODIES || descriptorCount <= ANVIL_PREPARE_CHUNK_DESCRIPTORS)
	{
		return false;
	}
	for(PxU32 k = 0; k < descriptorCount; ++k)
	{
		if(threadContext.contactConstraintDescArray[descriptors ? descriptors[k] : k].constraintType != DY_SC_TYPE_RB_CONTACT)
		{
			return false;
		}
	}
	if(parallelExecutor->acquireWorkerCount() < 2)
	{
		return false;
	}
	const PxU32 chunkCount = (descriptorCount + ANVIL_PREPARE_CHUNK_DESCRIPTORS - 1) / ANVIL_PREPARE_CHUNK_DESCRIPTORS;
	while(workspace.prepareChunks.size() < chunkCount)
	{
		workspace.prepareChunks.pushBack(PX_NEW(AnvilPrepareChunk));
	}
	// The friction stream is not thread safe, so each chunk reserves its states here.
	for(PxU32 i = 0; i < chunkCount; ++i)
	{
		const PxU32 count = PxMin(ANVIL_PREPARE_CHUNK_DESCRIPTORS, descriptorCount - i * ANVIL_PREPARE_CHUNK_DESCRIPTORS);
		AnvilFrictionState* states = threadContext.mFrictionPatchStreamPair.reserve<AnvilFrictionState>(count * PxU32(sizeof(AnvilFrictionState)));
		workspace.prepareChunks[i]->frictionStates = states == reinterpret_cast<AnvilFrictionState*>(-1) ? NULL : states;
	}
	AnvilParallelPrepare prepare = { &workspace, &context, threadContext.contactConstraintDescArray, descriptors, descriptorCount, bodyData, firstBodyIndex, bodyCount, &settings };
	parallelExecutor->parallelFor(int(chunkCount), prepareAnvilChunk, &prepare);
	PxU32 contacts = 0, bounds = 0, blocks = 0, points = 0, pairs = 0;
	double reportedWork = 0.0;
	PxU32 reportedRows = 0;
	for(PxU32 i = 0; i < chunkCount; ++i)
	{
		AnvilPrepareChunk& chunk = *workspace.prepareChunks[i];
		chunk.firstContact = contacts;
		chunk.firstBounds = bounds;
		chunk.firstBlock = blocks;
		chunk.firstPoint = points;
		chunk.firstPair = pairs;
		contacts += PxU32(chunk.problem.contacts.size());
		bounds += PxU32(chunk.problem.scalarBounds.size());
		blocks += PxU32(chunk.problem.contactBlocks.size());
		points += chunk.contacts.points.size();
		pairs += chunk.contacts.pairs.size();
		// Integer-valued work sums exactly in any order.
		reportedWork += chunk.problem.reportedRebuildWork;
		reportedRows += chunk.problem.reportedRebuildRows;
	}
	anvil::Problem& problem = workspace.problem;
	problem.contacts.resize(contacts);
	problem.scalarBounds.resize(bounds);
	problem.contactBlocks.resize(blocks);
	problem.reportedRebuildWork = reportedWork;
	problem.reportedRebuildRows = reportedRows;
	workspace.contacts.points.resize(points);
	workspace.contacts.pairs.resize(pairs);
	parallelExecutor->parallelFor(int(chunkCount), mergeAnvilChunk, &prepare);
	return true;
}

static void prepareAnvilRows(AnvilSolver& solver, AnvilIslandWorkspace& workspace, DynamicsContext& context, ThreadContext& threadContext, const Cm::SpatialVector* motionVelocities, PxSolverBodyData* bodyData, PxU32 firstBodyIndex, PxU32 bodyCount, const PxU32* descriptors, PxU32 descriptorCount, anvil::ParallelExecutor* parallelExecutor)
{
	anvil::Problem& problem = workspace.problem;
	problem.clearContacts();
	problem.columnCursors.clear();
	workspace.joints.clear();
	workspace.contacts.clear();
	AnvilJointSettings jointSettings = solver.jointSettings;
	jointSettings.bodyLockFlags = workspace.lockFlags.begin();
	jointSettings.initialVelocities = motionVelocities;
	AnvilContactSettings contactSettings = solver.contactSettings;
	contactSettings.bodyLockFlags = workspace.lockFlags.begin();
	contactSettings.initialVelocities = motionVelocities;
	// Without a grouped list, the island owns all of its thread context's descriptors.
	const PxU32 contactDescriptionCount = descriptors ? descriptorCount : threadContext.contactDescArraySize;
	if(prepareAnvilRowsParallel(workspace, context, threadContext, bodyData, firstBodyIndex, bodyCount, descriptors, contactDescriptionCount, contactSettings, parallelExecutor))
	{
		return;
	}
	for(PxU32 k = 0; k < contactDescriptionCount; ++k)
	{
		const PxSolverConstraintDesc& desc = threadContext.contactConstraintDescArray[descriptors ? descriptors[k] : k];
		const PxSolverBodyData& body0 = bodyData[desc.bodyADataIndex];
		const PxSolverBodyData& body1 = bodyData[desc.bodyBDataIndex];
		const PxI32 index0 = anvilBodyIndex(desc.bodyADataIndex, firstBodyIndex, bodyCount);
		const PxI32 index1 = anvilBodyIndex(desc.bodyBDataIndex, firstBodyIndex, bodyCount);
		if(desc.constraintType == DY_SC_TYPE_RB_1D)
		{
			const Constraint& constraint = *reinterpret_cast<const Constraint*>(desc.constraint);
			prepareAnvilJoint(constraint, body0, body1, index0, index1, &context.getConstraintWriteBackPool()[constraint.index], jointSettings, problem, workspace.joints);
		}
		else
		{
			PX_ASSERT(desc.constraintType == DY_SC_TYPE_RB_CONTACT);
			PxsContactManager& manager = *reinterpret_cast<PxsContactManager*>(desc.constraint);
			PxsContactManagerOutput& output = context.mOutputIterator.getContactManagerOutput(manager.getWorkUnit().mNpIndex);
			prepareAnvilContacts(manager, output, body0, body1, index0, index1, contactSettings, threadContext.mContactBuffer, &threadContext.mFrictionPatchStreamPair, NULL, problem, workspace.contacts);
		}
	}
}

static void storeAnvilCorrection(const anvil::Result& result, const PxSolverBodyData* data, Cm::SpatialVector* motion, PxSolverBody* bodies, PxU32 bodyCount)
{
	for(PxU32 i = 0; i < bodyCount; ++i)
	{
		PX_ASSERT(data[i].invMass >= 0.0f);
		const double rootInverseMass = std::sqrt(double(data[i].invMass));
		const PxU32 bodyOffset = 6 * i;
		PxVec3 linear, angular;
		for(PxU32 axis = 0; axis < 3; ++axis)
		{
			linear[axis] = PxReal(result.primal[bodyOffset + axis] * rootInverseMass);
			angular[axis] = PxReal(result.primal[bodyOffset + axis + 3]);
		}
		motion[i].linear = bodies[i].linearVelocity = linear;
		motion[i].angular = bodies[i].angularState = angular;
	}
}

static bool solveAnvilSystem(const anvil::Settings& settings, AnvilIslandWorkspace& workspace, const anvil::Result* previous)
{
	const anvil::SolveStatus::Enum status = anvil::solveAnvil(workspace.problem, settings, workspace.result, workspace.numeric, previous);
	PX_ASSERT(status != anvil::SolveStatus::eINVALID_INPUT);
	return status == anvil::SolveStatus::eSUCCESS || status == anvil::SolveStatus::eITERATION_LIMIT;
}

static void profileAnvilSolve(const AnvilIslandWorkspace& workspace, PxU64 contextId)
{
	PX_UNUSED(workspace);
	PX_UNUSED(contextId);
	PX_PROFILE_VALUE(workspace.result.iterations, "Dynamics.anvilIterations", contextId);
	PX_PROFILE_VALUE(workspace.result.factorizations, "Dynamics.anvilFactorizations", contextId);
	PX_PROFILE_VALUE(workspace.result.rankUpdates, "Dynamics.anvilRankUpdates", contextId);
	PX_PROFILE_VALUE(workspace.result.lineSearchEvaluations, "Dynamics.anvilLineEvaluations", contextId);
	PX_PROFILE_VALUE(workspace.problem.bodyCount(), "Dynamics.anvilBodies", contextId);
	PX_PROFILE_VALUE(workspace.problem.rowCount(), "Dynamics.anvilRows", contextId);
	PX_PROFILE_VALUE(int(workspace.problem.patches.size()), "Dynamics.anvilPatches", contextId);
	PX_PROFILE_VALUE(int(workspace.result.status), "Dynamics.anvilStatus", contextId);
	PX_PROFILE_VALUE(PxReal(workspace.result.scaledGradient), "Dynamics.anvilScaledGradient", contextId);
}

static bool solveAnvilRows(AnvilSolver& solver, AnvilIslandWorkspace& workspace, DynamicsContext& context, ThreadContext& threadContext, PxSolverBody* bodies, PxSolverBodyData* allBodyData, PxU32 firstBodyIndex, PxU32 threadBodyOffset, PxU32 bodyCount, const PxU32* descriptors, PxU32 descriptorCount, anvil::ParallelExecutor* parallelExecutor)
{
	PxsBodyCore* const* bodyCores = threadContext.mBodyCoreArray + threadBodyOffset;
	const PxU32* nodeIndices = threadContext.mNodeIndexArray + threadBodyOffset;
	Cm::SpatialVector* motionVelocities = threadContext.motionVelocityArray + threadBodyOffset;
	PX_PROFILE_ZONE("Dynamics.anvilIsland", context.getContextId());
	PxSolverBodyData* bodyData = allBodyData + firstBodyIndex + 1;
	const PxReal timestep = context.getDt();
	{
		PX_PROFILE_ZONE("Dynamics.anvilPrepare", context.getContextId());
		prepareAnvilBodies(solver, workspace, bodyCores, nodeIndices, bodyData, bodyCount, timestep);
		prepareAnvilRows(solver, workspace, context, threadContext, motionVelocities, allBodyData, firstBodyIndex, bodyCount, descriptors, descriptorCount, parallelExecutor);
		anvil::prepareCompactProblemFromColumnCounts(workspace.problem);
	}
	{
		PX_PROFILE_ZONE("Dynamics.anvilSolve", context.getContextId());
		anvil::Settings settings = solver.settings;
		settings.parallelExecutor = parallelExecutor;
		// Stop once a step would move no body by the displacement tolerance in this timestep.
		settings.velocityTolerance = double(solver.displacementTolerance) / timestep;
		settings.angularVelocityTolerance = settings.velocityTolerance / context.getLengthScale();
		if(!solveAnvilSystem(settings, workspace, &workspace.previous))
		{
			solver.report("Anvil solve failed.");
			return false;
		}
		profileAnvilSolve(workspace, context.getContextId());
		// A deferred point joins when the solve closes its gap, as an impact turning a body into it
		// does, and the problem is solved again from the solution so far, keeping the bodies'
		// ordering. Each pass adds the points the last solution closes. The rows are added with the
		// settings they were prepared with (prepareAnvilRows).
		AnvilContactSettings contactSettings = solver.contactSettings;
		contactSettings.bodyLockFlags = workspace.lockFlags.begin();
		contactSettings.initialVelocities = motionVelocities;
		while(addClosedAnvilContacts(workspace.contacts, workspace.problem, workspace.result, contactSettings, bodyData, bodyCount, threadContext.mContactBuffer, context))
		{
			PX_PROFILE_ZONE("Dynamics.anvilClosedContacts", context.getContextId());
			extern int gAnvilDebugStep, gAnvilDebugFrom, gAnvilDebugTo; // EXPERIMENT (temporary)
			if(gAnvilDebugStep >= gAnvilDebugFrom && gAnvilDebugStep <= gAnvilDebugTo) { printf("  [%d] re-solve: late points %u\n", gAnvilDebugStep, workspace.contacts.latePairs.size()); } // EXPERIMENT (temporary)
			anvil::prepareCompactProblemFromColumnCounts(workspace.problem);
			settings.keepOrdering = true;
			if(!solveAnvilSystem(settings, workspace, &workspace.result))
			{
				solver.report("Anvil solve failed.");
				return false;
			}
			profileAnvilSolve(workspace, context.getContextId());
		}
	}
	threadContext.mAxisConstraintCount += PxU32(workspace.problem.rowCount());
	storeAnvilCorrection(workspace.result, bodyData, motionVelocities, bodies, bodyCount);
	writebackAnvilJoints(workspace.joints, workspace.problem, workspace.result);
	writebackAnvilContacts(workspace.contacts, workspace.problem, workspace.result, bodies, bodyData, context);
	for(PxU32 i = 0; i < bodyCount; ++i)
	{
		AnvilBodySeed& seed = solver.seeds[nodeIndices[i]];
		const PxsBodyCore& body = *bodyCores[i];
		seed.body = &body;
		seed.update = solver.update;
		seed.linearCorrection = bodies[i].linearVelocity;
		seed.angularCorrection = bodyData[i].sqrtInvInertia * bodies[i].angularState;
		seed.inverseMass = body.inverseMass;
		seed.inverseInertia = body.inverseInertia;
		seed.lockFlags = PxU8(body.lockFlags);
		seed.timestep = timestep;
	}
	return true;
}

AnvilIslandWorkspace* acquireAnvilWorkspace(AnvilSolver& solver)
{
	return solver.acquire();
}

void releaseAnvilWorkspace(AnvilSolver& solver, AnvilIslandWorkspace* workspace)
{
	solver.release(workspace);
}

const PxU32* groupAnvilDescriptors(AnvilIslandWorkspace& workspace, const ThreadContext& threadContext, PxU32 firstBodyIndex, const PxU32* islandFirstBodies, PxU32 islandCount, PxU32* islandStarts)
{
	// Counting sort by island, keyed on each descriptor's first dynamic batch body.
	const PxU32 descriptorCount = threadContext.contactDescArraySize;
	const PxU32 bodyCount = islandFirstBodies[islandCount];
	// A batch without descriptors still returns storage, not NULL, which means ungrouped.
	workspace.descriptorOrder.resize(PxMax(descriptorCount, 1u));
	workspace.descriptorIslands.resize(descriptorCount);
	for(PxU32 island = 0; island <= islandCount; ++island)
	{
		islandStarts[island] = 0;
	}
	for(PxU32 i = 0; i < descriptorCount; ++i)
	{
		const PxSolverConstraintDesc& desc = threadContext.contactConstraintDescArray[i];
		const PxI32 index0 = anvilBodyIndex(desc.bodyADataIndex, firstBodyIndex, bodyCount);
		const PxU32 body = PxU32(index0 >= 0 ? index0 : anvilBodyIndex(desc.bodyBDataIndex, firstBodyIndex, bodyCount));
		PX_ASSERT(body < bodyCount);
		PxU32 island = 0;
		while(island + 1 < islandCount && islandFirstBodies[island + 1] <= body)
		{
			++island;
		}
		workspace.descriptorIslands[i] = island;
		++islandStarts[island + 1];
	}
	for(PxU32 island = 0; island < islandCount; ++island)
	{
		islandStarts[island + 1] += islandStarts[island];
	}
	// Place each island's descriptors from its start, keeping their original order.
	for(PxU32 i = 0; i < descriptorCount; ++i)
	{
		workspace.descriptorOrder[islandStarts[workspace.descriptorIslands[i]]++] = i;
	}
	for(PxU32 island = islandCount; island > 0; --island)
	{
		islandStarts[island] = islandStarts[island - 1];
	}
	islandStarts[0] = 0;
	return workspace.descriptorOrder.begin();
}

bool solveAnvilIsland(AnvilSolver& solver, AnvilIslandWorkspace& workspace, DynamicsContext& context, ThreadContext& threadContext, PxSolverBody* bodies, PxSolverBodyData* bodyData, PxU32 firstBodyIndex, PxU32 threadBodyOffset, PxU32 bodyCount, const PxU32* descriptors, PxU32 descriptorCount, PxBaseTask* continuation, PxU32 workerCount)
{
	AnvilParallelExecutor parallelExecutor(context, continuation, workerCount);
	const bool success = solveAnvilRows(solver, workspace, context, threadContext, bodies, bodyData, firstBodyIndex, threadBodyOffset, bodyCount, descriptors, descriptorCount, &parallelExecutor);
	parallelExecutor.finish();
	return success;
}

void saveAnvilPoses(AnvilSolver& solver, PxsBodyCore* const* bodies, const PxU32* nodeIndices, PxU32 bodyCount)
{
	for(PxU32 i = 0; i < bodyCount; ++i)
	{
		solver.seeds[nodeIndices[i]].pose = bodies[i]->body2World;
	}
}
}
}
