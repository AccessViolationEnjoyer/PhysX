// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//	notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//	notice, this list of conditions and the following disclaimer in the
//	documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//	contributors may be used to endorse or promote products derived
//	from this software without specific prior written permission.
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

#include "DyNewtonSolver.h"
#include "DyNewtonConstraintPrep.h"
#include "DyNewtonContactPrep.h"
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
#include "core/NewtonSolver.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <new>

namespace physx
{
namespace Dy
{
static volatile PxI32 gNewtonParallelTaskActive = 0;

class NewtonParallelExecutor;

class NewtonParallelTask : public Cm::Task
{
	NewtonParallelTask& operator=(const NewtonParallelTask&);
public:
	NewtonParallelTask(PxU64 contextId, NewtonParallelExecutor& executor, PxU32 worker, PxI32 generation) :
		Cm::Task(contextId), mExecutor(executor), mWorker(worker), mGeneration(generation)
	{
	}

	virtual void runInternal() PX_OVERRIDE;
	virtual const char* getName() const PX_OVERRIDE { return "PxsDynamics.newtonParallel"; }

private:
	NewtonParallelExecutor& mExecutor;
	PxU32 mWorker;
	PxI32 mGeneration;
};

class NewtonParallelExecutor : public newton::ParallelExecutor
{
	NewtonParallelExecutor& operator=(const NewtonParallelExecutor&);
	enum { MAX_WORKERS = 8 };
	struct Counter
	{
		volatile PxI32 value;
		PxU8 padding[60];
	};
public:
	NewtonParallelExecutor(DynamicsContext& context, PxBaseTask* continuation, PxU32 workerCount) :
		mContext(context), mContinuation(continuation),
		mWorkerCount(PxMin(workerCount, PxU32(MAX_WORKERS))), mCount(0), mBatchSize(1), mFunction(NULL), mFunctionContext(NULL),
		mFinish(0), mAcquired(false), mStarted(false)
	{
		mGeneration.value = 0;
		mNext.value = 0;
		for(PxU32 worker = 0; worker < MAX_WORKERS; ++worker)
			mProgress[worker].value = 0;
	}

	virtual int workerCapacity() const PX_OVERRIDE
	{
		return int(mWorkerCount);
	}

	virtual int acquireWorkerCount() PX_OVERRIDE
	{
		if(mWorkerCount < 2)
			return 1;
		if(mAcquired)
			return int(mWorkerCount);
		// A waiting solver task consumes one dispatcher worker. Let only one island
		// recruit helpers so other large islands continue to make serial progress.
		if(PxAtomicCompareExchange(&gNewtonParallelTaskActive, 1, 0) != 0)
			return 1;
		mAcquired = true;
		return int(mWorkerCount);
	}

	virtual void parallelFor(int count, newton::ParallelFunction function, void* context) PX_OVERRIDE
	{
		if(!mStarted)
			startWorkers();
		mCount = count;
		mBatchSize = PxMax(1, count / (int(mWorkerCount) * 8));
		mFunction = function;
		mFunctionContext = context;
		mNext.value = 0;
		PxMemoryBarrier();
		const PxI32 generation = PxAtomicIncrement(&mGeneration.value);
		runWork();
		for(PxU32 worker = 1; worker < mWorkerCount; ++worker)
			waitFor(&mProgress[worker].value, generation);
	}

	virtual void endParallelRegion() PX_OVERRIDE
	{
		if(!mStarted)
			return;
		mFinish = 1;
		PxMemoryBarrier();
		const PxI32 generation = PxAtomicIncrement(&mGeneration.value);
		for(PxU32 worker = 1; worker < mWorkerCount; ++worker)
			waitFor(&mProgress[worker].value, generation);
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
				break;
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
			return;
		endParallelRegion();
		PxAtomicExchange(&gNewtonParallelTaskActive, 0);
		mAcquired = false;
	}

private:
	static void waitFor(volatile PxI32* value, PxI32 target)
	{
		while(*value < target)
			PxThread::yieldProcessor();
	}

	void startWorkers()
	{
		mStarted = true;
		const PxI32 generation = mGeneration.value;
		for(PxU32 worker = 1; worker < mWorkerCount; ++worker)
		{
			void* memory = mContext.getTaskPool().allocate(sizeof(NewtonParallelTask));
			NewtonParallelTask* task = PX_PLACEMENT_NEW(memory, NewtonParallelTask)(mContext.getContextId(), *this, worker, generation);
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
				mFunction(mFunctionContext, index);
			first = PxAtomicAdd(&mNext.value, mBatchSize) - mBatchSize;
		}
	}

	DynamicsContext& mContext;
	PxBaseTask* mContinuation;
	PxU32 mWorkerCount;
	PxI32 mCount;
	PxI32 mBatchSize;
	newton::ParallelFunction mFunction;
	void* mFunctionContext;
	Counter mGeneration;
	Counter mNext;
	Counter mProgress[MAX_WORKERS];
	volatile PxI32 mFinish;
	bool mAcquired;
	bool mStarted;
};

void NewtonParallelTask::runInternal()
{
	mExecutor.runWorker(mWorker, mGeneration);
}

struct NewtonBodySeed
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

	NewtonBodySeed() : body(NULL), update(0), pose(PxIdentity), linearCorrection(0.0f),
		angularCorrection(0.0f), inverseInertia(0.0f), inverseMass(0.0f), timestep(0.0f), lockFlags(0) {}
};

struct NewtonIslandWorkspace : public PxUserAllocated
{
	newton::Problem problem;
	newton::Result result;
	newton::Result previous;
	newton::Workspace numeric;
	NewtonJointRows joints;
	NewtonContactRows contacts;
	PxArray<PxU8> lockFlags;
};

class NewtonSolver : public PxUserAllocated
{
	PX_NOCOPY(NewtonSolver)
public:
	explicit NewtonSolver(const PxSceneDesc& desc) : regularization(desc.newtonRegularization), update(0), errorReported(0)
	{
		settings.iterations = int(desc.newtonMaxIterations);
		settings.tolerance = desc.newtonTolerance;
	}

	~NewtonSolver()
	{
		for(PxU32 i = 0; i < workspaces.size(); ++i)
			PX_DELETE(workspaces[i]);
	}

	NewtonIslandWorkspace* acquire()
	{
		PxMutex::ScopedLock lock(mutex);
		if(!available.empty())
			return available.popBack();
		// Reserve both registries before publishing an object. PxArray reports
		// allocation failures through its return value, not a C++ exception.
		const PxU32 count = workspaces.size();
		if(count == PX_MAX_U32 || !workspaces.reserve(count + 1) || !available.reserve(count + 1))
			return NULL;
		void* memory = PxReflectionAllocator<NewtonIslandWorkspace>::allocate(sizeof(NewtonIslandWorkspace), PX_FL);
		if(!memory)
			return NULL;
		NewtonIslandWorkspace* workspace;
		try
		{
			workspace = PX_PLACEMENT_NEW(memory, NewtonIslandWorkspace);
		}
		catch(...)
		{
			PxReflectionAllocator<NewtonIslandWorkspace>::deallocate(memory);
			throw;
		}
		if(!workspaces.pushBack(workspace))
		{
			PX_DELETE(workspace);
			return NULL;
		}
		return workspace;
	}

	void release(NewtonIslandWorkspace* workspace)
	{
		PxMutex::ScopedLock lock(mutex);
		// acquire reserves one return slot per owned workspace before allocation.
		PX_ASSERT(available.size() < available.capacity());
		available.pushBack(workspace);
	}

	void report(const char* message)
	{
		if(PxAtomicCompareExchange(&errorReported, 1, 0) == 0)
			PxGetFoundation().error(PxErrorCode::eINVALID_OPERATION, PX_FL, "%s Island integration was skipped.", message);
	}

	newton::Settings settings;
	PxReal regularization;
	PxU64 update;
	PxArray<NewtonBodySeed> seeds;
	PxMutex mutex;
	PxArray<NewtonIslandWorkspace*> workspaces;
	PxArray<NewtonIslandWorkspace*> available;
	volatile PxI32 errorReported;
};

NewtonSolver* createNewtonSolver(const PxSceneDesc& desc)
{
	// Check allocation before placement construction, and contain exceptions
	// before returning through SDK translation units compiled without exceptions.
	void* memory = NULL;
	try
	{
		memory = PxReflectionAllocator<NewtonSolver>::allocate(sizeof(NewtonSolver), PX_FL);
		if(memory)
			return PX_PLACEMENT_NEW(memory, NewtonSolver)(desc);
	}
	catch(...)
	{
		PxReflectionAllocator<NewtonSolver>::deallocate(memory);
	}
	return NULL;
}

void destroyNewtonSolver(NewtonSolver* solver)
{
	PX_DELETE(solver);
}

bool beginNewtonUpdate(NewtonSolver& solver, PxU32 nodeCount)
{
	++solver.update;
	if(nodeCount > solver.seeds.size() && !solver.seeds.resize(nodeCount))
	{
		solver.report("Newton warm-start storage allocation failed.");
		return false;
	}
	return true;
}

static bool matchesNewtonSeed(const NewtonBodySeed& seed, const PxsBodyCore& body, PxU64 update)
{
	return seed.update + 1 == update && seed.body == &body && seed.pose.p == body.body2World.p &&
		seed.pose.q == body.body2World.q && seed.inverseMass == body.inverseMass &&
		seed.inverseInertia == body.inverseInertia && seed.lockFlags == PxU8(body.lockFlags);
}

static void applyNewtonLocks(PxSolverBodyData& data, PxU8 lockFlags)
{
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		if(lockFlags & (1 << axis))
			data.linearVelocity[axis] = 0.0f;
		if(lockFlags & (1 << (axis + 3)))
			data.angularVelocity[axis] = 0.0f;
	}
	if(!(lockFlags & 0x38))
		return;

	// Restrict the physical inverse inertia to the unlocked world axes. A symmetric
	// square root keeps the Newton response symmetric and makes integration's lock
	// clamp redundant; merely clamping the solved velocity would violate the rows.
	newton::Mat3 inverseRoot;
	for(PxU32 column = 0; column < 3; ++column)
		for(PxU32 row = 0; row < 3; ++row)
			inverseRoot(row, column) = data.sqrtInvInertia[column][row];
	newton::Mat3 response = inverseRoot * inverseRoot.transpose();
	for(PxU32 axis = 0; axis < 3; ++axis)
		if(lockFlags & (1 << (axis + 3)))
		{
			response.row(axis).setZero();
			response.col(axis).setZero();
		}
	Eigen::SelfAdjointEigenSolver<newton::Mat3> eigen(response);
	inverseRoot = eigen.eigenvectors() * eigen.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal() * eigen.eigenvectors().transpose();
	for(PxU32 column = 0; column < 3; ++column)
		for(PxU32 row = 0; row < 3; ++row)
			data.sqrtInvInertia[column][row] = (lockFlags & ((1 << (row + 3)) | (1 << (column + 3)))) ? 0.0f : PxReal(inverseRoot(row, column));
}

static void prepareNewtonBodies(NewtonSolver& solver, NewtonIslandWorkspace& workspace,
	ThreadContext& threadContext, PxSolverBodyData* bodyData, PxU32 bodyCount, PxReal timestep)
{
	newton::Problem& problem = workspace.problem;
	problem.timestep = timestep;
	problem.inverseMass.resize(bodyCount);
	problem.massDiagonal.resize(6 * bodyCount);
	problem.freeBodyVelocity.resize(6 * bodyCount);
	workspace.previous.primal.setZero(6 * bodyCount);
	if(!workspace.lockFlags.resize(bodyCount))
		throw std::bad_alloc();

	for(PxU32 i = 0; i < bodyCount; ++i)
	{
		const PxsBodyCore& body = *threadContext.mBodyCoreArray[i];
		PxSolverBodyData& data = bodyData[i];
		workspace.lockFlags[i] = PxU8(body.lockFlags);
		if(body.lockFlags)
			applyNewtonLocks(data, PxU8(body.lockFlags));
		problem.inverseMass[i] = data.invMass;
		const double mass = data.invMass > 0.0f ? 1.0 / data.invMass : 1.0;
		const PxVec3 inertia(body.inverseInertia.x > 0.0f ? 1.0f / body.inverseInertia.x : 1.0f,
			body.inverseInertia.y > 0.0f ? 1.0f / body.inverseInertia.y : 1.0f,
			body.inverseInertia.z > 0.0f ? 1.0f / body.inverseInertia.z : 1.0f);
		const PxMat33 rotation(data.body2World.q);
		for(PxU32 axis = 0; axis < 3; ++axis)
		{
			problem.massDiagonal[6 * i + axis] = mass;
			// This diagonal is only the stopping metric. Row response uses the full
			// world inertia, including off-diagonal terms, throughout the solve.
			problem.massDiagonal[6 * i + axis + 3] =
				double(rotation.column0[axis]) * rotation.column0[axis] * inertia.x +
				double(rotation.column1[axis]) * rotation.column1[axis] * inertia.y +
				double(rotation.column2[axis]) * rotation.column2[axis] * inertia.z;
			problem.freeBodyVelocity[6 * i + axis] = data.linearVelocity[axis];
			problem.freeBodyVelocity[6 * i + axis + 3] = data.angularVelocity[axis];
		}
		const NewtonBodySeed& seed = solver.seeds[threadContext.mNodeIndexArray[i]];
		if(matchesNewtonSeed(seed, body, solver.update) && seed.timestep == timestep)
		{
			const double rootMass = std::sqrt(mass);
			// Transform a physical world-space correction into the current inertia
			// basis. Reusing last frame's whitened angular vector after rotation is wrong.
			const PxVec3 localAngular = data.body2World.q.rotateInv(seed.angularCorrection);
			PxVec3 localScaled(0.0f);
			for(PxU32 axis = 0; axis < 3; ++axis)
				if(body.inverseInertia[axis] > 0.0f)
					localScaled[axis] = PxReal(localAngular[axis] / std::sqrt(double(body.inverseInertia[axis])));
			const PxVec3 angular = data.body2World.q.rotate(localScaled);
			for(PxU32 axis = 0; axis < 3; ++axis)
			{
				if(!(PxU8(body.lockFlags) & (1 << axis)) && data.invMass > 0.0f)
					workspace.previous.primal[6 * i + axis] = seed.linearCorrection[axis] * rootMass;
				if(!(PxU8(body.lockFlags) & (1 << (axis + 3))))
					workspace.previous.primal[6 * i + axis + 3] = angular[axis];
			}
		}
	}
}

static PxI32 newtonBodyIndex(PxU32 dataIndex, PxU32 firstBodyIndex, PxU32 bodyCount)
{
	return dataIndex > firstBodyIndex && dataIndex <= firstBodyIndex + bodyCount ? PxI32(dataIndex - firstBodyIndex - 1) : -1;
}

static const char* prepareNewtonRows(NewtonSolver& solver, NewtonIslandWorkspace& workspace, DynamicsContext& context,
	ThreadContext& threadContext, PxSolverBodyData* bodyData, PxU32 firstBodyIndex, PxU32 bodyCount)
{
	newton::Problem& problem = workspace.problem;
	problem.clearContacts();
	workspace.joints.clear();
	workspace.contacts.clear();
	NewtonJointSettings jointSettings;
	jointSettings.timestep = context.getDt();
	jointSettings.regularization = solver.regularization;
	jointSettings.bodyLockFlags = workspace.lockFlags.begin();
	jointSettings.initialVelocities = threadContext.motionVelocityArray;
	NewtonContactSettings contactSettings;
	contactSettings.timestep = context.getDt();
	contactSettings.regularization = solver.regularization;
	contactSettings.bounceThreshold = context.getBounceThreshold();
	contactSettings.ccdMaxSeparation = context.getCCDSeparationThreshold();
	contactSettings.dilatancyTolerance = 1.0e-5f * context.getLengthScale();
	contactSettings.correctDilatancy = true;
	contactSettings.bodyLockFlags = workspace.lockFlags.begin();
	contactSettings.initialVelocities = threadContext.motionVelocityArray;
	for(PxU32 i = 0; i < threadContext.contactDescArraySize; ++i)
	{
		const PxSolverConstraintDesc& desc = threadContext.contactConstraintDescArray[i];
		const PxSolverBodyData& body0 = bodyData[desc.bodyADataIndex];
		const PxSolverBodyData& body1 = bodyData[desc.bodyBDataIndex];
		const PxI32 index0 = newtonBodyIndex(desc.bodyADataIndex, firstBodyIndex, bodyCount);
		const PxI32 index1 = newtonBodyIndex(desc.bodyBDataIndex, firstBodyIndex, bodyCount);
		const char* error;
		if(desc.constraintType == DY_SC_TYPE_RB_1D)
		{
			const Constraint& constraint = *reinterpret_cast<const Constraint*>(desc.constraint);
			error = prepareNewtonJoint(constraint, body0, body1, index0, index1,
				&context.getConstraintWriteBackPool()[constraint.index], jointSettings, problem, workspace.joints);
		}
		else
		{
			PxsContactManager& manager = *reinterpret_cast<PxsContactManager*>(desc.constraint);
			PxsContactManagerOutput& output = context.mOutputIterator.getContactManagerOutput(manager.getWorkUnit().mNpIndex);
			error = prepareNewtonContacts(manager, output, body0, body1, index0, index1,
				contactSettings, threadContext, problem, workspace.contacts);
		}
		if(error)
			return error;
	}
	return NULL;
}

static void storeNewtonCorrection(const newton::Result& result, const PxSolverBodyData* data,
	Cm::SpatialVector* motion, PxSolverBody* bodies, PxU32 bodyCount)
{
	for(PxU32 i = 0; i < bodyCount; ++i)
	{
		const double rootInverseMass = std::sqrt(double(data[i].invMass));
		PxVec3 linear, angular;
		for(PxU32 axis = 0; axis < 3; ++axis)
		{
			linear[axis] = PxReal(result.primal[6 * i + axis] * rootInverseMass);
			angular[axis] = PxReal(result.primal[6 * i + axis + 3]);
		}
		if(motion)
		{
			motion[i].linear = linear;
			motion[i].angular = angular;
		}
		else
		{
			bodies[i].linearVelocity = linear;
			bodies[i].angularState = angular;
		}
	}
}

static bool solveNewtonSystem(NewtonSolver& solver, NewtonIslandWorkspace& workspace, const newton::Result* previous,
	newton::ParallelExecutor* parallelExecutor)
{
	newton::Settings settings = solver.settings;
	settings.parallelExecutor = parallelExecutor;
	const newton::SolveStatus::Enum status = newton::solveNewton(workspace.problem, settings,
		workspace.result, workspace.numeric, previous);
	return status == newton::SolveStatus::eSUCCESS || status == newton::SolveStatus::eITERATION_LIMIT;
}

static bool continueNewtonSystem(NewtonSolver& solver, NewtonIslandWorkspace& workspace, newton::ParallelExecutor* parallelExecutor)
{
	workspace.previous = workspace.result;
	newton::Settings settings = solver.settings;
	settings.parallelExecutor = parallelExecutor;
	const newton::SolveStatus::Enum status = newton::continueNewton(workspace.problem, settings,
		workspace.result, workspace.numeric, &workspace.previous);
	if(status == newton::SolveStatus::eSUCCESS || status == newton::SolveStatus::eITERATION_LIMIT)
		return true;
	workspace.result = workspace.previous;
	return false;
}

static void profileNewtonSolve(const NewtonIslandWorkspace& workspace, PxU64 contextId)
{
	PX_UNUSED(workspace);
	PX_UNUSED(contextId);
	PX_PROFILE_VALUE(workspace.result.iterations, "Dynamics.newtonIterations", contextId);
	PX_PROFILE_VALUE(workspace.result.factorizations, "Dynamics.newtonFactorizations", contextId);
	PX_PROFILE_VALUE(workspace.result.rankUpdates, "Dynamics.newtonRankUpdates", contextId);
	PX_PROFILE_VALUE(workspace.result.lineSearchEvaluations, "Dynamics.newtonLineEvaluations", contextId);
	PX_PROFILE_VALUE(workspace.problem.bodyCount(), "Dynamics.newtonBodies", contextId);
	PX_PROFILE_VALUE(workspace.problem.rowCount(), "Dynamics.newtonRows", contextId);
	PX_PROFILE_VALUE(int(workspace.problem.patches.size()), "Dynamics.newtonPatches", contextId);
	PX_PROFILE_VALUE(int(workspace.result.status), "Dynamics.newtonStatus", contextId);
	PX_PROFILE_VALUE(PxReal(workspace.result.scaledGradient), "Dynamics.newtonScaledGradient", contextId);
}

static bool solveNewtonRows(NewtonSolver& solver, NewtonIslandWorkspace& workspace, DynamicsContext& context,
	ThreadContext& threadContext, PxSolverBody* bodies, PxSolverBodyData* allBodyData, PxU32 firstBodyIndex, PxU32 bodyCount,
	newton::ParallelExecutor* parallelExecutor)
{
	PX_PROFILE_ZONE("Dynamics.newtonIsland", context.getContextId());
	PxSolverBodyData* bodyData = allBodyData + firstBodyIndex + 1;
	{
		PX_PROFILE_ZONE("Dynamics.newtonPrepare", context.getContextId());
		prepareNewtonBodies(solver, workspace, threadContext, bodyData, bodyCount, context.getDt());
		const char* error = prepareNewtonRows(solver, workspace, context, threadContext, allBodyData, firstBodyIndex, bodyCount);
		if(error)
		{
			solver.report(error);
			return false;
		}
		if(newton::prepareProblem(workspace.problem) != newton::SolveStatus::eSUCCESS)
		{
			solver.report("Newton equation preparation failed.");
			return false;
		}
	}
	threadContext.mAxisConstraintCount = PxU32(workspace.problem.rowCount());
	{
		PX_PROFILE_ZONE("Dynamics.newtonSolve", context.getContextId());
		if(!solveNewtonSystem(solver, workspace, &workspace.previous, parallelExecutor))
		{
			solver.report("Newton position solve failed.");
			return false;
		}
		const PxReal velocityTolerance = 1.0e-5f * context.getLengthScale() / context.getDt();
		for(PxU32 iteration = 0; iteration < 4; ++iteration)
		{
			if(!updateNewtonDilatancyBias(workspace.contacts, workspace.problem,
				workspace.result, velocityTolerance))
				break;
			if(!continueNewtonSystem(solver, workspace, parallelExecutor))
			{
				solver.report("Newton dilatancy correction failed.");
				break;
			}
		}
		profileNewtonSolve(workspace, context.getContextId());
	}
	storeNewtonCorrection(workspace.result, bodyData, threadContext.motionVelocityArray, NULL, bodyCount);
	storeNewtonCorrection(workspace.result, bodyData, NULL, bodies, bodyCount);
	writebackNewtonJoints(workspace.joints, workspace.problem, workspace.result);
	writebackNewtonContacts(workspace.contacts, workspace.problem, workspace.result, context);
	for(PxU32 i = 0; i < bodyCount; ++i)
	{
		NewtonBodySeed& seed = solver.seeds[threadContext.mNodeIndexArray[i]];
		const PxsBodyCore& body = *threadContext.mBodyCoreArray[i];
		seed.body = &body;
		seed.update = solver.update;
		seed.linearCorrection = bodies[i].linearVelocity;
		seed.angularCorrection = bodyData[i].sqrtInvInertia * bodies[i].angularState;
		seed.inverseMass = body.inverseMass;
		seed.inverseInertia = body.inverseInertia;
		seed.lockFlags = PxU8(body.lockFlags);
		seed.timestep = context.getDt();
	}
	return true;
}

static bool solveNewtonIslandInternal(NewtonSolver& solver, DynamicsContext& context, ThreadContext& threadContext,
	PxSolverBody* bodies, PxSolverBodyData* bodyData, PxU32 firstBodyIndex, PxU32 bodyCount, newton::ParallelExecutor* parallelExecutor)
{
	NewtonIslandWorkspace* workspace = NULL;
	try
	{
		workspace = solver.acquire();
		if(!workspace)
		{
			solver.report("Newton workspace allocation failed.");
			return false;
		}
		const bool success = solveNewtonRows(solver, *workspace, context, threadContext, bodies, bodyData, firstBodyIndex, bodyCount, parallelExecutor);
		solver.release(workspace);
		return success;
	}
	catch(const std::bad_alloc&)
	{
		solver.report("Newton workspace allocation failed.");
	}
	catch(...)
	{
		solver.report("Newton island preparation failed.");
	}
	if(workspace)
		solver.release(workspace);
	return false;
}

bool solveNewtonIsland(NewtonSolver& solver, DynamicsContext& context, ThreadContext& threadContext,
	PxSolverBody* bodies, PxSolverBodyData* bodyData, PxU32 firstBodyIndex, PxU32 bodyCount,
	PxBaseTask* continuation, PxU32 workerCount)
{
	NewtonParallelExecutor parallelExecutor(context, continuation, workerCount);
	const bool success = solveNewtonIslandInternal(solver, context, threadContext, bodies, bodyData,
		firstBodyIndex, bodyCount, &parallelExecutor);
	parallelExecutor.finish();
	return success;
}

void saveNewtonPoses(NewtonSolver& solver, PxsBodyCore* const* bodies, const PxU32* nodeIndices, PxU32 bodyCount)
{
	for(PxU32 i = 0; i < bodyCount; ++i)
		solver.seeds[nodeIndices[i]].pose = bodies[i]->body2World;
}
}
}
