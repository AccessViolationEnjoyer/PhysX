// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "ExtDefaultCpuDispatcher.h"
#include "ExtCpuWorkerThread.h"
#include "ExtTaskQueueHelper.h"
#include "foundation/PxString.h"

using namespace physx;

PxDefaultCpuDispatcher* physx::PxDefaultCpuDispatcherCreate(PxU32 numThreads, PxU32* affinityMasks, PxDefaultCpuDispatcherWaitForWorkMode::Enum mode, PxU32 yieldProcessorCount)
{
	return PX_NEW(Ext::DefaultCpuDispatcher)(numThreads, affinityMasks, mode, yieldProcessorCount);
}

#if !PX_SWITCH
void Ext::DefaultCpuDispatcher::getAffinityMasks(PxU32* affinityMasks, PxU32 threadCount)
{
	for(PxU32 i=0; i < threadCount; i++)
	{
		affinityMasks[i] = 0;
	}
}
#endif

Ext::DefaultCpuDispatcher::DefaultCpuDispatcher(PxU32 numThreads, PxU32* affinityMasks, PxDefaultCpuDispatcherWaitForWorkMode::Enum mode, PxU32 yieldProcessorCount) : mNumThreads(numThreads), mShuttingDown(false)
#if PX_PROFILE
	,mRunProfiled(true)
#else
	,mRunProfiled(false)
#endif
	, mWaitForWorkMode(mode)
	, mYieldProcessorCount(yieldProcessorCount)
{
	PX_CHECK_MSG((((PxDefaultCpuDispatcherWaitForWorkMode::eYIELD_PROCESSOR == mWaitForWorkMode) && (mYieldProcessorCount > 0)) ||
					(((PxDefaultCpuDispatcherWaitForWorkMode::eYIELD_THREAD == mWaitForWorkMode) || (PxDefaultCpuDispatcherWaitForWorkMode::eWAIT_FOR_WORK == mWaitForWorkMode)) && (0 == mYieldProcessorCount))), "Illegal yield processor count for chosen execute mode");
	mSleepingThreads = 0;
	mSleepOrder = 0;

	PxU32* defaultAffinityMasks = NULL;

	if(!affinityMasks)
	{
		defaultAffinityMasks = PX_ALLOCATE(PxU32, numThreads, "ThreadAffinityMasks");
		getAffinityMasks(defaultAffinityMasks, numThreads);
		affinityMasks = defaultAffinityMasks;
	}
	 
	// initialize threads first, then start

	mWorkerThreads = PX_ALLOCATE(CpuWorkerThread, numThreads, "CpuWorkerThread");
	const PxU32 nameLength = 32;
	mThreadNames = PX_ALLOCATE(PxU8, nameLength * numThreads, "CpuWorkerThreadName");

	if (mWorkerThreads)
	{
		for(PxU32 i = 0; i < numThreads; ++i)
		{
			PX_PLACEMENT_NEW(mWorkerThreads+i, CpuWorkerThread)();
			mWorkerThreads[i].initialize(this);
		}

		for(PxU32 i = 0; i < numThreads; ++i)
		{
			if (mThreadNames)
			{
				char* threadName = reinterpret_cast<char*>(mThreadNames + (i*nameLength));
				Pxsnprintf(threadName, nameLength, "PxWorker%02d", i);
				mWorkerThreads[i].setName(threadName);
			}

			mWorkerThreads[i].setAffinityMask(affinityMasks[i]);
			mWorkerThreads[i].start(PxThread::getDefaultStackSize());
		}

		PX_FREE(defaultAffinityMasks);
	}
	else
	{
		mNumThreads = 0;
	}
}

Ext::DefaultCpuDispatcher::~DefaultCpuDispatcher()
{
	for(PxU32 i = 0; i < mNumThreads; ++i)
		mWorkerThreads[i].signalQuit();

	mShuttingDown = true;
	if(PxDefaultCpuDispatcherWaitForWorkMode::eWAIT_FOR_WORK == mWaitForWorkMode)
	{
		for(PxU32 i = 0; i < mNumThreads; ++i)
			mWorkerThreads[i].wake();
	}
	for(PxU32 i = 0; i < mNumThreads; ++i)
		mWorkerThreads[i].waitForQuit();

	for(PxU32 i = 0; i < mNumThreads; ++i)
		mWorkerThreads[i].~CpuWorkerThread();

	PX_FREE(mWorkerThreads);
	PX_FREE(mThreadNames);
}

void Ext::DefaultCpuDispatcher::release()
{
	PX_DELETE_THIS;
}

void Ext::DefaultCpuDispatcher::submitTask(PxBaseTask& task)
{
	if(!mNumThreads)
	{
		// no worker threads, run directly
		runTask(task);
		task.release();
		return;
	}

	// TODO: Could use TLS to make this more efficient
	const PxThread::Id currentThread = PxThread::getId();
	const PxU32 nbThreads = mNumThreads;
	for(PxU32 i=0; i<nbThreads; ++i)
	{
		if(mWorkerThreads[i].tryAcceptJobToLocalQueue(task, currentThread))
		{
			if(PxDefaultCpuDispatcherWaitForWorkMode::eWAIT_FOR_WORK == mWaitForWorkMode)
				wakeSleepingThread();
			else
				PX_ASSERT(PxDefaultCpuDispatcherWaitForWorkMode::eYIELD_PROCESSOR == mWaitForWorkMode || PxDefaultCpuDispatcherWaitForWorkMode::eYIELD_THREAD == mWaitForWorkMode);
			return;
		}
	}

	if(mHelper.tryAcceptJobToQueue(task))
	{
		if(PxDefaultCpuDispatcherWaitForWorkMode::eWAIT_FOR_WORK == mWaitForWorkMode)
			wakeSleepingThread();
	}
}

void Ext::DefaultCpuDispatcher::wakeSleepingThread()
{
	// The locked read orders it after the job's enqueue. A worker that announced itself
	// later re-checks the queues, so no wake is needed when none had announced. One job
	// wakes one claimed worker; the others keep sleeping.
	if(PxAtomicAdd(&mSleepingThreads, 0) <= 0)
		return;
	// Prefer the most recent sleeper: its core is least likely to be in a deep idle state.
	const PxU32 nbThreads = mNumThreads;
	for(;;)
	{
		CpuWorkerThread* newest = NULL;
		PxI32 newestOrder = PX_MIN_I32;
		for(PxU32 i = 0; i < nbThreads; ++i)
		{
			const PxI32 order = mWorkerThreads[i].sleepOrder();
			if(order > newestOrder)
			{
				newest = mWorkerThreads + i;
				newestOrder = order;
			}
		}
		if(!newest)
			return;
		if(newest->claimSleeping())
		{
			PxAtomicDecrement(&mSleepingThreads);
			newest->wake();
			return;
		}
	}
}

void Ext::DefaultCpuDispatcher::prepareToWait(CpuWorkerThread& worker)
{
	PX_ASSERT(PxDefaultCpuDispatcherWaitForWorkMode::eWAIT_FOR_WORK == mWaitForWorkMode);
	// The worker resets its own signal before announcing, so a claim after the
	// announcement always wakes it.
	worker.announceSleep(PxAtomicIncrement(&mSleepOrder));
	PxAtomicIncrement(&mSleepingThreads);

	// A thread loops as follows while quit is not signaled:
	// 1) fetch work; if there is work, process it
	// 2) otherwise reset its wake signal, announce itself and fetch work again
	// 3) if there is still no work, wait for its wake signal
	//
	// A reset after shutdown signaled the wake up would leave the thread waiting
	// forever, so the signal is sent again during shutdown.
	if (mShuttingDown)
		worker.wake();
}

void Ext::DefaultCpuDispatcher::cancelWait(CpuWorkerThread& worker)
{
	// A submitter that already claimed the worker has removed it from the count.
	if(worker.claimSleeping())
		PxAtomicDecrement(&mSleepingThreads);
}
