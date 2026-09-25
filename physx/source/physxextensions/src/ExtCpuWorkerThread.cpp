// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "task/PxTask.h"
#include "ExtCpuWorkerThread.h"
#include "ExtDefaultCpuDispatcher.h"
#include "foundation/PxFPU.h"

using namespace physx;

Ext::CpuWorkerThread::CpuWorkerThread() : mOwner(NULL), mThreadId(0), mSleeping(0), mSleepOrder(0)
{
}

Ext::CpuWorkerThread::~CpuWorkerThread()
{
}

#define HighPriority	true
#define RegularPriority	false

PxBaseTask* Ext::CpuWorkerThread::findTask()
{
	// PT: look for high priority tasks first, across threads
	PxBaseTask* task = getJob<HighPriority>();
	if(!task)
		task = mOwner->fetchNextTask<HighPriority>();

	// PT: then look for regular tasks
	if(!task)
		task = getJob<RegularPriority>();
	if(!task)
		task = mOwner->fetchNextTask<RegularPriority>();
	return task;
}

void Ext::CpuWorkerThread::execute()
{
	mThreadId = getId();

	const PxDefaultCpuDispatcherWaitForWorkMode::Enum ownerWaitForWorkMode = mOwner->getWaitForWorkMode();

	while(!quitIsSignalled())
    {
		PxBaseTask* task = findTask();
		// The wake signal is reset only before sleeping, so busy workers make no
		// event calls. Jobs submitted before the announcement are found here.
		if(!task && PxDefaultCpuDispatcherWaitForWorkMode::eWAIT_FOR_WORK == ownerWaitForWorkMode)
		{
			mOwner->prepareToWait(*this);
			task = findTask();
			if(task)
				mOwner->cancelWait(*this);
		}

		if(task)
		{
			mOwner->runTask(*task);
			task->release();
		}
		else if(PxDefaultCpuDispatcherWaitForWorkMode::eYIELD_THREAD == ownerWaitForWorkMode)
		{
			PxThread::yield();
		}
		else if(PxDefaultCpuDispatcherWaitForWorkMode::eYIELD_PROCESSOR == ownerWaitForWorkMode)
		{
			const PxU32 pauseCounter = mOwner->getYieldProcessorCount();
			for(PxU32 j = 0; j < pauseCounter; j++)
				PxThread::yieldProcessor();
		}
		else
		{
			PX_ASSERT(PxDefaultCpuDispatcherWaitForWorkMode::eWAIT_FOR_WORK == ownerWaitForWorkMode);
			waitForWake();
		}
	}

	quit();
}
