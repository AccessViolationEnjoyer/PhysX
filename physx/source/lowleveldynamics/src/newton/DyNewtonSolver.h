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

#ifndef DY_NEWTON_SOLVER_H
#define DY_NEWTON_SOLVER_H

#include "foundation/PxSimpleTypes.h"

namespace physx
{
class PxSceneDesc;
class PxBaseTask;
struct PxSolverBody;
struct PxSolverBodyData;
struct PxsBodyCore;

namespace Dy
{
class DynamicsContext;
class NewtonSolver;
class ThreadContext;

NewtonSolver* createNewtonSolver(const PxSceneDesc& desc);
void destroyNewtonSolver(NewtonSolver* solver);

// Advance even on empty updates: recycled island node indices must not inherit an old warm start.
void beginNewtonUpdate(NewtonSolver& solver, PxU32 nodeCount);
bool solveNewtonIsland(NewtonSolver& solver, DynamicsContext& context, ThreadContext& threadContext, PxSolverBody* bodies, PxSolverBodyData* bodyData, PxU32 firstBodyIndex, PxU32 bodyCount, PxBaseTask* continuation, PxU32 workerCount);
void saveNewtonPoses(NewtonSolver& solver, PxsBodyCore* const* bodies, const PxU32* nodeIndices, PxU32 bodyCount);
}
}

#endif
