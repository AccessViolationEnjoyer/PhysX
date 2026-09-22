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
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.

#ifndef DY_NEWTON_CONSTRAINT_PREP_H
#define DY_NEWTON_CONSTRAINT_PREP_H

#include "foundation/PxArray.h"
#include "foundation/PxVec3.h"

namespace newton
{
struct Problem;
struct Result;
}

namespace physx
{
struct PxSolverBodyData;

namespace Cm
{
class SpatialVector;
}

namespace Dy
{
struct Constraint;
struct ConstraintWriteback;

struct NewtonJointSettings
{
	PxReal timestep;
	// Hard rows use R = regularization * J M^-1 J'. Spring compliance comes from k and d.
	PxReal regularization;
	const PxU8* bodyLockFlags = NULL;
	const Cm::SpatialVector* initialVelocities = NULL;
};

struct NewtonJointRow
{
	PxU32 contactIndex;
	// World-space Jacobian, before mass scaling. Zero for rows without OUTPUT_FORCE.
	PxVec3 linear0;
	PxVec3 angular0;
};

struct NewtonJointWriteback
{
	ConstraintWriteback* destination;
	PxU32 firstRow;
	PxU32 rowCount;
	PxVec3 body0WorldOffset;
	PxReal linearBreakImpulse;
	PxReal angularBreakImpulse;
};

struct NewtonJointRows
{
	PxArray<NewtonJointRow> rows;
	PxArray<NewtonJointWriteback> joints;

	void clear()
	{
		rows.clear();
		joints.clear();
	}
};

// The body index is the island's dynamic-body index, or -1 for a static or kinematic endpoint.
// The body data contains free velocities; the callback is invoked once and its rows are copied.
void prepareNewtonJoint(const Constraint& constraint, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, ConstraintWriteback* writeback, const NewtonJointSettings& settings, newton::Problem& problem, NewtonJointRows& output);

// Final physical impulses, including the CoM-to-anchor moment correction used by PGS.
void writebackNewtonJoints(const NewtonJointRows& rows, const newton::Problem& problem, const newton::Result& result);
}
}

#endif
