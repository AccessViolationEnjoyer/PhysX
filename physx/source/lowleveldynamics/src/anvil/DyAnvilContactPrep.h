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

#ifndef DY_ANVIL_CONTACT_PREP_H
#define DY_ANVIL_CONTACT_PREP_H

#include "foundation/PxArray.h"
#include "foundation/PxVec3.h"
#include "DyThresholdTable.h"

namespace anvil
{
struct Problem;
struct Result;
}

namespace physx
{
class PxsContactManager;
struct PxsContactManagerOutput;
struct PxSolverBody;
struct PxSolverBodyData;

namespace Cm
{
class SpatialVector;
}

namespace Dy
{
class DynamicsContext;
class ThreadContext;

struct AnvilContactSettings
{
	PxReal timestep;
	PxReal regularization;
	PxReal bounceThreshold;
	PxReal ccdMaxSeparation;
	PxReal dilatancyTolerance;
	bool correctDilatancy;
	// Reference-acceleration coefficients shared by every contact of a step.
	double impedance;
	double damping;
	double stiffness;
	// Regularization at zero penetration, reaching regularization at stiffeningDepth
	// (0 disables), and log2(regularization / surfaceRegularization).
	double surfaceRegularization;
	double regularizationLogRange;
	double stiffeningDepth;
	// Accumulated slip beyond this restarts a sticking pair's slip correction.
	PxReal slipLimit;
	const PxU8* bodyLockFlags = NULL;
	const Cm::SpatialVector* initialVelocities = NULL;
};

struct AnvilContactPoint
{
	PxU32 firstContact;
	PxU32 contactCount;
	PxReal* destination;
	// Friction coefficient of the point's friction rows; 0 for a point without them.
	PxReal friction;
};

// A contact pair's friction state, kept from step to step in PhysX's friction stream. While the
// pair sticks, slip and twist accumulate the tangential displacement of body 0 relative to body
// 1 at the contact centre and its rotation about the normal; friction rows correct them like
// penetration, with ANVIL_SLIP_STIFFNESS times its stiffness, so a sustained friction load holds
// instead of creeping.
// A sliding pair has no accumulated slip: its twist is ANVIL_SLIDING_TWIST, a value the slip
// limit keeps accumulated twist from reaching.
struct AnvilFrictionState
{
	PxReal slip[3];
	PxReal twist;
};

static const PxReal ANVIL_SLIDING_TWIST = PX_MAX_F32;
// Seconds over which accumulated slip relaxes, and the slip correction's stiffness relative to
// penetration's. A held load keeps creeping at a rate that falls with both; slip left by a
// transient locks tangential forces into stacks for about the relaxation time, and a stiffer
// correction holds them harder, so both also raise the solver's work on stacks.
static const PxReal ANVIL_SLIP_RELAXATION_TIME = 1.0f;
static const double ANVIL_SLIP_STIFFNESS = 2.0;

struct AnvilFrictionPoint
{
	PxU32 pointIndex;
	PxReal friction;
	PxReal freeNormalVelocity;
	PxReal targetNormalVelocity;
	PxReal freeTangentVelocity0;
	PxReal freeTangentVelocity1;
	PxReal dilatancyBias;
	bool correctDilatancy;
};

struct AnvilContactPair
{
	PxU32 firstPoint;
	PxU32 pointCount;
	ThresholdStreamElement threshold;
	bool reportThreshold;
	// Writeback adds the step's slip to the state from the solved velocities, which move the
	// bodies; the velocities at the start of the next step also hold the applied forces.
	AnvilFrictionState* state;
	PxI32 body[2];
	PxVec3 arm[2];
	PxVec3 normal;
	// Relative velocity at the contact centre and about the normal before the solve.
	PxVec3 freeSlipVelocity;
	PxReal freeTwistVelocity;
};

struct AnvilContactRows
{
	PxArray<AnvilContactPoint> points;
	PxArray<AnvilFrictionPoint> frictionPoints;
	PxArray<AnvilContactPair> pairs;

	void clear()
	{
		points.clear();
		frictionPoints.clear();
		pairs.clear();
	}
};

void prepareAnvilContacts(PxsContactManager& manager, PxsContactManagerOutput& contactOutput, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const AnvilContactSettings& settings, ThreadContext& threadContext, anvil::Problem& problem, AnvilContactRows& output);

void writebackAnvilContacts(const AnvilContactRows& rows, const anvil::Problem& problem, const anvil::Result& result, const PxSolverBody* bodies, const PxSolverBodyData* bodyData, DynamicsContext& context);

bool updateAnvilDilatancyBias(AnvilContactRows& rows, anvil::Problem& problem, const anvil::Result& result, PxReal velocityTolerance);
}
}

#endif
