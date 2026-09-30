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
class PxContactBuffer;
class FrictionPatchStreamPair;
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
	// Impulse bound of the two friction rows that follow a scalar normal row.
	PxReal frictionLimit;
};

// A contact pair's friction state, kept from step to step in PhysX's friction stream. While the
// pair sticks, slip and twist accumulate the tangential displacement of body 0 relative to body
// 1 at the contact centre and its rotation about the normal; friction rows correct them like
// penetration, with ANVIL_SLIP_STIFFNESS times its stiffness, so a sustained friction load holds
// instead of creeping.
// A sliding pair has no accumulated slip: its twist is ANVIL_SLIDING_TWIST, a value the slip
// limit keeps accumulated twist from reaching.
// normalImpulse is the pair's total normal impulse in the last step. Friction is lagged: the
// next step bounds each point's friction rows by the friction coefficient times its share, so
// friction does not depend on the step's normal rows and sliding contacts do not separate.
struct AnvilFrictionState
{
	PxReal slip[3];
	PxReal twist;
	PxReal normalImpulse;
};

static const PxReal ANVIL_SLIDING_TWIST = PX_MAX_F32;
// Seconds over which accumulated slip relaxes, and the slip correction's stiffness relative to
// penetration's. A held load keeps creeping at a rate that falls with both; slip left by a
// transient locks tangential forces into stacks for about the relaxation time, and a stiffer
// correction holds them harder, so both also raise the solver's work on stacks.
static const PxReal ANVIL_SLIP_RELAXATION_TIME = 1.0f;
static const double ANVIL_SLIP_STIFFNESS = 2.0;
// Normal rows of contacts with friction use this fraction of the contact regularization, so
// their friction rows are eight times softer, as the four-edge pyramid used before had at
// friction 0.5. Softer normals take more iterations on large piles and when many contacts
// change between sticking and sliding, and penetrate further (pallet slipsheets 9.3 um at 1/8,
// 10.6 um at 1/4, 13.8 um at 1); stiffer ones take more on resting stacks.
static const double ANVIL_FRICTION_NORMAL_REGULARIZATION = 0.125;
// A speculative contact point (offsets inflated by a body's motion) is kept within this gap, or
// when its approach over the step, scaled by this margin, covers the gap; otherwise its rows
// could never become active and would only cost preparation and factorization.
static const PxReal ANVIL_SPECULATIVE_KEEP_GAP = 1.0e-3f;
static const PxReal ANVIL_SPECULATIVE_APPROACH_MARGIN = 1.5f;

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
	PxArray<AnvilContactPair> pairs;

	void clear()
	{
		points.clear();
		pairs.clear();
	}
};

// The pair's friction state comes from frictionStream when given, otherwise it is reservedState,
// which may be NULL. buffer is scratch storage for the pair's contacts.
void prepareAnvilContacts(PxsContactManager& manager, PxsContactManagerOutput& contactOutput, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const AnvilContactSettings& settings, PxContactBuffer& buffer, FrictionPatchStreamPair* frictionStream, AnvilFrictionState* reservedState, anvil::Problem& problem, AnvilContactRows& output);

void writebackAnvilContacts(const AnvilContactRows& rows, const anvil::Problem& problem, const anvil::Result& result, const PxSolverBody* bodies, const PxSolverBodyData* bodyData, DynamicsContext& context);

}
}

#endif
