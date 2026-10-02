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
// normal0 is the contact normal in body 0's frame when the slip began; the slip lapses when the
// normal turns far enough in that frame that the contact has moved to other surfaces.
struct AnvilFrictionState
{
	PxReal slip[3];
	PxReal twist;
	PxReal normalImpulse;
	PxReal normal0[3];
};

static const PxReal ANVIL_SLIDING_TWIST = PX_MAX_F32;
// The slip correction's stiffness relative to penetration's. Accumulated slip does not relax:
// relaxing it let every held load creep (a card house by 0.03 mm/s, a 150 N grip by 0.007 mm/s
// with a 1 s relaxation). Slip that outlives its surfaces, kept by a hull that tipped onto
// another face, slowed collapsing piles instead; it lapses with the normal's turn. Four times
// this stiffness made the pallet five times slower.
static const double ANVIL_SLIP_STIFFNESS = 2.0;
// Normal rows of contacts with friction use this fraction of the contact regularization, so
// their friction rows are eight times softer, as the four-edge pyramid used before had at
// friction 0.5. Softer normals take more iterations on large piles and when many contacts
// change between sticking and sliding, and penetrate further (pallet slipsheets 9.3 um at 1/8,
// 10.6 um at 1/4, 13.8 um at 1); stiffer ones take more on resting stacks.
static const double ANVIL_FRICTION_NORMAL_REGULARIZATION = 0.125;
// A speculative contact point (offsets inflated by a body's motion) enters the first solve within
// this gap, or when its approach over the step, scaled by this margin, covers the gap. Others are
// deferred: rows that stay inactive would only cost preparation and factorization. The solve
// itself decides whether a deferred point's gap closes (an impact can turn a body into one), and
// then the point joins the problem; these only choose the first solve's points. The approach is
// toward the surface's target velocity: a pallet pivoting off a belt's end closes on the edge by
// the belt speed's component along the slanted edge normal, and ignoring it re-solved every
// fall-off step. It is taken before this step's gravity: with it, every hull of a resting pile
// approached the points below it by the fall of one step, and their rows cost 16% of the step.
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
	// The pair's deferred points (speculative points left out of the first solve), summarized
	// so the solved velocities can rule out closing any of them without reading the contacts
	// again: a point closes when its relative normal velocity falls below the surface's target
	// speed less its gap per step, and deferredLimit is the largest of those over the points
	// (-PX_MAX_F32 without any). The relative velocity at a point is at least the bodies'
	// relative velocity along deferredNormal, their common normal (zero when they differ, which
	// bounds by the speed instead), less each body's angular speed times its reach, the longest
	// arm from the body to one of the points.
	PxReal deferredLimit;
	PxReal deferredReach[2];
	PxVec3 deferredNormal;
	PxsContactManager* manager;
	const PxSolverBodyData* bodyData[2];
};

struct AnvilBodyVelocity
{
	PxVec3 linear;
	PxVec3 angular;
};

struct AnvilContactRows
{
	PxArray<AnvilContactPoint> points;
	PxArray<AnvilContactPair> pairs;
	// Points from lateBegin on joined after a solve closed their gaps; latePairs and
	// lateContacts hold each one's pair and its index among the pair's contacts.
	PxU32 lateBegin;
	PxArray<PxU32> latePairs;
	PxArray<PxU16> lateContacts;
	// Scratch: the island's body velocities after a solve.
	PxArray<AnvilBodyVelocity> solvedVelocities;

	AnvilContactRows() : lateBegin(0) {}

	void clear()
	{
		points.clear();
		pairs.clear();
		latePairs.clear();
		lateContacts.clear();
		lateBegin = 0;
	}
};

// The pair's friction state comes from frictionStream when given, otherwise it is reservedState,
// which may be NULL. buffer is scratch storage for the pair's contacts.
void prepareAnvilContacts(PxsContactManager& manager, PxsContactManagerOutput& contactOutput, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const AnvilContactSettings& settings, PxContactBuffer& buffer, FrictionPatchStreamPair* frictionStream, AnvilFrictionState* reservedState, anvil::Problem& problem, AnvilContactRows& output);

// Adds the rows of deferred points whose gaps the solved velocities close within the step, and
// returns whether it added any, so the problem must be prepared and solved again. settings must
// be the ones the rows were prepared with; bodyData holds the island's bodyCount bodies; buffer
// is scratch storage.
bool addClosedAnvilContacts(AnvilContactRows& rows, anvil::Problem& problem, const anvil::Result& result, const AnvilContactSettings& settings, const PxSolverBodyData* bodyData, PxU32 bodyCount, PxContactBuffer& buffer, DynamicsContext& context);

void writebackAnvilContacts(const AnvilContactRows& rows, const anvil::Problem& problem, const anvil::Result& result, const PxSolverBody* bodies, const PxSolverBodyData* bodyData, DynamicsContext& context);

}
}

#endif
