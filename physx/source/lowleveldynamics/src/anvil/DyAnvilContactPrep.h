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
	const PxU8* bodyLockFlags = NULL;
	const Cm::SpatialVector* initialVelocities = NULL;
};

struct AnvilContactPoint
{
	PxU32 firstContact;
	PxU32 contactCount;
	PxReal* destination;
};

struct AnvilFrictionPoint
{
	PxU32 pointIndex;
	PxReal friction;
	PxReal freeNormalVelocity;
	PxReal targetNormalVelocity;
	PxReal freeTangentVelocity0;
	PxReal freeTangentVelocity1;
	PxReal dilatancyBias;
	PxU8* state;
	bool correctDilatancy;
};

struct AnvilContactPair
{
	PxU32 firstPoint;
	PxU32 pointCount;
	ThresholdStreamElement threshold;
	bool reportThreshold;
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

void writebackAnvilContacts(const AnvilContactRows& rows, const anvil::Problem& problem, const anvil::Result& result, DynamicsContext& context);

bool updateAnvilDilatancyBias(AnvilContactRows& rows, anvil::Problem& problem, const anvil::Result& result, PxReal velocityTolerance);
}
}

#endif
