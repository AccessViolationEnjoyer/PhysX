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

#include "DyAnvilContactPrep.h"
#include "DyDynamics.h"
#include "DyThreadContext.h"
#include "PxsContactManager.h"
#include "PxsRigidBody.h"
#include "CmSpatialVector.h"
#include "foundation/PxAtomic.h"
#include "foundation/PxMemory.h"
#include "core/AnvilSolver.h"

#include <cmath>

namespace physx
{
namespace Dy
{
// A native friction-patch count cannot reach this value. Anvil uses it to identify
// the one-byte static/sliding state stored in PhysX's existing friction stream.
static const PxU8 ANVIL_FRICTION_STATE_MARKER = 0xff;

static PX_FORCE_INLINE double dotAnvilContact(const PxVec3& a, const PxVec3& b)
{
	return double(a.x) * b.x + double(a.y) * b.y + double(a.z) * b.z;
}

static PX_FORCE_INLINE bool hasContactImpulseLimit(PxReal maxImpulse)
{
	// PxsRigidCore uses 1e32f as its unlimited default.
	return maxImpulse < 1.0e32f;
}

static PX_FORCE_INLINE double contactSpeed(const PxVec3& linear0, const PxVec3& angularVelocity0, const PxVec3& linear1, const PxVec3& angularVelocity1, const PxVec3& direction, const PxVec3& angular0, const PxVec3& angular1)
{
	return dotAnvilContact(direction, linear0) - dotAnvilContact(direction, linear1) +
		dotAnvilContact(angular0, angularVelocity0) - dotAnvilContact(angular1, angularVelocity1);
}

static PX_FORCE_INLINE double contactSpeed(const PxSolverBodyData& body0, const PxSolverBodyData& body1, const PxVec3& direction, const PxVec3& angular0, const PxVec3& angular1)
{
	return contactSpeed(body0.linearVelocity, body0.angularVelocity, body1.linearVelocity, body1.angularVelocity, direction, angular0, angular1);
}

// Jacobian coefficients of one pair body, converted once for every row of the pair. A row's
// Jacobian is (s d, S a) for its direction d and angular direction a, with s the signed root
// inverse mass, zero on locked axes, and S the signed root inverse inertia. The coefficients
// are stored for the row pairs (0,1), (2,3) and (4,5):
//   (0,1) = linear01 * (d.x, d.y)
//   (2,3) = middleX * (d.z, a.x) + middleY * a.y + middleZ * a.z
//   (4,5) = upperX * a.x + upperY * a.y + upperZ * a.z
// An absent body has zero coefficients and so zero Jacobians.
struct AnvilJacobianBody
{
	double linear01[2];
	double middleX[2];
	double middleY[2];
	double middleZ[2];
	double upperX[2];
	double upperY[2];
	double upperZ[2];
};

static PX_FORCE_INLINE void prepareJacobianBody(AnvilJacobianBody& result, const PxSolverBodyData& body, PxI32 bodyIndex, double rootInverseMass, double sign, const PxU8* lockFlags)
{
	const bool present = bodyIndex >= 0;
	const PxU8 locks = lockFlags && present ? lockFlags[bodyIndex] : 0;
	const double linearScale = present ? sign * rootInverseMass : 0.0;
	const double angularSign = present ? sign : 0.0;
	double linear[3], inertia[3][3];
	const PxMat33& root = body.sqrtInvInertia;
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		linear[axis] = locks & (1 << axis) ? 0.0 : linearScale;
		inertia[axis][0] = angularSign * double(root.column0[axis]);
		inertia[axis][1] = angularSign * double(root.column1[axis]);
		inertia[axis][2] = angularSign * double(root.column2[axis]);
	}
	result.linear01[0] = linear[0];
	result.linear01[1] = linear[1];
	result.middleX[0] = linear[2];
	result.middleX[1] = inertia[0][0];
	result.middleY[0] = 0.0;
	result.middleY[1] = inertia[0][1];
	result.middleZ[0] = 0.0;
	result.middleZ[1] = inertia[0][2];
	for(PxU32 row = 0; row < 2; ++row)
	{
		result.upperX[row] = inertia[row + 1][0];
		result.upperY[row] = inertia[row + 1][1];
		result.upperZ[row] = inertia[row + 1][2];
	}
}

#if defined(ANVIL_SIMD128)
// Writes one body's Jacobian and returns its nonzero entry count.
static PX_FORCE_INLINE int contactJacobian(double* jacobian, const AnvilJacobianBody& body, anvil::simd::Double2 direction01, anvil::simd::Double2 middle, anvil::simd::Double2 angularX,
	anvil::simd::Double2 angularY, anvil::simd::Double2 angularZ)
{
	using namespace anvil;
	const simd::Double2 row01 = simd::multiply(simd::load(body.linear01), direction01);
	const simd::Double2 row23 = simd::multiplyAdd(simd::load(body.middleZ), angularZ, simd::multiplyAdd(simd::load(body.middleY), angularY, simd::multiply(simd::load(body.middleX), middle)));
	const simd::Double2 row45 = simd::multiplyAdd(simd::load(body.upperZ), angularZ, simd::multiplyAdd(simd::load(body.upperY), angularY, simd::multiply(simd::load(body.upperX), angularX)));
	simd::store(jacobian, row01);
	simd::store(jacobian + 2, row23);
	simd::store(jacobian + 4, row45);
	const simd::Double2 zero = simd::zero();
	const int nonzero01 = simd::mask(simd::notEqual(row01, zero));
	const int nonzero23 = simd::mask(simd::notEqual(row23, zero));
	const int nonzero45 = simd::mask(simd::notEqual(row45, zero));
	return (nonzero01 & 1) + (nonzero01 >> 1) + (nonzero23 & 1) + (nonzero23 >> 1) + (nonzero45 & 1) + (nonzero45 >> 1);
}
#endif

// Writes a row's two Jacobians and returns their nonzero entry count.
static PX_FORCE_INLINE int contactJacobians(anvil::CompactContact& row, const AnvilJacobianBody* bodies, const PxVec3& direction, const PxVec3& angular0, const PxVec3& angular1)
{
#if defined(ANVIL_SIMD128)
	using namespace anvil;
	const simd::Double2 direction01 = simd::make(direction.x, direction.y);
	return contactJacobian(row.jacobian[0].data(), bodies[0], direction01, simd::make(direction.z, angular0.x), simd::splat(angular0.x), simd::splat(angular0.y), simd::splat(angular0.z)) +
		contactJacobian(row.jacobian[1].data(), bodies[1], direction01, simd::make(direction.z, angular1.x), simd::splat(angular1.x), simd::splat(angular1.y), simd::splat(angular1.z));
#else
	const PxVec3* angular[2] = { &angular0, &angular1 };
	int count = 0;
	for(PxU32 end = 0; end < 2; ++end)
	{
		const AnvilJacobianBody& body = bodies[end];
		const PxVec3& a = *angular[end];
		double* jacobian = row.jacobian[end].data();
		jacobian[0] = body.linear01[0] * direction.x;
		jacobian[1] = body.linear01[1] * direction.y;
		jacobian[2] = body.middleZ[0] * a.z + (body.middleY[0] * a.y + body.middleX[0] * direction.z);
		jacobian[3] = body.middleZ[1] * a.z + (body.middleY[1] * a.y + body.middleX[1] * a.x);
		jacobian[4] = body.upperZ[0] * a.z + (body.upperY[0] * a.y + body.upperX[0] * a.x);
		jacobian[5] = body.upperZ[1] * a.z + (body.upperY[1] * a.y + body.upperX[1] * a.x);
		count += anvil::nonzeroCount6(jacobian);
	}
	return count;
#endif
}

static PX_FORCE_INLINE double prepareContactJacobian(anvil::CompactContact& row, const AnvilJacobianBody* jacobianBodies, PxI32 bodyIndex0, PxI32 bodyIndex1, const PxVec3& direction, const PxVec3& angular0, const PxVec3& angular1)
{
	row.body[0] = bodyIndex0;
	row.body[1] = bodyIndex1;
	contactJacobians(row, jacobianBodies, direction, angular0, angular1);
	return row.jacobian[0].squaredNorm() + row.jacobian[1].squaredNorm();
}

static PX_FORCE_INLINE void contactTangents(const PxVec3& normal, const PxVec3& relativeVelocity, PxVec3& tangent0, PxVec3& tangent1)
{
	// Selects rather than branches: resting and sliding points mix within a pair. Either choice
	// is nonzero, as the fallback is at least 1/sqrt(2) long, so it normalizes unconditionally.
	const PxVec3 sliding = relativeVelocity - normal * normal.dot(relativeVelocity);
	const PxVec3 fallback = PxAbs(normal.x) < 0.70710678f ? PxVec3(0.0f, -normal.z, normal.y) : PxVec3(-normal.y, normal.x, 0.0f);
	tangent0 = sliding.magnitudeSquared() <= 0.0001f ? fallback : sliding;
	tangent0 /= tangent0.magnitude();
	tangent1 = normal.cross(tangent0);
}

// 2^x for moderate x, with relative error below 1e-7. It is portable and deterministic,
// unlike the library's exp2 or pow, and cheaper. The fraction's near-minimax quintic has
// equal error at both ends, so the result stays continuous where the integer part steps,
// and its split evaluation keeps the chain of dependent multiplies short.
static PX_FORCE_INLINE double exp2Approximation(double x)
{
	int whole = int(x);
	whole -= x < double(whole) ? 1 : 0;
	const double f = x - whole;
	const double f2 = f * f;
	const double power = (0.9999999250645677 + f * 0.69315307314185293) +
		f2 * ((0.24015361754827008 + f * 0.055826316561799594) + f2 * (0.0089893418677127468 + f * 0.0018775759408752035));
	const PxU64 bits = PxU64(PxClamp(whole + 1023, 1, 2046)) << 52;
	double scale;
	PxMemCopy(&scale, &bits, sizeof(scale));
	return power * scale;
}

// Like MuJoCo's position-dependent impedance, a contact stiffens smoothly from its surface
// regularization to the regular one as penetration reaches the stiffening depth. Damping and
// stiffness keep the regular impedance; the impedance scales the position term and sets R.
// The smooth step runs in log regularization, unlike MuJoCo's step in impedance. A step in
// impedance makes nearly all of a large regularization range fall in the last few percent of
// the depth, and a light body carrying a heavy load then rocks or bounces with period two:
// its compliance, fixed at the start of each step, sets how far it sinks in that step, so
// the regularization must not fall faster than about the seventh power of the depth. In log
// regularization, the slope of log R against log depth peaks at log(surface / regular).
static PX_FORCE_INLINE double contactRegularization(const AnvilContactSettings& settings, double penetration)
{
	const double depth = -penetration / settings.stiffeningDepth;
	// Separated and touching points, such as the sides of neighbouring boxes, and deep ones
	// are common and need no power.
	if(depth <= 0.0)
	{
		return settings.surfaceRegularization;
	}
	if(depth >= 1.0)
	{
		return settings.regularization;
	}
	const double step = depth <= 0.5 ? 2.0 * depth * depth : 1.0 - 2.0 * (1.0 - depth) * (1.0 - depth);
	return settings.surfaceRegularization * exp2Approximation(step * settings.regularizationLogRange);
}

static PX_FORCE_INLINE PxVec3 initialPointVelocity(const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const PxVec3& arm0, const PxVec3& arm1, const AnvilContactSettings& settings)
{
	const PxVec3& linear0 = bodyIndex0 >= 0 ? settings.initialVelocities[bodyIndex0].linear : body0.linearVelocity;
	const PxVec3& angular0 = bodyIndex0 >= 0 ? settings.initialVelocities[bodyIndex0].angular : body0.angularVelocity;
	const PxVec3& linear1 = bodyIndex1 >= 0 ? settings.initialVelocities[bodyIndex1].linear : body1.linearVelocity;
	const PxVec3& angular1 = bodyIndex1 >= 0 ? settings.initialVelocities[bodyIndex1].angular : body1.angularVelocity;
	return linear0 + angular0.cross(arm0) - linear1 - angular1.cross(arm1);
}

// Whether a speculative contact point (offsets inflated by a body's motion) enters the first
// solve: within ANVIL_SPECULATIVE_KEEP_GAP, or when the approach over the step of the motion at
// the step's start (last step's solution, before this step's gravity, which the contacts that
// hold a resting body cancel) toward the surface's target velocity covers the gap with
// ANVIL_SPECULATIVE_APPROACH_MARGIN to spare for pushes from other contacts. Other points are
// deferred: the solve decides whether they close.
static PX_FORCE_INLINE bool speculativePointKept(const PxContactPoint& contact, PxReal gap, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const PxTransform& frame0, const PxTransform& frame1, const AnvilContactSettings& settings)
{
	if(gap <= ANVIL_SPECULATIVE_KEEP_GAP)
	{
		return true;
	}
	const PxVec3 initialVelocity = initialPointVelocity(body0, body1, bodyIndex0, bodyIndex1, contact.point - frame0.p, contact.point - frame1.p, settings);
	return gap <= ANVIL_SPECULATIVE_APPROACH_MARGIN * PxMax(0.0f, contact.normal.dot(contact.targetVel - initialVelocity)) * settings.timestep;
}

static void extractAnvilContacts(PxsContactManagerOutput& contactOutput, PxContactBuffer& buffer, PxU16* originalIndices, PxReal defaultMaxImpulse)
{
	buffer.count = 0;
	if(!contactOutput.nbContacts)
	{
		return;
	}
	PxContactStreamIterator iterator(contactOutput.contactPatches, contactOutput.contactPoints, contactOutput.getInternalFaceIndice(), contactOutput.nbPatches, contactOutput.nbContacts);
	if(iterator.forceNoResponse)
	{
		return;
	}
	PX_ASSERT(iterator.getInvMassScale0() == 1.0f && iterator.getInvMassScale1() == 1.0f && iterator.getInvInertiaScale0() == 1.0f && iterator.getInvInertiaScale1() == 1.0f);

	PxU32 originalIndex = 0;
	while(iterator.hasNextPatch())
	{
		iterator.nextPatch();
		const bool hasMaxImpulse = (iterator.patch->internalFlags & PxContactPatch::eHAS_MAX_IMPULSE) != 0;
		while(iterator.hasNextContact())
		{
			iterator.nextContact();
			const PxReal maxImpulse = hasMaxImpulse ? iterator.getMaxImpulse() : defaultMaxImpulse;
			if(maxImpulse != 0.0f)
			{
				PX_ASSERT(buffer.count < PxContactBuffer::MAX_CONTACTS);
				PxContactPoint& contact = buffer.contacts[buffer.count];
				contact.normal = iterator.getContactNormal();
				contact.point = iterator.getContactPoint();
				contact.separation = iterator.getSeparation();
				contact.materialFlags = PxU8(iterator.getMaterialFlags());
				contact.maxImpulse = maxImpulse;
				contact.staticFriction = iterator.getStaticFriction();
				contact.dynamicFriction = iterator.getDynamicFriction();
				contact.restitution = iterator.getRestitution();
				contact.damping = iterator.getDamping();
				contact.targetVel = iterator.getTargetVel();
				originalIndices[buffer.count++] = PxTo16(originalIndex);
			}
			++originalIndex;
		}
	}
}

// Nonzero Jacobian entries of a finished scalar row's present bodies.
static PX_FORCE_INLINE int rowNonzeroCount(const anvil::CompactContact& row)
{
	return (row.body[0] >= 0 ? anvil::nonzeroCount6(row.jacobian[0].data()) : 0) + (row.body[1] >= 0 ? anvil::nonzeroCount6(row.jacobian[1].data()) : 0);
}

// The speed a row's reference velocity lies below the free speed: the initial speed relaxes toward
// the target and the position error closes over the settings' time constant.
static PX_FORCE_INLINE double relaxedFreeVelocity(double initialSpeed, double freeSpeed, double targetSpeed, double positionError, double stiffness, double damping, double impedance, const AnvilContactSettings& settings)
{
	return freeSpeed - initialSpeed + settings.timestep * (damping * (initialSpeed - targetSpeed) + stiffness * impedance * positionError);
}

static PX_FORCE_INLINE void appendContactRow(const PxVec3& direction, const PxVec3& angular0, const PxVec3& angular1, double freeVelocity, double regularization, PxI32 bodyIndex0, PxI32 bodyIndex1, const AnvilJacobianBody* jacobianBodies, anvil::Problem& problem, double upperImpulse, double lowerImpulse = 0.0)
{
	anvil::CompactContact& row = problem.beginScalarContact();
	row.body[0] = bodyIndex0;
	row.body[1] = bodyIndex1;
	// Only an all-zero row is degenerate; its entry count is also its refactorization work.
	// Absent bodies have zero Jacobians, so the count covers the present ones.
	const int nonzeroCount = contactJacobians(row, jacobianBodies, direction, angular0, angular1);
	if(!nonzeroCount)
	{
		problem.cancelScalarContact();
		return;
	}
	row.freeVelocity = freeVelocity;
	row.regularization = regularization;
	problem.finishScalarContact(lowerImpulse, upperImpulse, nonzeroCount);
}

static bool prepareCompliantNormal(const PxContactPoint& contact, const PxVec3& normal, const PxVec3& angular0, const PxVec3& angular1, double penetration, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const AnvilJacobianBody* jacobianBodies, const AnvilContactSettings& settings, anvil::CompactContact& row)
{
	const double response = prepareContactJacobian(row, jacobianBodies, bodyIndex0, bodyIndex1, normal, angular0, angular1);
	if(response == 0.0)
	{
		return false;
	}
	const double speed = contactSpeed(body0, body1, normal, angular0, angular1);
	const double stiffness = -double(contact.restitution);
	const double damping = penetration >= 0.0 ? 0.0 : double(contact.damping);
	const double coefficient = settings.timestep * (damping + settings.timestep * stiffness);
	const double target = dotAnvilContact(contact.targetVel, normal);
	row.regularization = (contact.materialFlags & PxMaterialFlag::eCOMPLIANT_ACCELERATION_SPRING) ? response / coefficient : 1.0 / coefficient;
	row.freeVelocity = speed - target + settings.timestep * stiffness * penetration / coefficient;
	return true;
}

static void appendCompliantNormal(const PxContactPoint& contact, const PxVec3& normal, const PxVec3& angular0, const PxVec3& angular1, double penetration, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const AnvilJacobianBody* jacobianBodies, const AnvilContactSettings& settings, anvil::Problem& problem)
{
	anvil::CompactContact& row = problem.beginScalarContact();
	if(!prepareCompliantNormal(contact, normal, angular0, angular1, penetration, body0, body1, bodyIndex0, bodyIndex1, jacobianBodies, settings, row))
	{
		problem.cancelScalarContact();
		return;
	}
	const double upper = hasContactImpulseLimit(contact.maxImpulse) ? contact.maxImpulse : anvil::MAX_IMPULSE;
	problem.finishScalarContact(0.0, upper, rowNonzeroCount(row));
}

static void appendCompliantFriction(const PxContactPoint& contact, const PxVec3& normalAngular0, const PxVec3& normalAngular1, double penetration, const PxVec3& arm0, const PxVec3& arm1, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const AnvilJacobianBody* jacobianBodies, double friction, const AnvilContactSettings& settings, anvil::Problem& problem, PxVec3& tangent0, PxVec3& tangent1)
{
	anvil::CompactContact normal;
	if(!prepareCompliantNormal(contact, contact.normal, normalAngular0, normalAngular1, penetration, body0, body1, bodyIndex0, bodyIndex1, jacobianBodies, settings, normal))
	{
		return;
	}

	const PxVec3 pointVelocity = body0.linearVelocity + body0.angularVelocity.cross(arm0) -
		body1.linearVelocity - body1.angularVelocity.cross(arm1);
	contactTangents(contact.normal, pointVelocity, tangent0, tangent1);
	const PxVec3 tangents[2] = { tangent0, tangent1 };
	anvil::Contact block;
	block.body[0] = bodyIndex0;
	block.body[1] = bodyIndex1;
	block.jacobian[0].row(2) = normal.jacobian[0];
	block.jacobian[1].row(2) = normal.jacobian[1];
	block.freeVelocity[2] = normal.freeVelocity;
	block.regularization[2] = normal.regularization;
	double tangentResponse = 0.0;
	for(PxU32 tangent = 0; tangent < 2; ++tangent)
	{
		const PxVec3 angular0 = arm0.cross(tangents[tangent]);
		const PxVec3 angular1 = arm1.cross(tangents[tangent]);
		anvil::CompactContact rows;
		contactJacobians(rows, jacobianBodies, tangents[tangent], angular0, angular1);
		const anvil::Vec6& jacobian0 = rows.jacobian[0];
		const anvil::Vec6& jacobian1 = rows.jacobian[1];
		block.jacobian[0].row(tangent) = jacobian0;
		block.jacobian[1].row(tangent) = jacobian1;
		block.freeVelocity[tangent] = contactSpeed(body0, body1, tangents[tangent], angular0, angular1) - dotAnvilContact(contact.targetVel, tangents[tangent]);
		tangentResponse += jacobian0.squaredNorm() + jacobian1.squaredNorm();
	}
	const double tangentRegularization = std::max(1.0e-15, settings.regularization * tangentResponse * 0.5);
	block.regularization[0] = block.regularization[1] = tangentRegularization;
	block.friction = friction;
	block.maxNormalImpulse = hasContactImpulseLimit(contact.maxImpulse) ? contact.maxImpulse : anvil::MAX_IMPULSE;
	problem.addContact(block);
}

static void appendContactPoint(PxU32 firstContact, PxReal* forceDestination, PxReal friction, PxReal frictionLimit, const anvil::Problem& problem, AnvilContactRows& output)
{
	const AnvilContactPoint point = { firstContact, PxU32(problem.contacts.size()) - firstContact, forceDestination, friction, frictionLimit };
	output.points.pushBack(point);
}

static void appendAnvilContact(const PxContactPoint& contact, PxReal restDistance, PxReal ccdMaxSeparation, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const PxTransform& frame0, const PxTransform& frame1, const AnvilJacobianBody* jacobianBodies, double translationResponse, const AnvilContactSettings& settings, bool sliding, const PxVec3& slip, PxReal normalShare, PxReal pointFraction, PxReal* forceDestination, anvil::Problem& problem, AnvilContactRows& output)
{
	const PxVec3 arm0 = contact.point - frame0.p;
	const PxVec3 arm1 = contact.point - frame1.p;
	const PxVec3 normalAngular0 = arm0.cross(contact.normal);
	const PxVec3 normalAngular1 = arm1.cross(contact.normal);
	const double penetration = double(contact.separation) - restDistance;
	const PxU32 firstContact = PxU32(problem.contacts.size());
	const PxReal frictionCoefficient = sliding ? contact.dynamicFriction : contact.staticFriction;
	if(contact.restitution < 0.0f)
	{
		if(!(contact.materialFlags & PxMaterialFlag::eDISABLE_FRICTION) && frictionCoefficient > 0.0f)
		{
			PxVec3 tangent0, tangent1;
			appendCompliantFriction(contact, normalAngular0, normalAngular1, penetration, arm0, arm1, body0, body1, bodyIndex0, bodyIndex1, jacobianBodies, frictionCoefficient, settings, problem, tangent0, tangent1);
			appendContactPoint(firstContact, forceDestination, frictionCoefficient, 0.0f, problem, output);
			return;
		}
		appendCompliantNormal(contact, contact.normal, normalAngular0, normalAngular1, penetration, body0, body1, bodyIndex0, bodyIndex1, jacobianBodies, settings, problem);
		appendContactPoint(firstContact, forceDestination, 0.0f, 0.0f, problem, output);
		return;
	}
	const double ratio = settings.stiffeningDepth > 0.0 ? contactRegularization(settings, penetration) : double(settings.regularization);
	const double impedance = settings.stiffeningDepth > 0.0 ? 1.0 / (1.0 + ratio) : settings.impedance;
	const double damping = settings.damping;
	const double stiffness = settings.stiffness;
	const PxVec3 initialVelocity = initialPointVelocity(body0, body1, bodyIndex0, bodyIndex1, arm0, arm1, settings);
	const PxVec3 freePointVelocity = body0.linearVelocity + body0.angularVelocity.cross(arm0) -
		body1.linearVelocity - body1.angularVelocity.cross(arm1);
	const double initialNormalSpeed = dotAnvilContact(contact.normal, initialVelocity);
	const double penetrationSpeed = penetration / settings.timestep;
	const bool colliding = -initialNormalSpeed > penetrationSpeed;
	const bool bounce = contact.restitution > 0.0f && initialNormalSpeed < settings.bounceThreshold && colliding && penetration <= ccdMaxSeparation;
	const double normalTarget = dotAnvilContact(contact.targetVel, contact.normal) +
		(bounce ? -double(contact.restitution) * initialNormalSpeed : 0.0);
	const double freeNormalSpeed = dotAnvilContact(contact.normal, freePointVelocity);
	const double response = translationResponse > 0.0 ? translationResponse : 1.0;
	const double regularization = ratio * response;
	const double upper = hasContactImpulseLimit(contact.maxImpulse) ? contact.maxImpulse : anvil::MAX_IMPULSE;
	const bool friction = !hasContactImpulseLimit(contact.maxImpulse) && !(contact.materialFlags & PxMaterialFlag::eDISABLE_FRICTION) && frictionCoefficient > 0.0f;
	// A bounce leaves at the rebound speed within the step, as an impact does: relaxing toward it
	// would lose part of it, and position correction would add to or take from it by the depth or
	// gap at which the step finds the impact.
	// A separated point pushes only to keep the step from carrying it through the surface: its
	// gap may close within the step at any speed. The spring and damper act once the surfaces
	// touch; damping a separated point would slow a body before it collides.
	const bool separated = penetration > 0.0;
	const double targetNormalSpeed = bounce ? normalTarget : separated ? normalTarget - penetrationSpeed : initialNormalSpeed - settings.timestep *
		(damping * (initialNormalSpeed - normalTarget) + stiffness * impedance * penetration);
	appendContactRow(contact.normal, normalAngular0, normalAngular1, freeNormalSpeed - targetNormalSpeed,
		friction ? ANVIL_FRICTION_NORMAL_REGULARIZATION * regularization : regularization, bodyIndex0, bodyIndex1, jacobianBodies, problem, upper);
	if(!friction || PxU32(problem.contacts.size()) == firstContact)
	{
		appendContactPoint(firstContact, forceDestination, 0.0f, 0.0f, problem, output);
		return;
	}
	// Lagged Coulomb friction: tangent rows bounded by mu times the point's share of the pair's
	// previous normal impulse, independent of this step's normal row, so sliding contacts do not
	// separate. A point without a previous normal impulse takes the impulse that stops its approach.
	const double normalImpulse = normalShare > 0.0f ? double(normalShare) :
		std::max(0.0, targetNormalSpeed - freeNormalSpeed) / response * double(pointFraction);
	const double limit = double(frictionCoefficient) * normalImpulse;
	if(limit > 0.0)
	{
		// A sliding point's friction has magnitude mu N0 and opposes its slip, its velocity relative
		// to the surface's target. Its first row lies along the slip at the start of the step. A
		// slip s turned by a small velocity v across it turns the friction by v / |s|, so the
		// friction across the slip is mu N0 v / |s|: the second row has that compliance, |s| /
		// (mu N0), or the row compliance where that is smaller, as inside the friction limit. The
		// rows reduce to sticking's two rows as the slip vanishes. Two stiff rows would let the row
		// across the slip hold while the other slides, resisting motion that Coulomb friction does
		// not, such as the turn of a case sliding on belts at different speeds.
		PxVec3 tangents[2];
		double crossRegularization = regularization;
		const PxVec3 slipVelocity = initialVelocity - contact.targetVel;
		const PxVec3 tangentialSlip = slipVelocity - contact.normal * contact.normal.dot(slipVelocity);
		const PxReal slipSquared = tangentialSlip.magnitudeSquared();
		if(sliding && slipSquared > 0.0f)
		{
			const PxReal slipSpeed = PxSqrt(slipSquared);
			tangents[0] = tangentialSlip * (1.0f / slipSpeed);
			tangents[1] = contact.normal.cross(tangents[0]);
			crossRegularization = std::max(regularization, double(slipSpeed) / limit);
		}
		else
		{
			contactTangents(contact.normal, initialVelocity, tangents[0], tangents[1]);
		}
		const PxU32 normalEnd = PxU32(problem.contacts.size());
		for(PxU32 tangent = 0; tangent < 2; ++tangent)
		{
			const PxVec3& direction = tangents[tangent];
			const double freeVelocity = relaxedFreeVelocity(dotAnvilContact(direction, initialVelocity), dotAnvilContact(direction, freePointVelocity),
				dotAnvilContact(contact.targetVel, direction), ANVIL_SLIP_STIFFNESS * dotAnvilContact(direction, slip), stiffness, damping, impedance, settings);
			appendContactRow(direction, arm0.cross(direction), arm1.cross(direction), freeVelocity, tangent ? crossRegularization : regularization, bodyIndex0, bodyIndex1, jacobianBodies, problem, limit, -limit);
		}
		if(PxU32(problem.contacts.size()) == normalEnd + 2)
		{
			appendContactPoint(firstContact, forceDestination, frictionCoefficient, PxReal(limit), problem, output);
			return;
		}
		while(PxU32(problem.contacts.size()) > normalEnd)
		{
			problem.cancelScalarContact();
		}
	}
	appendContactPoint(firstContact, forceDestination, 0.0f, 0.0f, problem, output);
}

void prepareAnvilContacts(PxsContactManager& manager, PxsContactManagerOutput& contactOutput, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const AnvilContactSettings& settings, PxContactBuffer& buffer, FrictionPatchStreamPair* frictionStream, AnvilFrictionState* reservedState, anvil::Problem& problem, AnvilContactRows& output)
{
	PxcNpWorkUnit& unit = manager.getWorkUnit();
	if(contactOutput.contactForces)
	{
		PxMemZero(contactOutput.contactForces, sizeof(PxReal) * contactOutput.nbContacts);
	}
	PX_ASSERT(unit.getDominance0() && unit.getDominance1());
	const AnvilFrictionState* previous = unit.mFrictionPatchCount == ANVIL_FRICTION_STATE_MARKER ? reinterpret_cast<const AnvilFrictionState*>(unit.mFrictionDataPtr) : NULL;
	const bool sliding = previous && previous->twist == ANVIL_SLIDING_TWIST;
	PxU16 originalIndices[PxContactBuffer::MAX_CONTACTS];
	extractAnvilContacts(contactOutput, buffer, originalIndices, PxMin(body0.maxContactImpulse, body1.maxContactImpulse));
	if(buffer.count == 0)
	{
		unit.mFrictionDataPtr = NULL;
		unit.mFrictionPatchCount = 0;
		return;
	}
	// The stream packs records without padding; this is its only record, a multiple of 4 bytes.
	PX_COMPILE_TIME_ASSERT((sizeof(AnvilFrictionState) & 3) == 0);
	AnvilFrictionState* frictionState = frictionStream ? frictionStream->reserve<AnvilFrictionState>(sizeof(AnvilFrictionState)) : reservedState;
	if(frictionState == reinterpret_cast<AnvilFrictionState*>(-1))
	{
		frictionState = NULL;
	}
	unit.mFrictionDataPtr = reinterpret_cast<PxU8*>(frictionState);
	unit.mFrictionPatchCount = frictionState ? ANVIL_FRICTION_STATE_MARKER : 0;

	const PxTransform& frame0 = unit.mRigidCore0->body2World;
	const PxTransform identity(PxIdentity);
	const PxTransform& frame1 = unit.mRigidCore1 ? unit.mRigidCore1->body2World : identity;
	const PxU32 contactCount = buffer.count;
	// A sticking pair keeps the slip and twist accumulated up to the last step, which writeback
	// extended by that step's motion at the contact centre, in the plane of the current normal.
	// The slip belongs to the surfaces that were touching when it began. While a pair sticks, its
	// normal stays fixed in body 0's frame; when body 0 turns against body 1, as a hull tipping onto
	// another face does, the contact moves to other surfaces, and the slip lapses once that turn
	// moves body 0's material at the contact by more than the slip limit.
	AnvilContactPair pair;
	pair.state = frictionState;
	PxVec3 slip(0.0f), centre(0.0f);
	PxReal twist = 0.0f;
	const PxVec3& normal = buffer.contacts[0].normal;
	if(frictionState)
	{
		for(PxU32 i = 0; i < contactCount; ++i)
		{
			centre += buffer.contacts[i].point;
		}
		centre *= 1.0f / PxReal(contactCount);
		pair.arm[0] = centre - frame0.p;
		pair.arm[1] = centre - frame1.p;
		pair.normal = normal;
		pair.freeSlipVelocity = body0.linearVelocity + body0.angularVelocity.cross(pair.arm[0]) - body1.linearVelocity - body1.angularVelocity.cross(pair.arm[1]) - buffer.contacts[0].targetVel;
		pair.freeTwistVelocity = (body0.angularVelocity - body1.angularVelocity).dot(normal);
		PxVec3 localNormal = frame0.q.rotateInv(normal);
		if(previous && !sliding)
		{
			slip = PxVec3(previous->slip[0], previous->slip[1], previous->slip[2]);
			slip -= normal * normal.dot(slip);
			twist = previous->twist;
			PxReal radius = 0.0f;
			for(PxU32 i = 0; i < contactCount; ++i)
			{
				radius = PxMax(radius, (buffer.contacts[i].point - centre).magnitudeSquared());
			}
			const PxReal limit = settings.slipLimit;
			// The normal's turn in body 0's frame since the slip began, times body 0's arm to the contact.
			const PxVec3 anchorNormal(previous->normal0[0], previous->normal0[1], previous->normal0[2]);
			const PxReal turn = (localNormal - anchorNormal).magnitudeSquared() * pair.arm[0].magnitudeSquared();
			if(slip.magnitudeSquared() > limit * limit || twist * twist * radius > limit * limit || turn > limit * limit)
			{
				slip = PxVec3(0.0f);
				twist = 0.0f;
			}
			else
			{
				localNormal = anchorNormal;
			}
		}
		frictionState->normal0[0] = localNormal.x;
		frictionState->normal0[1] = localNormal.y;
		frictionState->normal0[2] = localNormal.z;
		frictionState->slip[0] = slip.x;
		frictionState->slip[1] = slip.y;
		frictionState->slip[2] = slip.z;
		frictionState->twist = twist;
		frictionState->normalImpulse = 0.0f;
	}
	// Points whose gap is unlikely to close in this step are deferred: the solve decides.
	bool keep[PxContactBuffer::MAX_CONTACTS];
	PxU32 keptCount = 0;
	pair.deferredLimit = -PX_MAX_F32;
	pair.deferredReach[0] = pair.deferredReach[1] = 0.0f;
	pair.deferredNormal = PxVec3(0.0f);
	bool deferredNormalsAgree = true;
	for(PxU32 i = 0; i < contactCount; ++i)
	{
		const PxContactPoint& contact = buffer.contacts[i];
		const PxReal gap = contact.separation - unit.mRestDistance;
		const bool kept = speculativePointKept(contact, gap, body0, body1, bodyIndex0, bodyIndex1, frame0, frame1, settings);
		keep[i] = kept;
		keptCount += kept;
		if(!kept)
		{
			if(pair.deferredLimit == -PX_MAX_F32)
			{
				pair.deferredNormal = contact.normal;
			}
			deferredNormalsAgree = deferredNormalsAgree && contact.normal == pair.deferredNormal;
			pair.deferredLimit = PxMax(pair.deferredLimit, contact.normal.dot(contact.targetVel) - gap / settings.timestep);
			pair.deferredReach[0] = PxMax(pair.deferredReach[0], (contact.point - frame0.p).magnitudeSquared());
			pair.deferredReach[1] = PxMax(pair.deferredReach[1], (contact.point - frame1.p).magnitudeSquared());
		}
	}
	if(!deferredNormalsAgree)
	{
		pair.deferredNormal = PxVec3(0.0f);
	}
	pair.deferredReach[0] = PxSqrt(pair.deferredReach[0]);
	pair.deferredReach[1] = PxSqrt(pair.deferredReach[1]);
	pair.manager = &manager;
	pair.bodyData[0] = &body0;
	pair.bodyData[1] = &body1;
	pair.body[0] = bodyIndex0;
	pair.body[1] = bodyIndex1;
	// Friction takes the previous normal impulse, shared equally by the pair's kept points.
	const PxReal normalShare = previous && frictionState && keptCount ? previous->normalImpulse / PxReal(keptCount) : 0.0f;
	pair.firstPoint = output.points.size();
	pair.reportThreshold = (unit.mFlags & PxcNpWorkUnitFlag::eFORCE_THRESHOLD) && (body0.reportThreshold < PX_MAX_F32 || body1.reportThreshold < PX_MAX_F32);
	pair.threshold.shapeInteraction = reinterpret_cast<Sc::ShapeInteraction*>(manager.getShapeInteraction());
	pair.threshold.threshold = PxMin(body0.reportThreshold, body1.reportThreshold);
	pair.threshold.nodeIndexA = PxNodeIndex(body0.nodeIndex);
	pair.threshold.nodeIndexB = PxNodeIndex(body1.nodeIndex);
	PxOrder(pair.threshold.nodeIndexA, pair.threshold.nodeIndexB);
	pair.threshold.normalForce = pair.threshold.accumulatedForce = 0.0f;
	pair.threshold.pad = 0;
	const PxU32 bodyFlags = PxU32(unit.mRigidCore0->mFlags) | (unit.mRigidCore1 ? PxU32(unit.mRigidCore1->mFlags) : 0);
	const PxReal ccdMaxSeparation = bodyFlags & PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD ? settings.ccdMaxSeparation : PX_MAX_F32;
	const double rootInverseMass0 = bodyIndex0 >= 0 ? std::sqrt(double(body0.invMass)) : 0.0;
	const double rootInverseMass1 = bodyIndex1 >= 0 ? std::sqrt(double(body1.invMass)) : 0.0;
	const double translationResponse = rootInverseMass0 * rootInverseMass0 + rootInverseMass1 * rootInverseMass1;
	AnvilJacobianBody jacobianBodies[2];
	prepareJacobianBody(jacobianBodies[0], body0, bodyIndex0, rootInverseMass0, 1.0, settings.bodyLockFlags);
	prepareJacobianBody(jacobianBodies[1], body1, bodyIndex1, rootInverseMass1, -1.0, settings.bodyLockFlags);
	for(PxU32 i = 0; i < contactCount; ++i)
	{
		if(!keep[i])
		{
			continue;
		}
		// Each point's slip adds the twist's displacement about the centre.
		const PxVec3 pointSlip = twist != 0.0f ? slip + normal.cross(buffer.contacts[i].point - centre) * twist : slip;
		appendAnvilContact(buffer.contacts[i], unit.mRestDistance, ccdMaxSeparation, body0, body1, bodyIndex0, bodyIndex1, frame0, frame1, jacobianBodies, translationResponse, settings, sliding, pointSlip, normalShare, 1.0f / PxReal(keptCount), contactOutput.contactForces ? contactOutput.contactForces + originalIndices[i] : NULL, problem, output);
	}
	pair.pointCount = output.points.size() - pair.firstPoint;
	output.pairs.pushBack(pair);
}

// A body's velocity after the solve: its free velocity plus the solve's correction, as
// integration applies them.
static AnvilBodyVelocity solvedBodyVelocity(const PxSolverBodyData& body, PxU32 bodyIndex, const anvil::Result& result)
{
	const PxU32 offset = 6 * bodyIndex;
	const PxReal rootInverseMass = PxSqrt(body.invMass);
	const PxVec3 linearCorrection(PxReal(result.primal[offset]), PxReal(result.primal[offset + 1]), PxReal(result.primal[offset + 2]));
	const PxVec3 angularState(PxReal(result.primal[offset + 3]), PxReal(result.primal[offset + 4]), PxReal(result.primal[offset + 5]));
	const AnvilBodyVelocity velocity = { body.linearVelocity + linearCorrection * rootInverseMass, body.angularVelocity + body.sqrtInvInertia * angularState };
	return velocity;
}

// Whether contact of pair joined the problem after an earlier solve.
static bool isLateContact(const AnvilContactRows& rows, PxU32 pair, PxU32 contact)
{
	const PxU32 lateCount = rows.latePairs.size();
	for(PxU32 i = 0; i < lateCount; ++i)
	{
		if(rows.latePairs[i] == pair && rows.lateContacts[i] == contact)
		{
			return true;
		}
	}
	return false;
}

bool addClosedAnvilContacts(AnvilContactRows& rows, anvil::Problem& problem, const anvil::Result& result, const AnvilContactSettings& settings, const PxSolverBodyData* bodyData, PxU32 bodyCount, PxContactBuffer& buffer, DynamicsContext& context)
{
	const PxReal timestep = settings.timestep;
	rows.solvedVelocities.resize(bodyCount);
	for(PxU32 i = 0; i < bodyCount; ++i)
	{
		rows.solvedVelocities[i] = solvedBodyVelocity(bodyData[i], i, result);
	}
	PxU16 originalIndices[PxContactBuffer::MAX_CONTACTS];
	bool added = false;
	const PxU32 pairCount = rows.pairs.size();
	for(PxU32 pairIndex = 0; pairIndex < pairCount; ++pairIndex)
	{
		const AnvilContactPair& pair = rows.pairs[pairIndex];
		if(pair.deferredLimit == -PX_MAX_F32)
		{
			continue;
		}
		// Bodies outside the island keep their velocity.
		AnvilBodyVelocity velocity[2];
		for(PxU32 end = 0; end < 2; ++end)
		{
			if(pair.body[end] >= 0)
			{
				velocity[end] = rows.solvedVelocities[PxU32(pair.body[end])];
			}
			else
			{
				velocity[end].linear = pair.bodyData[end]->linearVelocity;
				velocity[end].angular = pair.bodyData[end]->angularVelocity;
			}
		}
		const PxVec3 relative = velocity[0].linear - velocity[1].linear;
		const PxReal least = (pair.deferredNormal.isZero() ? -relative.magnitude() : pair.deferredNormal.dot(relative)) -
			velocity[0].angular.magnitude() * pair.deferredReach[0] - velocity[1].angular.magnitude() * pair.deferredReach[1];
		if(least >= pair.deferredLimit)
		{
			continue;
		}
		// The bodies move fast enough to close a deferred point: test each exactly.
		PxcNpWorkUnit& unit = pair.manager->getWorkUnit();
		const PxSolverBodyData& body0 = *pair.bodyData[0];
		const PxSolverBodyData& body1 = *pair.bodyData[1];
		PxsContactManagerOutput& contactOutput = context.mOutputIterator.getContactManagerOutput(unit.mNpIndex);
		extractAnvilContacts(contactOutput, buffer, originalIndices, PxMin(body0.maxContactImpulse, body1.maxContactImpulse));
		const PxTransform& frame0 = unit.mRigidCore0->body2World;
		const PxTransform identity(PxIdentity);
		const PxTransform& frame1 = unit.mRigidCore1 ? unit.mRigidCore1->body2World : identity;
		const PxU32 bodyFlags = PxU32(unit.mRigidCore0->mFlags) | (unit.mRigidCore1 ? PxU32(unit.mRigidCore1->mFlags) : 0);
		const PxReal ccdMaxSeparation = bodyFlags & PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD ? settings.ccdMaxSeparation : PX_MAX_F32;
		const double rootInverseMass0 = pair.body[0] >= 0 ? std::sqrt(double(body0.invMass)) : 0.0;
		const double rootInverseMass1 = pair.body[1] >= 0 ? std::sqrt(double(body1.invMass)) : 0.0;
		const double translationResponse = rootInverseMass0 * rootInverseMass0 + rootInverseMass1 * rootInverseMass1;
		AnvilJacobianBody jacobianBodies[2];
		bool prepared = false;
		const PxU32 contactCount = buffer.count;
		for(PxU32 i = 0; i < contactCount; ++i)
		{
			PxContactPoint& contact = buffer.contacts[i];
			const PxReal gap = contact.separation - unit.mRestDistance;
			if(speculativePointKept(contact, gap, body0, body1, pair.body[0], pair.body[1], frame0, frame1, settings) || isLateContact(rows, pairIndex, i))
			{
				continue;
			}
			// The step closes the gap when the solved velocity leaves the surface's target by more
			// than the gap per step: the bound a separated point's row enforces once it is added.
			const PxVec3 pointVelocity = velocity[0].linear + velocity[0].angular.cross(contact.point - frame0.p) - velocity[1].linear - velocity[1].angular.cross(contact.point - frame1.p);
			if(contact.normal.dot(pointVelocity) >= contact.normal.dot(contact.targetVel) - gap / timestep)
			{
				continue;
			}
			// The point joins without friction: lagged friction bounds a point by its share of the
			// pair's normal impulse in the last step, and this point had none. Its normal impulse
			// joins the pair's, which bounds the pair's friction from the next step. (Estimating the
			// impulse that would stop it from the translational response alone overstates it many
			// times for a turning body, and lets friction act at a point that carries almost no load.)
			if(!prepared)
			{
				prepareJacobianBody(jacobianBodies[0], body0, pair.body[0], rootInverseMass0, 1.0, settings.bodyLockFlags);
				prepareJacobianBody(jacobianBodies[1], body1, pair.body[1], rootInverseMass1, -1.0, settings.bodyLockFlags);
				prepared = true;
			}
			if(rows.latePairs.empty())
			{
				rows.lateBegin = rows.points.size();
			}
			contact.materialFlags |= PxMaterialFlag::eDISABLE_FRICTION;
			appendAnvilContact(contact, unit.mRestDistance, ccdMaxSeparation, body0, body1, pair.body[0], pair.body[1], frame0, frame1, jacobianBodies, translationResponse, settings,
				false, PxVec3(0.0f), 0.0f, 1.0f, contactOutput.contactForces ? contactOutput.contactForces + originalIndices[i] : NULL, problem, rows);
			while(rows.latePairs.size() < rows.points.size() - rows.lateBegin)
			{
				rows.latePairs.pushBack(pairIndex);
				rows.lateContacts.pushBack(PxU16(i));
			}
			added = true;
		}
	}
	return added;
}

static double normalImpulse(const AnvilContactPoint& point, const anvil::Problem& problem, const anvil::Result& result)
{
	// A point is one three-row block, whose last row is the normal, or a normal row followed by
	// its bounded friction rows.
	if(!point.contactCount)
	{
		return 0.0;
	}
	const anvil::CompactContact& contact = problem.contacts[point.firstContact];
	return result.impulse[contact.rowCount() == 3 ? contact.row + 2 : contact.row];
}

struct AnvilPointFriction
{
	enum Enum
	{
		eUNLOADED,
		eHOLDING,
		eSATURATED
	};
};

// A point's friction is saturated when it reaches its limit. Soft friction rows let a holding
// contact creep in proportion to its load, so slip speed cannot separate the cases. A point
// without friction rows carries no friction load.
static AnvilPointFriction::Enum pointFriction(const AnvilContactPoint& point, const anvil::Problem& problem, const anvil::Result& result)
{
	if(point.friction == 0.0f || point.contactCount == 0)
	{
		return AnvilPointFriction::eUNLOADED;
	}
	const anvil::CompactContact& contact = problem.contacts[point.firstContact];
	if(point.contactCount == 1 && contact.rowCount() == 3)
	{
		// Three-row blocks clamp each tangent impulse to friction times the normal impulse.
		const double normal = result.impulse[contact.row + 2];
		if(normal <= 0.0)
		{
			return AnvilPointFriction::eUNLOADED;
		}
		const double limit = double(point.friction) * normal * (1.0 - 1.0e-9);
		return std::abs(result.impulse[contact.row]) >= limit || std::abs(result.impulse[contact.row + 1]) >= limit ?
			AnvilPointFriction::eSATURATED : AnvilPointFriction::eHOLDING;
	}
	PX_ASSERT(point.contactCount == 3 && point.frictionLimit > 0.0f);
	// A normal row and two friction rows bounded by the lagged limit. The limit is stored in
	// single precision, so the comparison allows its rounding.
	const double friction0 = result.impulse[problem.contacts[point.firstContact + 1].row];
	const double friction1 = result.impulse[problem.contacts[point.firstContact + 2].row];
	if(result.impulse[contact.row] <= 0.0 && friction0 == 0.0 && friction1 == 0.0)
	{
		return AnvilPointFriction::eUNLOADED;
	}
	const double limit = double(point.frictionLimit) * (1.0 - 1.0e-6);
	return std::abs(friction0) >= limit || std::abs(friction1) >= limit ? AnvilPointFriction::eSATURATED : AnvilPointFriction::eHOLDING;
}

int gAnvilDebugStep = 0, gAnvilDebugFrom = -1, gAnvilDebugTo = -2; // EXPERIMENT (temporary)

void writebackAnvilContacts(const AnvilContactRows& rows, const anvil::Problem& problem, const anvil::Result& result, const PxSolverBody* bodies, const PxSolverBodyData* bodyData, DynamicsContext& context)
{
	if(gAnvilDebugStep >= gAnvilDebugFrom && gAnvilDebugStep <= gAnvilDebugTo) // EXPERIMENT (temporary): top-level pairs' friction
	{
		for(PxU32 i = 0; i < rows.pairs.size(); ++i)
		{
			const AnvilContactPair& pair = rows.pairs[i];
			const PxVec3 centre = pair.manager->getWorkUnit().mRigidCore0->body2World.p + pair.arm[0];
			if(centre.y < 0.30f || !pair.state)
				continue;
			printf("  [%d] pair %u bodies %d,%d centre (%.1f,%.1f,%.1f) mm normal (%.3f,%.3f,%.3f) slip (%.2f,%.2f,%.2f) um twist %g N0 %.1f uNs\n", gAnvilDebugStep, i, pair.body[0], pair.body[1], centre.x * 1e3f, centre.y * 1e3f, centre.z * 1e3f,
				pair.normal.x, pair.normal.y, pair.normal.z, pair.state->slip[0] * 1e6f, pair.state->slip[1] * 1e6f, pair.state->slip[2] * 1e6f, double(pair.state->twist), pair.state->normalImpulse * 1e6f);
			for(PxU32 j = pair.firstPoint; j < pair.firstPoint + pair.pointCount; ++j)
			{
				const AnvilContactPoint& point = rows.points[j];
				const double n = normalImpulse(point, problem, result);
				const double f0 = point.contactCount == 3 ? result.impulse[problem.contacts[point.firstContact + 1].row] : 0.0;
				const double f1 = point.contactCount == 3 ? result.impulse[problem.contacts[point.firstContact + 2].row] : 0.0;
				printf("      point: normal %8.2f friction (%8.2f,%8.2f) limit %8.2f uNs -> %s\n", n * 1e6, f0 * 1e6, f1 * 1e6, double(point.frictionLimit) * 1e6,
					pointFriction(point, problem, result) == AnvilPointFriction::eSATURATED ? "SATURATED" : pointFriction(point, problem, result) == AnvilPointFriction::eHOLDING ? "holding" : "unloaded");
			}
		}
	}
	// A sticking pair's slip grows by the step's tangential motion at the contact centre: the
	// free relative velocity plus the solve's corrections, as body integration applies them.
	// Slip does not relax, so a held load stays where friction caught it; it clears when the pair
	// slides or unloads, or beyond the slip limit.
	const PxReal timestep = context.getDt();
	const PxU32 pairCount = rows.pairs.size();
	for(PxU32 i = 0; i < pairCount; ++i)
	{
		const AnvilContactPair& pair = rows.pairs[i];
		AnvilFrictionState* state = pair.state;
		if(!state || state->twist == ANVIL_SLIDING_TWIST)
		{
			continue;
		}
		// The pair's total normal impulse bounds its friction in the next step.
		double pairNormal = 0.0;
		const PxU32 lastPoint = pair.firstPoint + pair.pointCount;
		for(PxU32 j = pair.firstPoint; j < lastPoint; ++j)
		{
			pairNormal += normalImpulse(rows.points[j], problem, result);
		}
		state->normalImpulse = PxReal(pairNormal);
		// A rigid contact slides only when every loaded point is saturated; one saturated corner
		// does not move a body the others still hold.
		bool loaded = false, saturated = true;
		for(PxU32 j = pair.firstPoint; j < lastPoint && saturated; ++j)
		{
			const AnvilPointFriction::Enum friction = pointFriction(rows.points[j], problem, result);
			loaded = loaded || friction != AnvilPointFriction::eUNLOADED;
			saturated = friction != AnvilPointFriction::eHOLDING;
		}
		if(!loaded)
		{
			// Friction that carries no load holds no slip.
			state->slip[0] = state->slip[1] = state->slip[2] = 0.0f;
			state->twist = 0.0f;
			continue;
		}
		if(saturated)
		{
			state->twist = ANVIL_SLIDING_TWIST;
			continue;
		}
		PxVec3 velocity = pair.freeSlipVelocity;
		PxReal twistVelocity = pair.freeTwistVelocity;
		for(PxU32 end = 0; end < 2; ++end)
		{
			const PxI32 body = pair.body[end];
			if(body < 0)
			{
				continue;
			}
			const PxVec3 angular = bodyData[body].sqrtInvInertia * bodies[body].angularState;
			const PxVec3 linear = bodies[body].linearVelocity + angular.cross(pair.arm[end]);
			velocity += end ? -linear : linear;
			twistVelocity += end ? -angular.dot(pair.normal) : angular.dot(pair.normal);
		}
		velocity -= pair.normal * pair.normal.dot(velocity);
		state->slip[0] += velocity.x * timestep;
		state->slip[1] += velocity.y * timestep;
		state->slip[2] += velocity.z * timestep;
		state->twist += twistVelocity * timestep;
	}
	// Points that joined after a solve add their normal impulse to their pair's.
	const PxU32 lateCount = rows.latePairs.size();
	for(PxU32 i = 0; i < lateCount; ++i)
	{
		AnvilFrictionState* state = rows.pairs[rows.latePairs[i]].state;
		if(state)
		{
			state->normalImpulse += PxReal(normalImpulse(rows.points[rows.lateBegin + i], problem, result));
		}
	}
	const PxU32 pointCount = rows.points.size();
	for(PxU32 i = 0; i < pointCount; ++i)
	{
		const AnvilContactPoint& point = rows.points[i];
		if(point.destination)
		{
			*point.destination = PxReal(normalImpulse(point, problem, result));
		}
	}
	for(PxU32 i = 0; i < pairCount; ++i)
	{
		const AnvilContactPair& pair = rows.pairs[i];
		if(!pair.reportThreshold)
		{
			continue;
		}
		ThresholdStreamElement element = pair.threshold;
		const PxU32 lastPoint = pair.firstPoint + pair.pointCount;
		for(PxU32 j = pair.firstPoint; j < lastPoint; ++j)
		{
			element.normalForce += PxReal(normalImpulse(rows.points[j], problem, result));
		}
		for(PxU32 j = 0; j < lateCount; ++j)
		{
			if(rows.latePairs[j] == i)
			{
				element.normalForce += PxReal(normalImpulse(rows.points[rows.lateBegin + j], problem, result));
			}
		}
		if(element.normalForce != 0.0f)
		{
			const PxU32 index = PxU32(PxAtomicIncrement(&context.mThresholdStreamOut) - 1);
			PX_ASSERT(index < context.getThresholdStream().size());
			context.getThresholdStream()[index] = element;
		}
	}
}
}
}
