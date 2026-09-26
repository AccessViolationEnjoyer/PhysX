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

#include "DyNewtonContactPrep.h"
#include "DyDynamics.h"
#include "DyThreadContext.h"
#include "PxsContactManager.h"
#include "PxsRigidBody.h"
#include "CmSpatialVector.h"
#include "foundation/PxAtomic.h"
#include "core/NewtonSolver.h"

#include <cmath>

namespace physx
{
namespace Dy
{
// A native friction-patch count cannot reach this value. Newton uses it to identify
// the one-byte static/sliding state stored in PhysX's existing friction stream.
static const PxU8 NEWTON_FRICTION_STATE_MARKER = 0xff;

static PX_FORCE_INLINE double dotNewtonContact(const PxVec3& a, const PxVec3& b)
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
	return dotNewtonContact(direction, linear0) - dotNewtonContact(direction, linear1) +
		dotNewtonContact(angular0, angularVelocity0) - dotNewtonContact(angular1, angularVelocity1);
}

static PX_FORCE_INLINE double contactSpeed(const PxSolverBodyData& body0, const PxSolverBodyData& body1, const PxVec3& direction, const PxVec3& angular0, const PxVec3& angular1)
{
	return contactSpeed(body0.linearVelocity, body0.angularVelocity, body1.linearVelocity, body1.angularVelocity, direction, angular0, angular1);
}

// Jacobian inputs of one pair body, converted once for every row of the pair.
struct NewtonJacobianBody
{
	double inertia[3][3];
	double linearScale;
	double sign;
	PxU8 locks;
	bool present;
};

static PX_FORCE_INLINE void prepareJacobianBody(NewtonJacobianBody& result, const PxSolverBodyData& body, PxI32 bodyIndex, double rootInverseMass, double sign, const PxU8* lockFlags)
{
	result.present = bodyIndex >= 0;
	result.sign = sign;
	result.linearScale = sign * rootInverseMass;
	result.locks = lockFlags && bodyIndex >= 0 ? lockFlags[bodyIndex] : 0;
	const PxMat33& inertia = body.sqrtInvInertia;
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		result.inertia[axis][0] = double(inertia.column0[axis]);
		result.inertia[axis][1] = double(inertia.column1[axis]);
		result.inertia[axis][2] = double(inertia.column2[axis]);
	}
}

static PX_FORCE_INLINE void contactJacobian(newton::Vec6& jacobian, const NewtonJacobianBody& body, const PxVec3& direction, const PxVec3& angular)
{
	if(!body.present)
	{
		jacobian.setZero();
		return;
	}
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		jacobian[axis] = body.locks & (1 << axis) ? 0.0 : body.linearScale * direction[axis];
		jacobian[axis + 3] = body.sign * (body.inertia[axis][0] * angular.x + body.inertia[axis][1] * angular.y + body.inertia[axis][2] * angular.z);
	}
}

static PX_FORCE_INLINE double prepareContactJacobian(newton::CompactContact& row, const NewtonJacobianBody* jacobianBodies, PxI32 bodyIndex0, PxI32 bodyIndex1, const PxVec3& direction, const PxVec3& angular0, const PxVec3& angular1)
{
	row.body[0] = bodyIndex0;
	row.body[1] = bodyIndex1;
	contactJacobian(row.jacobian[0], jacobianBodies[0], direction, angular0);
	contactJacobian(row.jacobian[1], jacobianBodies[1], direction, angular1);
	return row.jacobian[0].squaredNorm() + row.jacobian[1].squaredNorm();
}

static PX_FORCE_INLINE void contactTangents(const PxVec3& normal, const PxVec3& relativeVelocity, PxVec3& tangent0, PxVec3& tangent1)
{
	tangent0 = relativeVelocity - normal * normal.dot(relativeVelocity);
	if(tangent0.magnitudeSquared() <= 0.0001f)
	{
		tangent0 = PxAbs(normal.x) < 0.70710678f ? PxVec3(0.0f, -normal.z, normal.y) : PxVec3(-normal.y, normal.x, 0.0f);
	}
	tangent0.normalize();
	tangent1 = normal.cross(tangent0);
}

// Like MuJoCo's position-dependent impedance, a contact stiffens smoothly from its surface
// impedance to the regular one as penetration reaches the stiffening depth. Damping and
// stiffness keep the regular impedance; the impedance scales the position term and sets R.
static PX_FORCE_INLINE double contactImpedance(const NewtonContactSettings& settings, double penetration)
{
	const double depth = PxClamp(-penetration / settings.stiffeningDepth, 0.0, 1.0);
	const double step = depth <= 0.5 ? 2.0 * depth * depth : 1.0 - 2.0 * (1.0 - depth) * (1.0 - depth);
	return settings.surfaceImpedance + step * (settings.impedance - settings.surfaceImpedance);
}

static PX_FORCE_INLINE PxVec3 initialPointVelocity(const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const PxVec3& arm0, const PxVec3& arm1, const NewtonContactSettings& settings)
{
	const PxVec3& linear0 = bodyIndex0 >= 0 ? settings.initialVelocities[bodyIndex0].linear : body0.linearVelocity;
	const PxVec3& angular0 = bodyIndex0 >= 0 ? settings.initialVelocities[bodyIndex0].angular : body0.angularVelocity;
	const PxVec3& linear1 = bodyIndex1 >= 0 ? settings.initialVelocities[bodyIndex1].linear : body1.linearVelocity;
	const PxVec3& angular1 = bodyIndex1 >= 0 ? settings.initialVelocities[bodyIndex1].angular : body1.angularVelocity;
	return linear0 + angular0.cross(arm0) - linear1 - angular1.cross(arm1);
}

static void extractNewtonContacts(PxsContactManagerOutput& contactOutput, PxContactBuffer& buffer, PxU16* originalIndices, PxReal defaultMaxImpulse, bool& separateFrictionCoefficients)
{
	buffer.count = 0;
	separateFrictionCoefficients = false;
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
		separateFrictionCoefficients = separateFrictionCoefficients || iterator.getStaticFriction() != iterator.getDynamicFriction();
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

static PX_FORCE_INLINE void appendContactRow(const PxVec3& direction, const PxVec3& angular0, const PxVec3& angular1, double initialSpeed, double freeSpeed, double targetSpeed, double positionError, double stiffness, double damping, double impedance, double regularization, PxI32 bodyIndex0, PxI32 bodyIndex1, const NewtonJacobianBody* jacobianBodies, const NewtonContactSettings& settings, newton::Problem& problem, double upperImpulse)
{
	newton::CompactContact& row = problem.beginScalarContact();
	if(prepareContactJacobian(row, jacobianBodies, bodyIndex0, bodyIndex1, direction, angular0, angular1) == 0.0)
	{
		problem.cancelScalarContact();
		return;
	}
	row.freeVelocity = freeSpeed - initialSpeed + settings.timestep *
		(damping * (initialSpeed - targetSpeed) + stiffness * impedance * positionError);
	row.regularization = regularization;
	problem.finishScalarContact(0.0, upperImpulse);
}

static PX_FORCE_INLINE void appendEdgeRow(const newton::Vec6* normal, const newton::Vec6* tangent, double scale, PxI32 bodyIndex0, PxI32 bodyIndex1, double freeVelocity, double regularization, newton::Problem& problem)
{
	newton::CompactContact& row = problem.beginScalarContact();
	row.body[0] = bodyIndex0;
	row.body[1] = bodyIndex1;
	// Only an all-zero row is degenerate. Testing each entry avoids a serial sum of squares.
	bool nonzero = false;
	for(PxU32 end = 0; end < 2; ++end)
	{
		if((end ? bodyIndex1 : bodyIndex0) < 0)
		{
			row.jacobian[end].setZero();
			continue;
		}
		for(PxU32 column = 0; column < 6; ++column)
		{
			const double value = normal[end][column] + scale * tangent[end][column];
			row.jacobian[end][column] = value;
			nonzero = nonzero | (value != 0.0);
		}
	}
	if(!nonzero)
	{
		problem.cancelScalarContact();
		return;
	}
	row.freeVelocity = freeVelocity;
	row.regularization = regularization;
	problem.finishScalarContact(0.0, newton::MAX_IMPULSE);
}

static bool prepareCompliantNormal(const PxContactPoint& contact, const PxVec3& normal, const PxVec3& angular0, const PxVec3& angular1, double penetration, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const NewtonJacobianBody* jacobianBodies, const NewtonContactSettings& settings, newton::CompactContact& row)
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
	const double target = dotNewtonContact(contact.targetVel, normal);
	row.regularization = (contact.materialFlags & PxMaterialFlag::eCOMPLIANT_ACCELERATION_SPRING) ? response / coefficient : 1.0 / coefficient;
	row.freeVelocity = speed - target + settings.timestep * stiffness * penetration / coefficient;
	return true;
}

static void appendCompliantNormal(const PxContactPoint& contact, const PxVec3& normal, const PxVec3& angular0, const PxVec3& angular1, double penetration, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const NewtonJacobianBody* jacobianBodies, const NewtonContactSettings& settings, newton::Problem& problem)
{
	newton::CompactContact& row = problem.beginScalarContact();
	if(!prepareCompliantNormal(contact, normal, angular0, angular1, penetration, body0, body1, bodyIndex0, bodyIndex1, jacobianBodies, settings, row))
	{
		problem.cancelScalarContact();
		return;
	}
	const double upper = hasContactImpulseLimit(contact.maxImpulse) ? contact.maxImpulse : newton::MAX_IMPULSE;
	problem.finishScalarContact(0.0, upper);
}

static void appendCompliantFriction(const PxContactPoint& contact, const PxVec3& normalAngular0, const PxVec3& normalAngular1, double penetration, const PxVec3& arm0, const PxVec3& arm1, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const NewtonJacobianBody* jacobianBodies, double friction, const NewtonContactSettings& settings, newton::Problem& problem, PxVec3& tangent0, PxVec3& tangent1)
{
	newton::CompactContact normal;
	if(!prepareCompliantNormal(contact, contact.normal, normalAngular0, normalAngular1, penetration, body0, body1, bodyIndex0, bodyIndex1, jacobianBodies, settings, normal))
	{
		return;
	}

	const PxVec3 pointVelocity = body0.linearVelocity + body0.angularVelocity.cross(arm0) -
		body1.linearVelocity - body1.angularVelocity.cross(arm1);
	contactTangents(contact.normal, pointVelocity, tangent0, tangent1);
	const PxVec3 tangents[2] = { tangent0, tangent1 };
	newton::Contact block;
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
		newton::Vec6 jacobian0, jacobian1;
		contactJacobian(jacobian0, jacobianBodies[0], tangents[tangent], angular0);
		contactJacobian(jacobian1, jacobianBodies[1], tangents[tangent], angular1);
		block.jacobian[0].row(tangent) = jacobian0;
		block.jacobian[1].row(tangent) = jacobian1;
		block.freeVelocity[tangent] = contactSpeed(body0, body1, tangents[tangent], angular0, angular1) - dotNewtonContact(contact.targetVel, tangents[tangent]);
		tangentResponse += jacobian0.squaredNorm() + jacobian1.squaredNorm();
	}
	const double tangentRegularization = std::max(1.0e-15, settings.regularization * tangentResponse * 0.5);
	block.regularization[0] = block.regularization[1] = tangentRegularization;
	block.friction = friction;
	block.maxNormalImpulse = hasContactImpulseLimit(contact.maxImpulse) ? contact.maxImpulse : newton::MAX_IMPULSE;
	problem.addContact(block);
}

static void appendContactPoint(PxU32 firstContact, PxReal* forceDestination, PxU8* frictionState, PxReal friction, double freeNormalVelocity, double targetNormalVelocity, double freeTangentVelocity0, double freeTangentVelocity1, bool correctDilatancy, newton::Problem& problem, NewtonContactRows& output)
{
	const NewtonContactPoint point = { firstContact, PxU32(problem.contacts.size()) - firstContact, forceDestination };
	const PxU32 pointIndex = output.points.size();
	output.points.pushBack(point);
	if((frictionState || correctDilatancy) && point.contactCount)
	{
		const NewtonFrictionPoint frictionPoint = { pointIndex, friction, PxReal(freeNormalVelocity), PxReal(targetNormalVelocity), PxReal(freeTangentVelocity0), PxReal(freeTangentVelocity1), 0.0f, frictionState, correctDilatancy };
		output.frictionPoints.pushBack(frictionPoint);
	}
}

static void appendNewtonContact(const PxContactPoint& contact, PxReal restDistance, PxReal ccdMaxSeparation, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const PxTransform& frame0, const PxTransform& frame1, const NewtonJacobianBody* jacobianBodies, double translationResponse, const NewtonContactSettings& settings, bool sliding, PxU8* frictionState, PxReal* forceDestination, newton::Problem& problem, NewtonContactRows& output)
{
	const PxVec3 arm0 = contact.point - frame0.p;
	const PxVec3 arm1 = contact.point - frame1.p;
	const PxVec3 normalAngular0 = arm0.cross(contact.normal);
	const PxVec3 normalAngular1 = arm1.cross(contact.normal);
	const double penetration = double(contact.separation) - restDistance;
	const PxU32 firstContact = PxU32(problem.contacts.size());
	const bool separateFrictionCoefficients = frictionState && contact.staticFriction != contact.dynamicFriction;
	const PxReal frictionCoefficient = separateFrictionCoefficients && sliding ? contact.dynamicFriction : contact.staticFriction;
	PxVec3 tangent0(0.0f), tangent1(0.0f);
	double freeNormalVelocity = 0.0, targetNormalVelocity = 0.0;
	double freeTangentVelocity0 = 0.0, freeTangentVelocity1 = 0.0;
	bool correctDilatancy = false;
	if(contact.restitution < 0.0f)
	{
		if(!(contact.materialFlags & PxMaterialFlag::eDISABLE_FRICTION) && frictionCoefficient > 0.0f)
		{
			appendCompliantFriction(contact, normalAngular0, normalAngular1, penetration, arm0, arm1, body0, body1, bodyIndex0, bodyIndex1, jacobianBodies, frictionCoefficient, settings, problem, tangent0, tangent1);
			const PxVec3 freePointVelocity = body0.linearVelocity + body0.angularVelocity.cross(arm0) - body1.linearVelocity - body1.angularVelocity.cross(arm1) - contact.targetVel;
			freeTangentVelocity0 = dotNewtonContact(freePointVelocity, tangent0);
			freeTangentVelocity1 = dotNewtonContact(freePointVelocity, tangent1);
		}
		else
		{
			appendCompliantNormal(contact, contact.normal, normalAngular0, normalAngular1, penetration, body0, body1, bodyIndex0, bodyIndex1, jacobianBodies, settings, problem);
		}
	}
	else
	{
		const double impedance = settings.stiffeningDepth > 0.0 ? contactImpedance(settings, penetration) : settings.impedance;
		const double ratio = settings.stiffeningDepth > 0.0 ? (1.0 - impedance) / impedance : double(settings.regularization);
		const double damping = settings.damping;
		const double stiffness = settings.stiffness;
		const PxVec3 initialVelocity = initialPointVelocity(body0, body1, bodyIndex0, bodyIndex1, arm0, arm1, settings);
		const PxVec3 freePointVelocity = body0.linearVelocity + body0.angularVelocity.cross(arm0) -
			body1.linearVelocity - body1.angularVelocity.cross(arm1);
		const double initialNormalSpeed = dotNewtonContact(contact.normal, initialVelocity);
		const double penetrationSpeed = penetration / settings.timestep;
		const bool colliding = -initialNormalSpeed > penetrationSpeed;
		const bool bounce = contact.restitution > 0.0f && initialNormalSpeed < settings.bounceThreshold && colliding && penetration <= ccdMaxSeparation;
		const double normalTarget = dotNewtonContact(contact.targetVel, contact.normal) +
			(bounce ? -double(contact.restitution) * initialNormalSpeed : 0.0);
		freeNormalVelocity = dotNewtonContact(contact.normal, freePointVelocity);
		targetNormalVelocity = initialNormalSpeed - settings.timestep *
			(damping * (initialNormalSpeed - normalTarget) + stiffness * impedance * penetration);
		const bool friction = !hasContactImpulseLimit(contact.maxImpulse) && !(contact.materialFlags & PxMaterialFlag::eDISABLE_FRICTION) && frictionCoefficient > 0.0f;
		if(friction)
		{
			contactTangents(contact.normal, initialVelocity, tangent0, tangent1);
			freeTangentVelocity0 = dotNewtonContact(freePointVelocity - contact.targetVel, tangent0);
			freeTangentVelocity1 = dotNewtonContact(freePointVelocity - contact.targetVel, tangent1);
			const double mu = frictionCoefficient;
			// Compare squared slip distance with the squared tolerance to avoid a square root.
			const double slipScale = settings.timestep * mu;
			correctDilatancy = settings.correctDilatancy && !bounce && slipScale * slipScale * (freeTangentVelocity0 * freeTangentVelocity0 + freeTangentVelocity1 * freeTangentVelocity1) >
				double(settings.dilatancyTolerance) * settings.dilatancyTolerance;
			const double diagonalApproximation = translationResponse * (1.0 + mu * mu);
			const double edgeRegularization = std::max(1.0e-15, 2.0 * mu * mu * ratio * diagonalApproximation);
			// Edge directions n +/- mu t are linear, so every edge row combines the point's
			// normal and tangent Jacobians and speeds instead of rebuilding its own.
			newton::Vec6 normalJacobian[2];
			contactJacobian(normalJacobian[0], jacobianBodies[0], contact.normal, normalAngular0);
			contactJacobian(normalJacobian[1], jacobianBodies[1], contact.normal, normalAngular1);
			const PxVec3 tangents[2] = { tangent0, tangent1 };
			for(PxU32 tangent = 0; tangent < 2; ++tangent)
			{
				newton::Vec6 tangentJacobian[2];
				contactJacobian(tangentJacobian[0], jacobianBodies[0], tangents[tangent], arm0.cross(tangents[tangent]));
				contactJacobian(tangentJacobian[1], jacobianBodies[1], tangents[tangent], arm1.cross(tangents[tangent]));
				const double tangentTarget = dotNewtonContact(contact.targetVel, tangents[tangent]);
				const double tangentInitialSpeed = dotNewtonContact(tangents[tangent], initialVelocity);
				const double tangentFreeSpeed = dotNewtonContact(tangents[tangent], freePointVelocity);
				for(PxU32 sign = 0; sign < 2; ++sign)
				{
					const double scale = sign ? -mu : mu;
					const double initialSpeed = initialNormalSpeed + scale * tangentInitialSpeed;
					const double target = normalTarget + scale * tangentTarget;
					const double freeVelocity = freeNormalVelocity + scale * tangentFreeSpeed - initialSpeed + settings.timestep *
						(damping * (initialSpeed - target) + stiffness * impedance * penetration);
					appendEdgeRow(normalJacobian, tangentJacobian, scale, bodyIndex0, bodyIndex1, freeVelocity, edgeRegularization, problem);
				}
			}
			appendContactPoint(firstContact, forceDestination, separateFrictionCoefficients ? frictionState : NULL, frictionCoefficient, freeNormalVelocity, targetNormalVelocity, freeTangentVelocity0, freeTangentVelocity1, correctDilatancy, problem, output);
			return;
		}
		const double response = translationResponse > 0.0 ? translationResponse : 1.0;
		const double regularization = std::max(1.0e-15, ratio * response);
		const double upper = hasContactImpulseLimit(contact.maxImpulse) ? contact.maxImpulse : newton::MAX_IMPULSE;
		appendContactRow(contact.normal, normalAngular0, normalAngular1, initialNormalSpeed, dotNewtonContact(contact.normal, freePointVelocity), normalTarget, penetration, stiffness, damping, impedance, regularization, bodyIndex0, bodyIndex1, jacobianBodies, settings, problem, upper);
	}
	appendContactPoint(firstContact, forceDestination, separateFrictionCoefficients ? frictionState : NULL, frictionCoefficient, freeNormalVelocity, targetNormalVelocity, freeTangentVelocity0, freeTangentVelocity1, correctDilatancy, problem, output);
}

void prepareNewtonContacts(PxsContactManager& manager, PxsContactManagerOutput& contactOutput, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, const NewtonContactSettings& settings, ThreadContext& threadContext, newton::Problem& problem, NewtonContactRows& output)
{
	PxcNpWorkUnit& unit = manager.getWorkUnit();
	if(contactOutput.contactForces)
	{
		PxMemZero(contactOutput.contactForces, sizeof(PxReal) * contactOutput.nbContacts);
	}
	PX_ASSERT(unit.getDominance0() && unit.getDominance1());
	const bool sliding = unit.mFrictionPatchCount == NEWTON_FRICTION_STATE_MARKER && unit.mFrictionDataPtr && *unit.mFrictionDataPtr != 0;
	PxContactBuffer& buffer = threadContext.mContactBuffer;
	PxU16 originalIndices[PxContactBuffer::MAX_CONTACTS];
	bool separateFrictionCoefficients;
	extractNewtonContacts(contactOutput, buffer, originalIndices, PxMin(body0.maxContactImpulse, body1.maxContactImpulse), separateFrictionCoefficients);
	if(buffer.count == 0)
	{
		unit.mFrictionDataPtr = NULL;
		unit.mFrictionPatchCount = 0;
		return;
	}
	PxU8* frictionState = separateFrictionCoefficients ? threadContext.mFrictionPatchStreamPair.reserve<PxU8>(sizeof(PxU8)) : NULL;
	if(frictionState)
	{
		*frictionState = 0;
	}
	unit.mFrictionDataPtr = frictionState;
	unit.mFrictionPatchCount = frictionState ? NEWTON_FRICTION_STATE_MARKER : 0;

	const PxTransform& frame0 = unit.mRigidCore0->body2World;
	const PxTransform identity(PxIdentity);
	const PxTransform& frame1 = unit.mRigidCore1 ? unit.mRigidCore1->body2World : identity;
	NewtonContactPair pair;
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
	NewtonJacobianBody jacobianBodies[2];
	prepareJacobianBody(jacobianBodies[0], body0, bodyIndex0, rootInverseMass0, 1.0, settings.bodyLockFlags);
	prepareJacobianBody(jacobianBodies[1], body1, bodyIndex1, rootInverseMass1, -1.0, settings.bodyLockFlags);
	const PxU32 contactCount = buffer.count;
	for(PxU32 i = 0; i < contactCount; ++i)
	{
		appendNewtonContact(buffer.contacts[i], unit.mRestDistance, ccdMaxSeparation, body0, body1, bodyIndex0, bodyIndex1, frame0, frame1, jacobianBodies, translationResponse, settings, sliding, frictionState, contactOutput.contactForces ? contactOutput.contactForces + originalIndices[i] : NULL, problem, output);
	}
	pair.pointCount = output.points.size() - pair.firstPoint;
	output.pairs.pushBack(pair);
}

static double normalImpulse(const NewtonContactPoint& point, const newton::Problem& problem, const newton::Result& result)
{
	double impulse = 0.0;
	const PxU32 contactCount = point.contactCount;
	for(PxU32 i = 0; i < contactCount; ++i)
	{
		const newton::CompactContact& contact = problem.contacts[point.firstContact + i];
		impulse += result.impulse[contact.row + (contact.rowCount() == 3 ? 2 : 0)];
	}
	return impulse;
}

static double contactVelocityCorrection(const newton::CompactContact& contact, PxU32 axis, const newton::Problem& problem, const newton::Result& result)
{
	double velocity = 0.0;
	for(PxU32 end = 0; end < 2; ++end)
	{
		const PxI32 body = contact.body[end];
		if(body < 0)
		{
			continue;
		}
		for(PxU32 column = 0; column < 6; ++column)
		{
			velocity += problem.contactEntry(contact, end, axis, PxI32(column)) *
						result.primal[6 * body + column];
		}
	}
	return velocity;
}

static bool frictionVelocities(const NewtonFrictionPoint& frictionPoint, const NewtonContactRows& rows, const newton::Problem& problem, const newton::Result& result, double& normalVelocity, double& tangentVelocity0, double& tangentVelocity1)
{
	const NewtonContactPoint& point = rows.points[frictionPoint.pointIndex];
	if(point.contactCount != 4 || frictionPoint.friction <= 0.0f)
	{
		return false;
	}
	const newton::CompactContact& edge0 = problem.contacts[point.firstContact];
	const newton::CompactContact& edge1 = problem.contacts[point.firstContact + 1];
	const newton::CompactContact& edge2 = problem.contacts[point.firstContact + 2];
	const newton::CompactContact& edge3 = problem.contacts[point.firstContact + 3];
	const double correction0 = contactVelocityCorrection(edge0, 2, problem, result);
	const double correction1 = contactVelocityCorrection(edge1, 2, problem, result);
	const double correction2 = contactVelocityCorrection(edge2, 2, problem, result);
	const double correction3 = contactVelocityCorrection(edge3, 2, problem, result);
	const double tangentScale = 0.5 / frictionPoint.friction;
	normalVelocity = frictionPoint.freeNormalVelocity +
		0.25 * (correction0 + correction1 + correction2 + correction3);
	tangentVelocity0 = frictionPoint.freeTangentVelocity0 + tangentScale * (correction0 - correction1);
	tangentVelocity1 = frictionPoint.freeTangentVelocity1 + tangentScale * (correction2 - correction3);
	return true;
}

bool updateNewtonDilatancyBias(NewtonContactRows& rows, newton::Problem& problem, const newton::Result& result, PxReal velocityTolerance)
{
	bool changed = false;
	const PxU32 frictionPointCount = rows.frictionPoints.size();
	for(PxU32 i = 0; i < frictionPointCount; ++i)
	{
		NewtonFrictionPoint& frictionPoint = rows.frictionPoints[i];
		if(!frictionPoint.correctDilatancy)
		{
			continue;
		}
		double normalVelocity, tangentVelocity0, tangentVelocity1;
		if(!frictionVelocities(frictionPoint, rows, problem, result, normalVelocity, tangentVelocity0, tangentVelocity1))
		{
			continue;
		}
		if(frictionPoint.dilatancyBias == 0.0f && normalVelocity - frictionPoint.targetNormalVelocity <= velocityTolerance)
		{
			continue;
		}
		const double bias = frictionPoint.friction *
			std::sqrt(tangentVelocity0 * tangentVelocity0 + tangentVelocity1 * tangentVelocity1);
		const double difference = bias - frictionPoint.dilatancyBias;
		if(std::abs(difference) <= velocityTolerance)
		{
			continue;
		}
		const NewtonContactPoint& point = rows.points[frictionPoint.pointIndex];
		for(PxU32 edge = 0; edge < 4; ++edge)
		{
			problem.freeVelocity[problem.contacts[point.firstContact + edge].row] += difference;
		}
		frictionPoint.dilatancyBias = PxReal(bias);
		changed = true;
	}
	return changed;
}

struct NewtonPointFriction
{
	enum Enum
	{
		eUNLOADED,
		eHOLDING,
		eSATURATED
	};
};

// A point's friction is saturated when it reaches the cone limit. Soft friction rows let a
// holding contact creep in proportion to its load, so slip speed cannot separate the cases.
static NewtonPointFriction::Enum pointFriction(const NewtonFrictionPoint& frictionPoint, const NewtonContactRows& rows, const newton::Problem& problem, const newton::Result& result)
{
	const NewtonContactPoint& point = rows.points[frictionPoint.pointIndex];
	const newton::CompactContact& contact = problem.contacts[point.firstContact];
	if(point.contactCount == 1 && contact.rowCount() == 3)
	{
		// Three-row blocks clamp each tangent impulse to friction times the normal impulse.
		const double normal = result.impulse[contact.row + 2];
		if(normal <= 0.0)
		{
			return NewtonPointFriction::eUNLOADED;
		}
		const double limit = double(frictionPoint.friction) * normal * (1.0 - 1.0e-9);
		return std::abs(result.impulse[contact.row]) >= limit || std::abs(result.impulse[contact.row + 1]) >= limit ?
			NewtonPointFriction::eSATURATED : NewtonPointFriction::eHOLDING;
	}
	if(point.contactCount == 4 && frictionPoint.friction > 0.0f)
	{
		// Edges n + mu t0, n - mu t0, n + mu t1, n - mu t1 give friction mu (e0 - e1, e2 - e3)
		// and normal e0 + e1 + e2 + e3. |f0| + |f1| reaches mu N exactly when one edge of
		// each opposing pair is inactive; inactive rows have exactly zero impulse.
		double edge[4];
		for(PxU32 i = 0; i < 4; ++i)
		{
			edge[i] = result.impulse[problem.contacts[point.firstContact + i].row];
		}
		if(edge[0] + edge[1] + edge[2] + edge[3] <= 0.0)
		{
			return NewtonPointFriction::eUNLOADED;
		}
		return PxMin(edge[0], edge[1]) == 0.0 && PxMin(edge[2], edge[3]) == 0.0 ? NewtonPointFriction::eSATURATED : NewtonPointFriction::eHOLDING;
	}
	return frictionPoint.friction == 0.0f ? NewtonPointFriction::eSATURATED : NewtonPointFriction::eHOLDING;
}

void writebackNewtonContacts(const NewtonContactRows& rows, const newton::Problem& problem, const newton::Result& result, DynamicsContext& context)
{
	// A pair's points share one friction state. A rigid contact slides only when every loaded
	// point is saturated; one saturated corner does not move a body the others still hold.
	const PxU32 frictionPointCount = rows.frictionPoints.size();
	for(PxU32 first = 0; first < frictionPointCount;)
	{
		PxU8* state = rows.frictionPoints[first].state;
		PxU32 end = first + 1;
		while(end < frictionPointCount && rows.frictionPoints[end].state == state)
		{
			++end;
		}
		if(state && *state == 0)
		{
			bool loaded = false, saturated = true;
			for(PxU32 i = first; i < end && saturated; ++i)
			{
				const NewtonPointFriction::Enum friction = pointFriction(rows.frictionPoints[i], rows, problem, result);
				loaded = loaded || friction != NewtonPointFriction::eUNLOADED;
				saturated = friction != NewtonPointFriction::eHOLDING;
			}
			if(loaded && saturated)
			{
				*state = 1;
			}
		}
		first = end;
	}
	const PxU32 pointCount = rows.points.size();
	for(PxU32 i = 0; i < pointCount; ++i)
	{
		const NewtonContactPoint& point = rows.points[i];
		if(point.destination)
		{
			*point.destination = PxReal(normalImpulse(point, problem, result));
		}
	}
	const PxU32 pairCount = rows.pairs.size();
	for(PxU32 i = 0; i < pairCount; ++i)
	{
		const NewtonContactPair& pair = rows.pairs[i];
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
