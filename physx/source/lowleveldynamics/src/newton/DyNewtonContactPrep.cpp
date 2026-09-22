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

static double dotNewtonContact(const PxVec3& a, const PxVec3& b)
{
	return double(a.x) * b.x + double(a.y) * b.y + double(a.z) * b.z;
}

static bool hasContactImpulseLimit(PxReal maxImpulse)
{
	// PxsRigidCore uses 1e32f as its unlimited default.
	return maxImpulse < 1.0e32f;
}

static double contactSpeed(const PxVec3& linear0, const PxVec3& angularVelocity0,
	const PxVec3& linear1, const PxVec3& angularVelocity1, const PxVec3& direction,
	const PxVec3& angular0, const PxVec3& angular1)
{
	return dotNewtonContact(direction, linear0) - dotNewtonContact(direction, linear1) +
		dotNewtonContact(angular0, angularVelocity0) - dotNewtonContact(angular1, angularVelocity1);
}

static double contactSpeed(const PxSolverBodyData& body0, const PxSolverBodyData& body1,
	const PxVec3& direction, const PxVec3& angular0, const PxVec3& angular1)
{
	return contactSpeed(body0.linearVelocity, body0.angularVelocity, body1.linearVelocity,
		body1.angularVelocity, direction, angular0, angular1);
}

static void contactJacobian(newton::Vec6& jacobian, const PxSolverBodyData& body,
	const PxVec3& direction, const PxVec3& angular, PxI32 bodyIndex, double rootInverseMass,
	double sign, const PxU8* lockFlags)
{
	if(bodyIndex < 0)
	{
		jacobian.setZero();
		return;
	}
	const double linearScale = sign * rootInverseMass;
	const PxMat33& inertia = body.sqrtInvInertia;
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		jacobian[axis] = lockFlags && (lockFlags[bodyIndex] & (1 << axis)) ? 0.0 : linearScale * direction[axis];
		jacobian[axis + 3] = sign * (double(inertia.column0[axis]) * angular.x +
			double(inertia.column1[axis]) * angular.y + double(inertia.column2[axis]) * angular.z);
	}
}

static double prepareContactJacobian(newton::CompactContact& row,
	const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1,
	double rootInverseMass0, double rootInverseMass1, const PxVec3& direction,
	const PxVec3& angular0, const PxVec3& angular1, const PxU8* lockFlags)
{
	row.body[0] = bodyIndex0;
	row.body[1] = bodyIndex1;
	contactJacobian(row.jacobian[0], body0, direction, angular0, bodyIndex0, rootInverseMass0, 1.0, lockFlags);
	contactJacobian(row.jacobian[1], body1, direction, angular1, bodyIndex1, rootInverseMass1, -1.0, lockFlags);
	return row.jacobian[0].squaredNorm() + row.jacobian[1].squaredNorm();
}

static void applyContactSlop(PxVec3& vector, double slop)
{
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		if(std::abs(double(vector[axis])) < slop)
		{
			vector[axis] = 0.0f;
		}
	}
}

static void contactTangents(const PxVec3& normal, const PxVec3& relativeVelocity, PxVec3& tangent0, PxVec3& tangent1)
{
	tangent0 = relativeVelocity - normal * normal.dot(relativeVelocity);
	if(tangent0.magnitudeSquared() <= 0.0001f)
	{
		tangent0 = PxAbs(normal.x) < 0.70710678f ? PxVec3(0.0f, -normal.z, normal.y) : PxVec3(-normal.y, normal.x, 0.0f);
	}
	tangent0.normalize();
	tangent1 = normal.cross(tangent0);
}

static PX_FORCE_INLINE PxVec3 initialPointVelocity(const PxSolverBodyData& body0, const PxSolverBodyData& body1,
	PxI32 bodyIndex0, PxI32 bodyIndex1, const PxVec3& arm0, const PxVec3& arm1,
	const NewtonContactSettings& settings)
{
	const PxVec3& linear0 = bodyIndex0 >= 0 ? settings.initialVelocities[bodyIndex0].linear : body0.linearVelocity;
	const PxVec3& angular0 = bodyIndex0 >= 0 ? settings.initialVelocities[bodyIndex0].angular : body0.angularVelocity;
	const PxVec3& linear1 = bodyIndex1 >= 0 ? settings.initialVelocities[bodyIndex1].linear : body1.linearVelocity;
	const PxVec3& angular1 = bodyIndex1 >= 0 ? settings.initialVelocities[bodyIndex1].angular : body1.angularVelocity;
	return linear0 + angular0.cross(arm0) - linear1 - angular1.cross(arm1);
}

static const char* extractNewtonContacts(PxsContactManagerOutput& contactOutput, PxContactBuffer& buffer,
	PxU16* originalIndices, PxReal defaultMaxImpulse, bool& separateFrictionCoefficients)
{
	buffer.count = 0;
	separateFrictionCoefficients = false;
	if(!contactOutput.nbContacts)
	{
		return NULL;
	}
	PxContactStreamIterator iterator(contactOutput.contactPatches, contactOutput.contactPoints,
		contactOutput.getInternalFaceIndice(), contactOutput.nbPatches, contactOutput.nbContacts);
	if(iterator.forceNoResponse)
	{
		return NULL;
	}
	if(iterator.getInvMassScale0() != 1.0f || iterator.getInvMassScale1() != 1.0f ||
	   iterator.getInvInertiaScale0() != 1.0f || iterator.getInvInertiaScale1() != 1.0f)
	{
		return "Newton does not support contact-local inverse mass or inertia scaling.";
	}

	PxU32 originalIndex = 0;
	while(iterator.hasNextPatch())
	{
		iterator.nextPatch();
		separateFrictionCoefficients = separateFrictionCoefficients ||
			iterator.getStaticFriction() != iterator.getDynamicFriction();
		const bool hasMaxImpulse = (iterator.patch->internalFlags & PxContactPatch::eHAS_MAX_IMPULSE) != 0;
		while(iterator.hasNextContact())
		{
			iterator.nextContact();
			const PxReal maxImpulse = hasMaxImpulse ? iterator.getMaxImpulse() : defaultMaxImpulse;
			if(maxImpulse != 0.0f)
			{
				if(buffer.count == PxContactBuffer::MAX_CONTACTS)
				{
					return "Newton contact preparation exceeded PxContactBuffer::MAX_CONTACTS.";
				}
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
	return NULL;
}

static void appendContactRow(const PxVec3& direction,
	const PxVec3& angular0, const PxVec3& angular1, double initialSpeed, double freeSpeed,
	double targetSpeed, double positionError,
	double stiffness, double damping, double impedance, double regularization,
	const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1,
	double rootInverseMass0, double rootInverseMass1, const NewtonContactSettings& settings,
	newton::Problem& problem, double upperImpulse)
{
	newton::CompactContact row;
	if(prepareContactJacobian(row, body0, body1, bodyIndex0, bodyIndex1,
							  rootInverseMass0, rootInverseMass1, direction, angular0, angular1, settings.bodyLockFlags) == 0.0)
	{
		return;
	}
	row.freeVelocity = freeSpeed - initialSpeed + settings.timestep *
		(damping * (initialSpeed - targetSpeed) + stiffness * impedance * positionError);
	row.regularization = regularization;
	problem.addScalarContact(row, 0.0, upperImpulse);
}

static bool prepareCompliantNormal(const PxContactPoint& contact, const PxVec3& normal,
	const PxVec3& angular0, const PxVec3& angular1, double penetration,
	const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1,
	double rootInverseMass0, double rootInverseMass1,
	const NewtonContactSettings& settings, newton::CompactContact& row)
{
	const double response = prepareContactJacobian(row, body0, body1, bodyIndex0, bodyIndex1,
		rootInverseMass0, rootInverseMass1, normal, angular0, angular1, settings.bodyLockFlags);
	if(response == 0.0)
	{
		return false;
	}
	const double speed = contactSpeed(body0, body1, normal, angular0, angular1);
	const double stiffness = -double(contact.restitution);
	const double damping = penetration >= 0.0 ? 0.0 : double(contact.damping);
	const double coefficient = settings.timestep * (damping + settings.timestep * stiffness);
	const double target = dotNewtonContact(contact.targetVel, normal);
	row.regularization = (contact.materialFlags & PxMaterialFlag::eCOMPLIANT_ACCELERATION_SPRING) ?
		response / coefficient : 1.0 / coefficient;
	row.freeVelocity = speed - target + settings.timestep * stiffness * penetration / coefficient;
	return true;
}

static void appendCompliantNormal(const PxContactPoint& contact, const PxVec3& normal,
	const PxVec3& angular0, const PxVec3& angular1, double penetration,
	const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1,
	double rootInverseMass0, double rootInverseMass1,
	const NewtonContactSettings& settings, newton::Problem& problem)
{
	newton::CompactContact row;
	if(!prepareCompliantNormal(contact, normal, angular0, angular1, penetration,
							   body0, body1, bodyIndex0, bodyIndex1, rootInverseMass0, rootInverseMass1, settings, row))
	{
		return;
	}
	const double upper = hasContactImpulseLimit(contact.maxImpulse) ? contact.maxImpulse : newton::MAX_IMPULSE;
	problem.addScalarContact(row, 0.0, upper);
}

static void appendCompliantFriction(const PxContactPoint& contact,
	const PxVec3& normalAngular0, const PxVec3& normalAngular1, double penetration,
	const PxVec3& arm0, const PxVec3& arm1,
	const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1,
	double rootInverseMass0, double rootInverseMass1, PxReal offsetSlop, double friction,
	const NewtonContactSettings& settings, newton::Problem& problem, PxVec3& tangent0, PxVec3& tangent1)
{
	newton::CompactContact normal;
	if(!prepareCompliantNormal(contact, contact.normal, normalAngular0, normalAngular1, penetration,
							   body0, body1, bodyIndex0, bodyIndex1, rootInverseMass0, rootInverseMass1, settings, normal))
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
		PxVec3 angular0 = arm0.cross(tangents[tangent]);
		PxVec3 angular1 = arm1.cross(tangents[tangent]);
		applyContactSlop(angular0, offsetSlop);
		applyContactSlop(angular1, offsetSlop);
		newton::Vec6 jacobian0, jacobian1;
		contactJacobian(jacobian0, body0, tangents[tangent], angular0,
			bodyIndex0, rootInverseMass0, 1.0, settings.bodyLockFlags);
		contactJacobian(jacobian1, body1, tangents[tangent], angular1,
			bodyIndex1, rootInverseMass1, -1.0, settings.bodyLockFlags);
		block.jacobian[0].row(tangent) = jacobian0;
		block.jacobian[1].row(tangent) = jacobian1;
		block.freeVelocity[tangent] = contactSpeed(body0, body1, tangents[tangent], angular0, angular1) -
			dotNewtonContact(contact.targetVel, tangents[tangent]);
		tangentResponse += jacobian0.squaredNorm() + jacobian1.squaredNorm();
	}
	const double tangentRegularization = std::max(1.0e-15, settings.regularization * tangentResponse * 0.5);
	block.regularization[0] = block.regularization[1] = tangentRegularization;
	block.friction = friction;
	block.maxNormalImpulse = hasContactImpulseLimit(contact.maxImpulse) ? contact.maxImpulse : newton::MAX_IMPULSE;
	problem.addContact(block);
}

static void appendContactPoint(PxU32 firstContact, PxReal* forceDestination, PxU8* frictionState,
	PxReal friction, double freeNormalVelocity, double targetNormalVelocity,
	double freeTangentVelocity0, double freeTangentVelocity1, bool correctDilatancy,
	newton::Problem& problem, NewtonContactRows& output)
{
	const NewtonContactPoint point = { firstContact, PxU32(problem.contacts.size()) - firstContact, forceDestination };
	const PxU32 pointIndex = output.points.size();
	output.points.pushBack(point);
	if((frictionState || correctDilatancy) && point.contactCount)
	{
		const NewtonFrictionPoint frictionPoint = { pointIndex, friction, PxReal(freeNormalVelocity),
			PxReal(targetNormalVelocity), PxReal(freeTangentVelocity0), PxReal(freeTangentVelocity1),
			0.0f, frictionState, correctDilatancy };
		output.frictionPoints.pushBack(frictionPoint);
	}
}

static void appendNewtonContact(const PxContactPoint& contact, PxReal restDistance, PxReal offsetSlop,
	PxReal ccdMaxSeparation, const PxSolverBodyData& body0, const PxSolverBodyData& body1,
	PxI32 bodyIndex0, PxI32 bodyIndex1, const PxTransform& frame0, const PxTransform& frame1,
	double rootInverseMass0, double rootInverseMass1, double translationResponse,
	const NewtonContactSettings& settings, bool sliding, PxU8* frictionState, PxReal* forceDestination,
	newton::Problem& problem, NewtonContactRows& output)
{
	const PxVec3 arm0 = contact.point - frame0.p;
	const PxVec3 arm1 = contact.point - frame1.p;
	PxVec3 normalAngular0 = arm0.cross(contact.normal);
	PxVec3 normalAngular1 = arm1.cross(contact.normal);
	applyContactSlop(normalAngular0, offsetSlop);
	applyContactSlop(normalAngular1, offsetSlop);
	const double penetration = double(contact.separation) - restDistance;
	const PxU32 firstContact = PxU32(problem.contacts.size());
	const bool separateFrictionCoefficients = frictionState && contact.staticFriction != contact.dynamicFriction;
	const PxReal frictionCoefficient = separateFrictionCoefficients && sliding ?
		contact.dynamicFriction : contact.staticFriction;
	PxVec3 tangent0(0.0f), tangent1(0.0f);
	double freeNormalVelocity = 0.0, targetNormalVelocity = 0.0;
	double freeTangentVelocity0 = 0.0, freeTangentVelocity1 = 0.0;
	bool correctDilatancy = false;
	if(contact.restitution < 0.0f)
	{
		if(!(contact.materialFlags & PxMaterialFlag::eDISABLE_FRICTION) && frictionCoefficient > 0.0f)
		{
			appendCompliantFriction(contact, normalAngular0, normalAngular1, penetration, arm0, arm1,
				body0, body1, bodyIndex0, bodyIndex1, rootInverseMass0, rootInverseMass1,
				offsetSlop, frictionCoefficient, settings, problem, tangent0, tangent1);
			const PxVec3 freePointVelocity = body0.linearVelocity + body0.angularVelocity.cross(arm0) -
				body1.linearVelocity - body1.angularVelocity.cross(arm1) - contact.targetVel;
			freeTangentVelocity0 = dotNewtonContact(freePointVelocity, tangent0);
			freeTangentVelocity1 = dotNewtonContact(freePointVelocity, tangent1);
		}
		else
		{
			appendCompliantNormal(contact, contact.normal, normalAngular0, normalAngular1, penetration,
								  body0, body1, bodyIndex0, bodyIndex1, rootInverseMass0, rootInverseMass1, settings, problem);
		}
	}
	else
	{
		const double ratio = settings.regularization;
		const double impedance = 1.0 / (1.0 + ratio);
		const double timeConstant = std::max(0.02, 2.0 * double(settings.timestep));
		const double damping = 2.0 / (impedance * timeConstant);
		const double stiffness = 1.0 / (impedance * impedance * timeConstant * timeConstant);
		const PxVec3 initialVelocity = initialPointVelocity(body0, body1, bodyIndex0, bodyIndex1, arm0, arm1, settings);
		const PxVec3 freePointVelocity = body0.linearVelocity + body0.angularVelocity.cross(arm0) -
			body1.linearVelocity - body1.angularVelocity.cross(arm1);
		const double initialNormalSpeed = dotNewtonContact(contact.normal, initialVelocity);
		const double penetrationSpeed = penetration / settings.timestep;
		const bool colliding = -initialNormalSpeed > penetrationSpeed;
		const bool bounce = contact.restitution > 0.0f && initialNormalSpeed < settings.bounceThreshold && colliding &&
			penetration <= ccdMaxSeparation;
		const double normalTarget = dotNewtonContact(contact.targetVel, contact.normal) +
			(bounce ? -double(contact.restitution) * initialNormalSpeed : 0.0);
		freeNormalVelocity = dotNewtonContact(contact.normal, freePointVelocity);
		targetNormalVelocity = initialNormalSpeed - settings.timestep *
			(damping * (initialNormalSpeed - normalTarget) + stiffness * impedance * penetration);
		const bool friction = !hasContactImpulseLimit(contact.maxImpulse) &&
			!(contact.materialFlags & PxMaterialFlag::eDISABLE_FRICTION) && frictionCoefficient > 0.0f;
		if(friction)
		{
			contactTangents(contact.normal, initialVelocity, tangent0, tangent1);
			freeTangentVelocity0 = dotNewtonContact(freePointVelocity - contact.targetVel, tangent0);
			freeTangentVelocity1 = dotNewtonContact(freePointVelocity - contact.targetVel, tangent1);
			const double mu = frictionCoefficient;
			correctDilatancy = settings.correctDilatancy && !bounce && settings.timestep * mu *
				std::sqrt(freeTangentVelocity0 * freeTangentVelocity0 +
					freeTangentVelocity1 * freeTangentVelocity1) > settings.dilatancyTolerance;
			const double diagonalApproximation = translationResponse * (1.0 + mu * mu);
			const double edgeRegularization = std::max(1.0e-15,
				2.0 * mu * mu * ratio * diagonalApproximation);
			const PxVec3 tangents[2] = { tangent0, tangent1 };
			for(PxU32 tangent = 0; tangent < 2; ++tangent)
			{
				for(PxU32 sign = 0; sign < 2; ++sign)
				{
					const double scale = sign ? -mu : mu;
					const PxVec3 direction = contact.normal + tangents[tangent] * PxReal(scale);
					PxVec3 angular0 = arm0.cross(direction);
					PxVec3 angular1 = arm1.cross(direction);
					applyContactSlop(angular0, offsetSlop);
					applyContactSlop(angular1, offsetSlop);
					const double target = normalTarget + scale * dotNewtonContact(contact.targetVel, tangents[tangent]);
					appendContactRow(direction, angular0, angular1,
						dotNewtonContact(direction, initialVelocity), dotNewtonContact(direction, freePointVelocity),
						target, penetration,
						stiffness, damping, impedance, edgeRegularization, body0, body1,
						bodyIndex0, bodyIndex1, rootInverseMass0, rootInverseMass1,
						settings, problem, newton::MAX_IMPULSE);
				}
			}
			appendContactPoint(firstContact, forceDestination, separateFrictionCoefficients ? frictionState : NULL,
				frictionCoefficient, freeNormalVelocity, targetNormalVelocity,
				freeTangentVelocity0, freeTangentVelocity1, correctDilatancy, problem, output);
			return;
		}
		const double response = translationResponse > 0.0 ? translationResponse : 1.0;
		const double regularization = std::max(1.0e-15, ratio * response);
		const double upper = hasContactImpulseLimit(contact.maxImpulse) ? contact.maxImpulse : newton::MAX_IMPULSE;
		appendContactRow(contact.normal, normalAngular0, normalAngular1, initialNormalSpeed,
			dotNewtonContact(contact.normal, freePointVelocity), normalTarget, penetration,
			stiffness, damping, impedance, regularization, body0, body1,
			bodyIndex0, bodyIndex1, rootInverseMass0, rootInverseMass1, settings, problem, upper);
	}
	appendContactPoint(firstContact, forceDestination, separateFrictionCoefficients ? frictionState : NULL,
		frictionCoefficient, freeNormalVelocity, targetNormalVelocity,
		freeTangentVelocity0, freeTangentVelocity1, correctDilatancy, problem, output);
}

const char* prepareNewtonContacts(PxsContactManager& manager, PxsContactManagerOutput& contactOutput,
	const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1,
	const NewtonContactSettings& settings, ThreadContext& threadContext,
	newton::Problem& problem, NewtonContactRows& output)
{
	PxcNpWorkUnit& unit = manager.getWorkUnit();
	if(contactOutput.contactForces)
	{
		PxMemZero(contactOutput.contactForces, sizeof(PxReal) * contactOutput.nbContacts);
	}
	if(!unit.getDominance0() || !unit.getDominance1())
	{
		return "Newton does not support asymmetric contact dominance.";
	}
	const bool sliding = unit.mFrictionPatchCount == NEWTON_FRICTION_STATE_MARKER &&
		unit.mFrictionDataPtr && *unit.mFrictionDataPtr != 0;
	PxContactBuffer& buffer = threadContext.mContactBuffer;
	PxU16 originalIndices[PxContactBuffer::MAX_CONTACTS];
	bool separateFrictionCoefficients;
	const char* error = extractNewtonContacts(contactOutput, buffer, originalIndices,
		PxMin(body0.maxContactImpulse, body1.maxContactImpulse), separateFrictionCoefficients);
	if(error)
	{
		return error;
	}
	if(buffer.count == 0)
	{
		unit.mFrictionDataPtr = NULL;
		unit.mFrictionPatchCount = 0;
		return NULL;
	}
	PxU8* frictionState = separateFrictionCoefficients ?
		threadContext.mFrictionPatchStreamPair.reserve<PxU8>(sizeof(PxU8)) : NULL;
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
	pair.reportThreshold = (unit.mFlags & PxcNpWorkUnitFlag::eFORCE_THRESHOLD) &&
		(body0.reportThreshold < PX_MAX_F32 || body1.reportThreshold < PX_MAX_F32);
	pair.threshold.shapeInteraction = reinterpret_cast<Sc::ShapeInteraction*>(manager.getShapeInteraction());
	pair.threshold.threshold = PxMin(body0.reportThreshold, body1.reportThreshold);
	pair.threshold.nodeIndexA = PxNodeIndex(body0.nodeIndex);
	pair.threshold.nodeIndexB = PxNodeIndex(body1.nodeIndex);
	PxOrder(pair.threshold.nodeIndexA, pair.threshold.nodeIndexB);
	pair.threshold.normalForce = pair.threshold.accumulatedForce = 0.0f;
	pair.threshold.pad = 0;
	const PxU32 bodyFlags = PxU32(unit.mRigidCore0->mFlags) | (unit.mRigidCore1 ? PxU32(unit.mRigidCore1->mFlags) : 0);
	const PxReal ccdMaxSeparation = bodyFlags & PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD ?
		settings.ccdMaxSeparation : PX_MAX_F32;
	const double rootInverseMass0 = bodyIndex0 >= 0 ? std::sqrt(double(body0.invMass)) : 0.0;
	const double rootInverseMass1 = bodyIndex1 >= 0 ? std::sqrt(double(body1.invMass)) : 0.0;
	const double translationResponse = rootInverseMass0 * rootInverseMass0 + rootInverseMass1 * rootInverseMass1;
	for(PxU32 i = 0; i < buffer.count; ++i)
	{
		appendNewtonContact(buffer.contacts[i], unit.mRestDistance, unit.mOffsetSlop, ccdMaxSeparation,
							body0, body1, bodyIndex0, bodyIndex1, frame0, frame1,
							rootInverseMass0, rootInverseMass1, translationResponse, settings, sliding, frictionState,
							contactOutput.contactForces ? contactOutput.contactForces + originalIndices[i] : NULL, problem, output);
	}
	pair.pointCount = output.points.size() - pair.firstPoint;
	output.pairs.pushBack(pair);
	return NULL;
}

static double normalImpulse(const NewtonContactPoint& point, const newton::Problem& problem, const newton::Result& result)
{
	double impulse = 0.0;
	for(PxU32 i = 0; i < point.contactCount; ++i)
	{
		const newton::CompactContact& contact = problem.contacts[point.firstContact + i];
		impulse += result.impulse[contact.row + (contact.rowCount() == 3 ? 2 : 0)];
	}
	return impulse;
}

static double contactVelocityCorrection(const newton::CompactContact& contact, PxU32 axis,
	const newton::Problem& problem, const newton::Result& result)
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

static bool frictionVelocities(const NewtonFrictionPoint& frictionPoint, const NewtonContactRows& rows,
	const newton::Problem& problem, const newton::Result& result,
	double& normalVelocity, double& tangentVelocity0, double& tangentVelocity1)
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

bool updateNewtonDilatancyBias(NewtonContactRows& rows, newton::Problem& problem,
	const newton::Result& result, PxReal velocityTolerance)
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
		if(!frictionVelocities(frictionPoint, rows, problem, result,
							   normalVelocity, tangentVelocity0, tangentVelocity1))
		{
			continue;
		}
		if(frictionPoint.dilatancyBias == 0.0f &&
		   normalVelocity - frictionPoint.targetNormalVelocity <= velocityTolerance)
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

static bool isSliding(const NewtonFrictionPoint& frictionPoint, const NewtonContactRows& rows,
	const newton::Problem& problem, const newton::Result& result, double velocityThreshold)
{
	const NewtonContactPoint& point = rows.points[frictionPoint.pointIndex];
	const newton::CompactContact& contact = problem.contacts[point.firstContact];
	double tangentVelocity0 = frictionPoint.freeTangentVelocity0;
	double tangentVelocity1 = frictionPoint.freeTangentVelocity1;
	if(point.contactCount == 1 && contact.rowCount() == 3)
	{
		tangentVelocity0 += contactVelocityCorrection(contact, 0, problem, result);
		tangentVelocity1 += contactVelocityCorrection(contact, 1, problem, result);
	}
	else if(point.contactCount == 4 && frictionPoint.friction > 0.0f)
	{
		const newton::CompactContact& edge1 = problem.contacts[point.firstContact + 1];
		const newton::CompactContact& edge2 = problem.contacts[point.firstContact + 2];
		const newton::CompactContact& edge3 = problem.contacts[point.firstContact + 3];
		const double scale = 0.5 / frictionPoint.friction;
		tangentVelocity0 += scale * (contactVelocityCorrection(contact, 2, problem, result) -
			contactVelocityCorrection(edge1, 2, problem, result));
		tangentVelocity1 += scale * (contactVelocityCorrection(edge2, 2, problem, result) -
			contactVelocityCorrection(edge3, 2, problem, result));
	}
	else
	{
		return frictionPoint.friction == 0.0f;
	}
	return tangentVelocity0 * tangentVelocity0 + tangentVelocity1 * tangentVelocity1 >
		velocityThreshold * velocityThreshold;
}

void writebackNewtonContacts(const NewtonContactRows& rows, const newton::Problem& problem,
	const newton::Result& result, DynamicsContext& context)
{
	if(rows.frictionPoints.size())
	{
		const double velocityThreshold = double(PX_EPS_F32) * context.getLengthScale() / context.getDt();
		const PxU32 frictionPointCount = rows.frictionPoints.size();
		for(PxU32 i = 0; i < frictionPointCount; ++i)
		{
			const NewtonFrictionPoint& point = rows.frictionPoints[i];
			if(point.state && *point.state == 0 && isSliding(point, rows, problem, result, velocityThreshold))
			{
				*point.state = 1;
			}
		}
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
		for(PxU32 j = pair.firstPoint; j < pair.firstPoint + pair.pointCount; ++j)
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
