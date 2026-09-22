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

#include "DyNewtonConstraintPrep.h"
#include "DyConstraintPrep.h"
#include "PxsRigidBody.h"
#include "CmSpatialVector.h"
#include "solver/PxSolverDefs.h"
#include "core/NewtonSolver.h"

#include <cmath>

namespace physx
{
namespace Dy
{
static double projectNewtonVelocity(const PxSolverBodyData& body, const PxVec3& linear, const PxVec3& angular)
{
	return double(linear.x) * body.linearVelocity.x + double(linear.y) * body.linearVelocity.y +
		double(linear.z) * body.linearVelocity.z + double(angular.x) * body.angularVelocity.x +
		double(angular.y) * body.angularVelocity.y + double(angular.z) * body.angularVelocity.z;
}

static double projectInitialVelocity(const PxSolverBodyData& body, PxI32 bodyIndex, const PxVec3& linear, const PxVec3& angular, const NewtonJointSettings& settings)
{
	if(bodyIndex < 0)
	{
		return projectNewtonVelocity(body, linear, angular);
	}
	const Cm::SpatialVector& velocity = settings.initialVelocities[bodyIndex];
	return double(linear.x) * velocity.linear.x + double(linear.y) * velocity.linear.y +
		double(linear.z) * velocity.linear.z + double(angular.x) * velocity.angular.x +
		double(angular.y) * velocity.angular.y + double(angular.z) * velocity.angular.z;
}

static void setNewtonJointJacobian(newton::Vec6& jacobian, const PxSolverBodyData& body, const PxVec3& linear, const PxVec3& angular, PxI32 bodyIndex, double sign, const PxU8* bodyLockFlags)
{
	if(bodyIndex < 0)
	{
		jacobian.setZero();
		return;
	}
	const double linearScale = sign * std::sqrt(double(body.invMass));
	const PxMat33& inertia = body.sqrtInvInertia;
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		jacobian[axis] = bodyLockFlags && (bodyLockFlags[bodyIndex] & (1 << axis)) ? 0.0 : linearScale * linear[axis];
		jacobian[axis + 3] = sign * (double(inertia.column0[axis]) * angular.x +
			double(inertia.column1[axis]) * angular.y + double(inertia.column2[axis]) * angular.z);
	}
}

static void preprocessNewtonSlerp(Px1DConstraint* rows, PxU32 rowCount, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1)
{
	Px1DConstraint driveRows[3];
	PxU32 indices[3];
	PxU32 driveCount = 0;
	for(PxU32 i = 0; i < rowCount; ++i)
	{
		if((rows[i].solveHint >> 8) != 1)
		{
			continue;
		}
		PX_ASSERT(driveCount < 3);
		indices[driveCount] = i;
		driveRows[driveCount++] = rows[i];
	}
	if(driveCount == 0)
	{
		return;
	}
	PX_ASSERT(driveCount == 3);

	// Native SLERP applies acceleration gains and per-axis force caps in response
	// eigenaxes. Preprocess only this triple; leave other joint rows in their physical axes.
	Px1DConstraint* sorted[3];
	PX_ALIGN(16, PxVec4) angular0[3];
	PX_ALIGN(16, PxVec4) angular1[3];
	const PxMat33 zero(PxZero);
	preprocessRows(sorted, driveRows, angular0, angular1, 3, bodyIndex0 >= 0 ? body0.sqrtInvInertia : zero, bodyIndex1 >= 0 ? body1.sqrtInvInertia : zero, bodyIndex0 >= 0 ? body0.invMass : 0.0f, bodyIndex1 >= 0 ? body1.invMass : 0.0f, PxConstraintInvMassScale(1.0f, 1.0f, 1.0f, 1.0f), false, true);
	for(PxU32 i = 0; i < 3; ++i)
	{
		rows[indices[i]] = driveRows[i];
	}
}

void prepareNewtonJoint(const Constraint& constraint, const PxSolverBodyData& body0, const PxSolverBodyData& body1, PxI32 bodyIndex0, PxI32 bodyIndex1, ConstraintWriteback* writeback, const NewtonJointSettings& settings, newton::Problem& problem, NewtonJointRows& output)
{
	if((constraint.flags & PxConstraintFlag::eBROKEN) || (writeback && writeback->broken))
	{
		return;
	}

	NewtonJointWriteback joint;
	joint.destination = writeback;
	joint.firstRow = output.rows.size();
	joint.body0WorldOffset = PxVec3(0.0f);
	joint.linearBreakImpulse = constraint.linBreakForce * settings.timestep;
	joint.angularBreakImpulse = constraint.angBreakForce * settings.timestep;

	Px1DConstraint rows[MAX_CONSTRAINT_ROWS];
	PxU32 rowCount = 0;
	if(constraint.solverPrep && !(constraint.flags & PxConstraintFlag::eDISABLE_CONSTRAINT))
	{
		setupConstraintRows(rows, MAX_CONSTRAINT_ROWS);
		PxConstraintInvMassScale massScales(1.0f, 1.0f, 1.0f, 1.0f);
		const PxTransform identity(PxIdentity);
		const PxTransform& frame0 = constraint.body0 ? constraint.body0->getPose() : identity;
		const PxTransform& frame1 = constraint.body1 ? constraint.body1->getPose() : identity;
		PxVec3p anchor0, anchor1, body0WorldOffset;
		body0WorldOffset = PxVec3(0.0f);
		rowCount = constraint.solverPrep(rows, body0WorldOffset, MAX_CONSTRAINT_ROWS, massScales, constraint.constantBlock, frame0, frame1, (constraint.flags & PxConstraintFlag::eENABLE_EXTENDED_LIMITS) != 0, anchor0, anchor1);
		joint.body0WorldOffset = body0WorldOffset;
		PX_ASSERT(rowCount <= MAX_CONSTRAINT_ROWS);
		// Per-constraint mass scaling makes different constraints act through different mass
		// matrices. It cannot be represented by the shared symmetric Newton objective.
		PX_ASSERT(!rowCount || (massScales.linear0 == 1.0f && massScales.angular0 == 1.0f && massScales.linear1 == 1.0f && massScales.angular1 == 1.0f));
	}

	if((constraint.flags & PxConstraintFlag::eIMPROVED_SLERP) && !(constraint.flags & PxConstraintFlag::eDISABLE_PREPROCESSING))
	{
		preprocessNewtonSlerp(rows, rowCount, body0, body1, bodyIndex0, bodyIndex1);
	}

	const double timestep = settings.timestep;
	for(PxU32 i = 0; i < rowCount; ++i)
	{
		const Px1DConstraint& row = rows[i];
		newton::CompactContact contact;
		contact.body[0] = bodyIndex0;
		contact.body[1] = bodyIndex1;
		setNewtonJointJacobian(contact.jacobian[0], body0, row.linear0, row.angular0, bodyIndex0, 1.0, settings.bodyLockFlags);
		setNewtonJointJacobian(contact.jacobian[1], body1, row.linear1, row.angular1, bodyIndex1, -1.0, settings.bodyLockFlags);
		if(bodyIndex0 >= 0 && bodyIndex0 == bodyIndex1)
		{
			contact.jacobian[0] += contact.jacobian[1];
			contact.jacobian[1].setZero();
			contact.body[1] = -1;
		}
		const double response = contact.jacobian[0].squaredNorm() + contact.jacobian[1].squaredNorm();
		if(response == 0.0)
		{
			continue;
		}
		const bool spring = (row.flags & Px1DConstraintFlag::eSPRING) != 0;
		const bool accelerationSpring = (row.flags & Px1DConstraintFlag::eACCELERATION_SPRING) != 0;
		if((!spring || accelerationSpring) && response <= constraint.minResponseThreshold)
		{
			continue;
		}

		const double freeSpeed = projectNewtonVelocity(body0, row.linear0, row.angular0) - projectNewtonVelocity(body1, row.linear1, row.angular1);
		const double driveScale = (row.flags & Px1DConstraintFlag::eHAS_DRIVE_LIMIT) && (constraint.flags & PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES) ? timestep : 1.0;
		double lower = row.minImpulse == -PX_MAX_F32 ? -newton::MAX_IMPULSE : double(row.minImpulse) * driveScale;
		double upper = row.maxImpulse == PX_MAX_F32 ? newton::MAX_IMPULSE : double(row.maxImpulse) * driveScale;
		if(spring)
		{
			const double stiffness = row.mods.spring.stiffness;
			const double damping = row.mods.spring.damping;
			PX_ASSERT(stiffness >= 0.0 && damping >= 0.0);
			const double a = timestep * (damping + timestep * stiffness);
			if(a == 0.0)
			{
				// A zero spring produces zero impulse, clamped to any explicitly supplied limits.
				const double fixedImpulse = PxClamp(0.0, lower, upper);
				if(fixedImpulse == 0.0)
				{
					continue;
				}
				lower = upper = fixedImpulse;
				contact.regularization = response;
				contact.freeVelocity = freeSpeed;
			}
			else
			{
				const double b = timestep * (damping * row.velocityTarget - stiffness * row.geometricError);
				contact.regularization = (accelerationSpring ? response : 1.0) / a;
				contact.freeVelocity = freeSpeed - b / a;
			}
		}
		else
		{
			contact.regularization = response * settings.regularization;
			const double bounceSpeed = -double(row.mods.bounce.restitution) * freeSpeed;
			if((row.flags & Px1DConstraintFlag::eRESTITUTION) && -freeSpeed > row.mods.bounce.velocityThreshold && bounceSpeed * row.geometricError <= 0.0 && bounceSpeed != 0.0)
			{
				contact.freeVelocity = freeSpeed - bounceSpeed;
			}
			else
			{
				const double initialSpeed = projectInitialVelocity(body0, bodyIndex0, row.linear0, row.angular0, settings) - projectInitialVelocity(body1, bodyIndex1, row.linear1, row.angular1, settings);
				const double impedance = 1.0 / (1.0 + double(settings.regularization));
				const double timeConstant = std::max(0.02, 2.0 * timestep);
				const double damping = 2.0 / (impedance * timeConstant);
				const double stiffness = 1.0 / (impedance * impedance * timeConstant * timeConstant);
				contact.freeVelocity = freeSpeed - initialSpeed + timestep *
					(damping * (initialSpeed - row.velocityTarget) + stiffness * impedance * row.geometricError);
			}
		}

		NewtonJointRow outputRow;
		outputRow.contactIndex = PxU32(problem.addScalarContact(contact, lower, upper));
		outputRow.linear0 = (row.flags & Px1DConstraintFlag::eOUTPUT_FORCE) ? row.linear0 : PxVec3(0.0f);
		outputRow.angular0 = (row.flags & Px1DConstraintFlag::eOUTPUT_FORCE) ? row.angular0 : PxVec3(0.0f);
		output.rows.pushBack(outputRow);
	}
	joint.rowCount = output.rows.size() - joint.firstRow;
	output.joints.pushBack(joint);
}

void writebackNewtonJoints(const NewtonJointRows& rows, const newton::Problem& problem, const newton::Result& result)
{
	const PxU32 jointCount = rows.joints.size();
	for(PxU32 i = 0; i < jointCount; ++i)
	{
		const NewtonJointWriteback& joint = rows.joints[i];
		if(!joint.destination)
		{
			continue;
		}
		PxVec3 linearImpulse(0.0f), angularImpulse(0.0f);
		const PxU32 lastRow = joint.firstRow + joint.rowCount;
		for(PxU32 j = joint.firstRow; j < lastRow; ++j)
		{
			const NewtonJointRow& row = rows.rows[j];
			const PxReal impulse = PxReal(result.impulse[problem.contacts[row.contactIndex].row]);
			linearImpulse += row.linear0 * impulse;
			angularImpulse += row.angular0 * impulse;
		}
		angularImpulse -= joint.body0WorldOffset.cross(linearImpulse);
		joint.destination->linearImpulse = linearImpulse;
		joint.destination->angularImpulse = angularImpulse;
		joint.destination->broken = PxU32(linearImpulse.magnitude() > joint.linearBreakImpulse || angularImpulse.magnitude() > joint.angularBreakImpulse);
	}
}
}
}
