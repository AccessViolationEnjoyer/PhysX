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

// Reproduces the native stiff-cable validation test. Twenty spherical links
// form a gravity-loaded cantilever with an angular spring at each joint.

#include "PxPhysicsAPI.h"
#include <ctype.h>

using namespace physx;

static const PxU32				gLinkCount = 20;
static PxDefaultAllocator		gAllocator;
static PxDefaultErrorCallback	gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics = NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene = NULL;
static PxMaterial*				gMaterial = NULL;
static PxSolverType::Enum		gSolverType = PxSolverType::eNEWTON;
static bool						gPaused = false;
static bool						gSingleStep = false;

static void createCable()
{
	PxRigidDynamic* previous = NULL;
	for(PxU32 i = 0; i < gLinkCount; ++i)
	{
		PxRigidDynamic* body = PxCreateDynamic(*gPhysics, PxTransform(PxVec3(0.05f * i, 2.0f, 0.0f)),
			PxSphereGeometry(0.025f), *gMaterial, 1.0f);
		PxRigidBodyExt::setMassAndUpdateInertia(*body, 0.1f);
		body->setAngularDamping(0.0f);
		body->setSleepThreshold(0.0f);
		body->setSolverIterationCounts(16, 2);
		body->setName(i == 0 ? "Cable anchor" : i + 1 == gLinkCount ? "Cable tip" : "Cable link");
		gScene->addActor(*body);

		PxD6Joint* joint = PxD6JointCreate(*gPhysics, previous,
			previous ? PxTransform(PxVec3(0.025f, 0.0f, 0.0f)) : body->getGlobalPose(),
			body, previous ? PxTransform(PxVec3(-0.025f, 0.0f, 0.0f)) : PxTransform(PxIdentity));
		if(previous)
		{
			joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
			joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
			joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
			joint->setAngularDriveConfig(PxD6AngularDriveConfig::eSLERP);
			joint->setDrive(PxD6Drive::eSLERP, PxD6JointDrive(5729.57795f, 1145.91559f, PX_MAX_F32, false));
		}
		previous = body;
	}
}

void initPhysics(bool /*interactive*/)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale());
	PxInitExtensions(*gPhysics, NULL);
	gDispatcher = PxDefaultCpuDispatcherCreate(8);

	PxSceneDesc description(gPhysics->getTolerancesScale());
	description.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	description.cpuDispatcher = gDispatcher;
	description.filterShader = PxDefaultSimulationFilterShader;
	description.solverType = gSolverType;
	description.newtonRegularization = 1e-8f;
	description.newtonMaxIterations = 100;
	description.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
	gScene = gPhysics->createScene(description);
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.0f);
	gPaused = false;
	gSingleStep = false;
	createCable();

	printf("%s solver: 20-link stiff cable, 10 ms timestep\n",
		gSolverType == PxSolverType::eNEWTON ? "Newton" : "PGS");
	printf("N: Newton, G: PGS, P: pause, O: single step, R: reset\n");
}

void stepPhysics(bool interactive)
{
	if(interactive && gPaused && !gSingleStep)
	{
		return;
	}

	gSingleStep = false;
	gScene->simulate(0.01f);
	gScene->fetchResults(true);
}

void cleanupPhysics(bool /*interactive*/)
{
	PX_RELEASE(gScene);
	PX_RELEASE(gMaterial);
	PX_RELEASE(gDispatcher);
	PxCloseExtensions();
	PX_RELEASE(gPhysics);
	PX_RELEASE(gFoundation);
}

static void resetScene(PxSolverType::Enum solverType)
{
	cleanupPhysics(true);
	gSolverType = solverType;
	initPhysics(true);
}

void keyPress(unsigned char key, const PxTransform& /*camera*/)
{
	switch(toupper(key))
	{
	case 'N':
		resetScene(PxSolverType::eNEWTON);
		break;
	case 'G':
		resetScene(PxSolverType::ePGS);
		break;
	case 'P':
		gPaused = !gPaused;
		break;
	case 'O':
		gPaused = true;
		gSingleStep = true;
		break;
	case 'R':
		resetScene(gSolverType);
		break;
	}
}

int snippetMain(int, const char* const*)
{
#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	initPhysics(false);
	for(PxU32 i = 0; i < 300; ++i)
	{
		stepPhysics(false);
	}
	cleanupPhysics(false);
#endif
	return 0;
}
