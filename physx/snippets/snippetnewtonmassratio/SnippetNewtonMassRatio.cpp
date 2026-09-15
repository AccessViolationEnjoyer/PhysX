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

// Reproduces the native 10000:1 mass-ratio test with a heavy box resting on
// a light box. Newton is selected initially; PGS is available for comparison.

#include "PxPhysicsAPI.h"
#include <ctype.h>

using namespace physx;

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

static PxRigidDynamic* createBox(const PxTransform& pose, PxReal mass, const char* name)
{
	PxRigidDynamic* body = PxCreateDynamic(*gPhysics, pose, PxBoxGeometry(PxVec3(0.5f)), *gMaterial, 1.0f);
	PxRigidBodyExt::setMassAndUpdateInertia(*body, mass);
	body->setLinearDamping(0.0f);
	body->setAngularDamping(0.0f);
	body->setSleepThreshold(0.0f);
	body->setSolverIterationCounts(16, 2);
	body->setName(name);
	gScene->addActor(*body);
	return body;
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

	gScene->addActor(*PxCreatePlane(*gPhysics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *gMaterial));
	createBox(PxTransform(PxVec3(0.0f, 0.5f, 0.0f)), 1.0f, "Light box (1 kg)");
	createBox(PxTransform(PxVec3(0.0f, 1.5f, 0.0f)), 10000.0f, "Heavy box (10000 kg)");
	gPaused = false;
	gSingleStep = false;

	printf("%s solver: 10000:1 mass ratio, 10 ms timestep\n",
		gSolverType == PxSolverType::eNEWTON ? "Newton" : "PGS");
	printf("N: Newton, G: PGS, P: pause, O: single step, R: reset\n");
}

void stepPhysics(bool interactive)
{
	if(interactive && gPaused && !gSingleStep)
		return;

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
		stepPhysics(false);
	cleanupPhysics(false);
#endif
	return 0;
}
