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

// Places identical boxes on ramps spanning the static-friction threshold.
// With a coefficient of 0.5, sliding begins above atan(0.5) = 26.6 degrees.

#include "PxPhysicsAPI.h"
#include <ctype.h>

using namespace physx;

static const PxU32				gRampCount = 6;
static const PxReal				gRampAngles[gRampCount] = { 10.0f, 20.0f, 25.0f, 28.0f, 35.0f, 45.0f };
static const char* const		gBoxNames[gRampCount] =
{
	"Hold 10 degrees", "Hold 20 degrees", "Hold 25 degrees",
	"Slide 28 degrees", "Slide 35 degrees", "Slide 45 degrees"
};
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

static void createRamp(PxU32 index)
{
	const PxReal radians = gRampAngles[index] * PxPi / 180.0f;
	const PxReal halfLength = 2.5f;
	const PxReal rampX = (PxReal(index) - PxReal(gRampCount - 1) * 0.5f) * 1.5f;
	const PxQuat rotation(-radians, PxVec3(1.0f, 0.0f, 0.0f));
	const PxTransform rampPose(PxVec3(rampX, 0.15f + halfLength * PxSin(radians), 0.0f), rotation);
	PxRigidStatic* ramp = PxCreateStatic(*gPhysics, rampPose,
		PxBoxGeometry(PxVec3(0.6f, 0.1f, halfLength)), *gMaterial);
	ramp->setName("Ramp");
	gScene->addActor(*ramp);

	// Start in contact so the box has not acquired sliding velocity before static friction is solved.
	const PxTransform boxPose = rampPose * PxTransform(PxVec3(0.0f, 0.35f, 1.5f));
	PxRigidDynamic* box = PxCreateDynamic(*gPhysics, boxPose,
		PxBoxGeometry(PxVec3(0.35f, 0.25f, 0.35f)), *gMaterial, 1.0f);
	PxRigidBodyExt::setMassAndUpdateInertia(*box, 1.0f);
	box->setLinearDamping(0.0f);
	box->setAngularDamping(0.0f);
	box->setSleepThreshold(0.0f);
	box->setSolverIterationCounts(16, 2);
	box->setName(gBoxNames[index]);
	gScene->addActor(*box);
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
	description.newtonRegularization = 1e-4f;
	description.newtonMaxIterations = 100;
	description.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
	gScene = gPhysics->createScene(description);
	gMaterial = gPhysics->createMaterial(0.5f, 0.4f, 0.0f);

	gScene->addActor(*PxCreatePlane(*gPhysics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *gMaterial));
	for(PxU32 i = 0; i < gRampCount; ++i)
		createRamp(i);
	gPaused = false;
	gSingleStep = false;

	printf("%s solver: incline friction comparison, 10 ms timestep\n",
		gSolverType == PxSolverType::eNEWTON ? "Newton" : "PGS");
	printf("Static friction = 0.5; theoretical breakaway angle = %.3f degrees\n",
		double(PxAtan(0.5f) * 180.0f / PxPi));
	printf("Ramps from left to right: 10, 20, 25, 28, 35, 45 degrees\n");
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
