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

#ifdef RENDER_SNIPPET

#include "PxPhysicsAPI.h"
#include "../snippetrender/SnippetCamera.h"
#include "../snippetrender/SnippetRender.h"
#include <cstring>

using namespace physx;

extern void initPhysics(bool interactive);
extern void stepPhysics(bool interactive);
extern void cleanupPhysics(bool interactive);
extern void keyPress(unsigned char key, const PxTransform& camera);

namespace
{
Snippets::Camera* sCamera;

static PxVec3 actorColor(const PxRigidActor& actor)
{
	if(actor.is<PxRigidStatic>())
	{
		return PxVec3(0.25f, 0.25f, 0.25f);
	}

	const char* name = actor.getName();
	if(name && std::strstr(name, "sheet"))
	{
		return PxVec3(0.9f, 0.75f, 0.15f);
	}
	if(name && std::strstr(name, "pallet"))
	{
		return PxVec3(0.45f, 0.25f, 0.1f);
	}
	return PxVec3(0.2f, 0.55f, 0.9f);
}

void renderCallback()
{
	stepPhysics(true);
	Snippets::startRender(sCamera);

	PxScene* scene;
	PxGetPhysics().getScenes(&scene, 1);
	const PxActorTypeFlags actorTypes = PxActorTypeFlag::eRIGID_DYNAMIC | PxActorTypeFlag::eRIGID_STATIC;
	const PxU32 actorCount = scene->getNbActors(actorTypes);
	if(actorCount)
	{
		PxArray<PxRigidActor*> actors(actorCount);
		scene->getActors(actorTypes, reinterpret_cast<PxActor**>(&actors[0]), actorCount);
		for(PxU32 i = 0; i < actorCount; ++i)
		{
			Snippets::renderActors(&actors[i], 1, true, actorColor(*actors[i]));
		}
	}

	Snippets::finishRender();
}

void exitCallback()
{
	delete sCamera;
	cleanupPhysics(true);
}
}

void renderLoop()
{
	sCamera = new Snippets::Camera(PxVec3(8.0f, 7.0f, 16.0f), PxVec3(-0.42f, -0.32f, -0.85f));
	Snippets::setupDefault("PhysX Anvil Pallet Conveyor", sCamera, keyPress, renderCallback, exitCallback);
	initPhysics(true);
	glutMainLoop();
}
#endif
