#include "PxPhysicsAPI.h"
#include "../../tests/anvil/pallet/PalletScene.h"
#include <ctype.h>

#ifdef PALLET_CONVEYOR_BENCHMARK
#include "../../tests/anvil/native/NativeProfiler.h"
#include <cstring>
#endif

using namespace physx;

// Contact target velocity is actor[0] relative to actor[1]. Only belt contacts
// are modified; the stacked boxes and sheets use ordinary contact constraints.
class ConveyorContacts : public PxContactModifyCallback
{
public:
	virtual void onContactModify(PxContactModifyPair* const pairs, PxU32 count)
	{
		for(PxU32 i = 0; i < count; ++i)
		{
			const bool beltFirst = pairs[i].shape[0]->getSimulationFilterData().word0 != 0;
			const PxVec3 velocity(beltFirst ? -PxReal(pallet::beltSpeed) : PxReal(pallet::beltSpeed), 0.0f, 0.0f);
			const PxU32 contactCount = pairs[i].contacts.size();
			for(PxU32 j = 0; j < contactCount; ++j)
			{
				pairs[i].contacts.setTargetVelocity(j, velocity);
			}
		}
	}
};

static PxFilterFlags filterShader(PxFilterObjectAttributes, PxFilterData data0, PxFilterObjectAttributes, PxFilterData data1, PxPairFlags& flags, const void*, PxU32)
{
	flags = PxPairFlag::eCONTACT_DEFAULT;
	if(data0.word0 || data1.word0)
	{
		flags |= PxPairFlag::eMODIFY_CONTACTS;
	}
	return PxFilterFlag::eDEFAULT;
}

static void createScene(PxPhysics& physics, PxScene& scene, PxMaterial& material, const std::vector<pallet::Body>& bodies, PxU32 positionIterations, PxU32 velocityIterations, std::vector<PxRigidDynamic*>* actors)
{
	for(int lane = 0; lane < pallet::conveyorCount; ++lane)
	{
		PxRigidStatic* belt = physics.createRigidStatic(PxTransform(PxVec3(0.0f, 0.4f,
			PxReal((lane - 2) * pallet::conveyorPitch))));
		PxShape* shape = physics.createShape(PxBoxGeometry(PxReal(pallet::conveyorLength * 0.5), 0.1f,
			PxReal(pallet::conveyorWidth * 0.5)), material);
		shape->setSimulationFilterData(PxFilterData(1, 0, 0, 0));
		belt->attachShape(*shape);
		shape->release();
		scene.addActor(*belt);
	}

	const size_t bodyCount = bodies.size();
	for(size_t i = 0; i < bodyCount; ++i)
	{
		const pallet::Body& body = bodies[i];
		PxRigidDynamic* actor = physics.createRigidDynamic(PxTransform(PxVec3(PxReal(body.position[0]),
			PxReal(body.position[1]), PxReal(body.position[2]))));
		PxShape* shape = physics.createShape(PxBoxGeometry(PxReal(body.halfSize[0]), PxReal(body.halfSize[1]),
			PxReal(body.halfSize[2])), material);
		actor->attachShape(*shape);
		shape->release();
		PxRigidBodyExt::setMassAndUpdateInertia(*actor, PxReal(body.mass));
		actor->setSolverIterationCounts(positionIterations, velocityIterations);
		actor->setSleepThreshold(0.0f);
		actor->setName(body.name.c_str());
		scene.addActor(*actor);
		if(actors)
		{
			actors->push_back(actor);
		}
	}
}

#ifdef PALLET_CONVEYOR_BENCHMARK
int main(int argc, const char* const* argv)
{
	if(argc < 2)
	{
		printf("SnippetPalletConveyor output-prefix [steps=2000] [dt=.01] [position=16] [velocity=2] [threads=8] [pgs|anvil] [regularization=1e-4] [iterations=100] [friction-corrections=4] [surface-regularization=1e-2] [stiffening-depth=2e-5]\n");
		return 1;
	}
	const int steps = argc > 2 ? atoi(argv[2]) : 2000;
	const PxReal timestep = argc > 3 ? PxReal(atof(argv[3])) : 0.01f;
	const PxU32 positionIterations = argc > 4 ? PxU32(atoi(argv[4])) : 16;
	const PxU32 velocityIterations = argc > 5 ? PxU32(atoi(argv[5])) : 2;
	const PxU32 threads = argc > 6 ? PxU32(atoi(argv[6])) : 8;
	PxDefaultAllocator allocator;
	PxDefaultErrorCallback errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	PxInitExtensions(*physics, NULL);
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(threads);
	ConveyorContacts contactCallback;
	PxSceneDesc description(physics->getTolerancesScale());
	description.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	description.cpuDispatcher = dispatcher;
	description.filterShader = filterShader;
	description.contactModifyCallback = &contactCallback;
	description.solverType = argc > 7 && std::strcmp(argv[7], "anvil") == 0 ? PxSolverType::eANVIL : PxSolverType::ePGS;
	description.anvilRegularization = argc > 8 ? PxReal(atof(argv[8])) : description.anvilRegularization;
	description.anvilMaxIterations = argc > 9 ? PxU32(atoi(argv[9])) : description.anvilMaxIterations;
	description.anvilFrictionCorrections = argc > 10 ? PxU32(atoi(argv[10])) : description.anvilFrictionCorrections;
	description.anvilSurfaceRegularization = argc > 11 ? PxReal(atof(argv[11])) : description.anvilSurfaceRegularization;
	description.anvilStiffeningDepth = argc > 12 ? PxReal(atof(argv[12])) : description.anvilStiffeningDepth;
	description.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
	NativeSolverProfiler profiler;
	if(description.solverType == PxSolverType::eANVIL)
	{
		PxSetProfilerCallback(&profiler);
	}
	PxScene* scene = physics->createScene(description);
	PxMaterial* material = physics->createMaterial(0.5f, 0.5f, 0.0f);
	const std::vector<pallet::Body> bodies = pallet::createBodies();
	std::vector<PxRigidDynamic*> actors;
	createScene(*physics, *scene, *material, bodies, positionIterations, velocityIterations, &actors);
	pallet::Recorder recorder(argv[1], bodies);
	std::vector<pallet::Pose> poses(bodies.size());
	printf("%s: %zu bodies, %u+%u iterations, %.6g s, %u workers; default contact/rest offsets, CCD off\n",
		description.solverType == PxSolverType::eANVIL ? "Anvil" : "PGS", bodies.size(), positionIterations, velocityIterations, double(timestep), threads);
	FILE* profile = description.solverType == PxSolverType::eANVIL ? pallet::openOutput(std::string(argv[1]) + "-profile.csv") : NULL;
	if(profile)
	{
		fprintf(profile, "step,island_wall_ms,solve_wall_ms,prepare_cpu_ms,solve_cpu_ms,islands,rows,iterations,iteration_limits,scaled_gradient,factorizations,rank_updates,line_evaluations\n");
	}
	for(int step = 0; step < steps; ++step)
	{
		profiler.reset();
		const pallet::Clock::time_point start = pallet::Clock::now();
		scene->simulate(timestep);
		scene->fetchResults(true);
		const double stepMs = pallet::elapsed(start);
		PxSimulationStatistics statistics;
		scene->getSimulationStatistics(statistics);
		const size_t actorCount = actors.size();
		for(size_t i = 0; i < actorCount; ++i)
		{
			const PxTransform pose = actors[i]->getGlobalPose();
			const PxMat33 rotation(pose.q);
			const PxVec3 velocity = actors[i]->getLinearVelocity();
			for(PxU32 j = 0; j < 3; ++j)
			{
				poses[i].position[j] = pose.p[j];
				poses[i].velocity[j] = velocity[j];
				for(PxU32 k = 0; k < 3; ++k)
				{
					poses[i].rotation[j * 3 + k] = rotation(j, k);
				}
			}
		}
		if(profile)
		{
			fprintf(profile, "%d,%.9g,%.9g,%.9g,%.9g,%d,%d,%d,%d,%.9g,%d,%d,%d\n", step + 1,
				profiler.wallMilliseconds(), profiler.solveWallMilliseconds(),
				double(profiler.prepareTime.load()) * 1e-6, double(profiler.solveTime.load()) * 1e-6,
				profiler.islandCount.load(), profiler.rows.load(), profiler.iterations.load(), profiler.iterationLimits.load(),
				double(profiler.scaledGradient.load()), profiler.factors.load(), profiler.updates.load(), profiler.lineEvaluations.load());
		}
		recorder.record(step + 1, (step + 1) * double(timestep), stepMs, profiler.wallMilliseconds(),
			int(statistics.nbDiscreteContactPairsWithContacts), -1, -1,
			description.solverType == PxSolverType::eANVIL ? profiler.iterations.load() : int(positionIterations + velocityIterations), bodies, poses);
	}
	if(profile)
	{
		fclose(profile);
	}
	PxSetProfilerCallback(NULL);
	scene->release();
	material->release();
	dispatcher->release();
	PxCloseExtensions();
	physics->release();
	foundation->release();
	return 0;
}
#else
static PxDefaultAllocator		gAllocator;
static PxDefaultErrorCallback	gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics = NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene = NULL;
static PxMaterial*				gMaterial = NULL;
static ConveyorContacts			gContactCallback;
static bool						gPaused = false;
static bool						gSingleStep = false;

void initPhysics(bool /*interactive*/)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale());
	PxInitExtensions(*gPhysics, NULL);
	gDispatcher = PxDefaultCpuDispatcherCreate(8);

	PxSceneDesc description(gPhysics->getTolerancesScale());
	description.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	description.cpuDispatcher = gDispatcher;
	description.filterShader = filterShader;
	description.contactModifyCallback = &gContactCallback;
	description.solverType = PxSolverType::eANVIL;
	description.anvilRegularization = 1e-4f;
	description.anvilMaxIterations = 100;
	description.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
	gScene = gPhysics->createScene(description);
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.0f);
	gPaused = false;
	gSingleStep = false;
	const std::vector<pallet::Body> bodies = pallet::createBodies();
	createScene(*gPhysics, *gScene, *gMaterial, bodies, 16, 2, NULL);

	printf("P: pause, O: single step, R: reset\n");
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

void keyPress(unsigned char key, const PxTransform& /*camera*/)
{
	switch(toupper(key))
	{
	case 'P':
		gPaused = !gPaused;
		break;
	case 'O':
		gPaused = true;
		gSingleStep = true;
		break;
	case 'R':
		cleanupPhysics(true);
		initPhysics(true);
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
	for(PxU32 i = 0; i < 2000; ++i)
	{
		stepPhysics(false);
	}
	cleanupPhysics(false);
#endif
	return 0;
}
#endif

