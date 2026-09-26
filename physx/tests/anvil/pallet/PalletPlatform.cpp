// One pallet on a kinematic platform shuttling from side to side (PalletPlatform.h),
// simulated by PhysX with PGS or Anvil.
#include "PxPhysicsAPI.h"
#include "PalletPlatform.h"
#include "NativeProfiler.h"
#include <cstdlib>
#include <cstring>

using namespace physx;

static PxFilterFlags filterShader(PxFilterObjectAttributes, PxFilterData, PxFilterObjectAttributes, PxFilterData, PxPairFlags& flags, const void*, PxU32)
{
	flags = PxPairFlag::eCONTACT_DEFAULT;
	return PxFilterFlag::eDEFAULT;
}

int main(int argc, const char* const* argv)
{
	if(argc < 2)
	{
		printf("AnvilPalletPlatform output-prefix [steps=6000] [dt=.01] [pgs|anvil] [threads=8] [position=16] [velocity=2] [contact-offset=5e-4]\n");
		return 1;
	}
	const int steps = argc > 2 ? atoi(argv[2]) : 6000;
	const PxReal timestep = argc > 3 ? PxReal(atof(argv[3])) : 0.01f;
	const bool anvil = argc > 4 && std::strcmp(argv[4], "anvil") == 0;
	const PxU32 threads = argc > 5 ? PxU32(atoi(argv[5])) : 8;
	const PxU32 positionIterations = argc > 6 ? PxU32(atoi(argv[6])) : 16;
	const PxU32 velocityIterations = argc > 7 ? PxU32(atoi(argv[7])) : 2;
	const PxReal contactOffset = PxReal(argc > 8 ? atof(argv[8]) : platform::contactOffset);
	PxDefaultAllocator allocator;
	PxDefaultErrorCallback errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(threads);
	PxSceneDesc description(physics->getTolerancesScale());
	description.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	description.cpuDispatcher = dispatcher;
	description.filterShader = filterShader;
	description.solverType = anvil ? PxSolverType::eANVIL : PxSolverType::ePGS;
	description.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
	NativeSolverProfiler profiler;
	if(anvil)
		PxSetProfilerCallback(&profiler);
	PxScene* scene = physics->createScene(description);
	PxMaterial* material = physics->createMaterial(0.5f, 0.5f, 0.0f);

	// The platform's contacts take its velocity from its kinematic target each step.
	const PxReal platformY = PxReal(platform::top - platform::halfExtents[1]);
	PxRigidDynamic* platformActor = physics->createRigidDynamic(PxTransform(PxVec3(PxReal(platform::startX), platformY, 0.0f)));
	platformActor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
	PxShape* platformShape = physics->createShape(PxBoxGeometry(PxReal(platform::halfExtents[0]), PxReal(platform::halfExtents[1]),
		PxReal(platform::halfExtents[2])), *material);
	platformShape->setContactOffset(contactOffset);
	platformActor->attachShape(*platformShape);
	platformShape->release();
	scene->addActor(*platformActor);

	const std::vector<pallet::Body> bodies = platform::createStack();
	std::vector<PxRigidDynamic*> actors;
	for(size_t i = 0; i < bodies.size(); ++i)
	{
		const pallet::Body& body = bodies[i];
		PxRigidDynamic* actor = physics->createRigidDynamic(PxTransform(PxVec3(PxReal(body.position[0]), PxReal(body.position[1]), PxReal(body.position[2]))));
		PxShape* shape = physics->createShape(PxBoxGeometry(PxReal(body.halfSize[0]), PxReal(body.halfSize[1]), PxReal(body.halfSize[2])), *material);
		shape->setContactOffset(contactOffset);
		actor->attachShape(*shape);
		shape->release();
		PxRigidBodyExt::setMassAndUpdateInertia(*actor, PxReal(body.mass));
		actor->setSolverIterationCounts(positionIterations, velocityIterations);
		actor->setSleepThreshold(0.0f);
		scene->addActor(*actor);
		actors.push_back(actor);
	}
	platform::Recorder recorder(argv[1]);
	if(!recorder.isValid())
		return 1;
	// Anvil's profile zones exist only in profile and checked builds.
	FILE* profile = anvil ? pallet::openOutput(std::string(argv[1]) + "-profile.csv") : NULL;
	if(profile)
		fprintf(profile, "step,island_wall_ms,solve_wall_ms,prepare_cpu_ms,solve_cpu_ms,islands,rows,iterations,iteration_limits,factorizations,rank_updates,line_evaluations\n");
	printf("%s: one pallet of %zu bodies on a platform shuttling %.3g m at up to %.3g m/s and %.3g m/s^2 (%.3g s strokes), %.6g s steps, %u workers, "
		"contact offset %.3g m, PGS iterations %u+%u\n", anvil ? "Anvil" : "PGS", bodies.size(), platform::stroke, platform::maximumSpeed,
		platform::acceleration, platform::strokeDuration(), double(timestep), threads, double(contactOffset), positionIterations, velocityIterations);
	std::vector<pallet::Pose> poses(bodies.size());
	for(int step = 0; step < steps; ++step)
	{
		const double time = (step + 1) * double(timestep);
		platformActor->setKinematicTarget(PxTransform(PxVec3(PxReal(platform::position(time)), platformY, 0.0f)));
		profiler.reset();
		const pallet::Clock::time_point start = pallet::Clock::now();
		scene->simulate(timestep);
		scene->fetchResults(true);
		const double stepMs = pallet::elapsed(start);
		PxSimulationStatistics statistics;
		scene->getSimulationStatistics(statistics);
		for(size_t i = 0; i < bodies.size(); ++i)
		{
			const PxTransform pose = actors[i]->getGlobalPose();
			const PxVec3 velocity = actors[i]->getLinearVelocity();
			const PxMat33 rotation(pose.q);
			for(PxU32 j = 0; j < 3; ++j)
			{
				poses[i].position[j] = pose.p[j];
				poses[i].velocity[j] = velocity[j];
				for(PxU32 k = 0; k < 3; ++k)
					poses[i].rotation[j * 3 + k] = rotation(j, k);
			}
		}
		recorder.record(step + 1, time, stepMs, int(statistics.nbDiscreteContactPairsWithContacts),
			anvil ? profiler.iterations.load() : int(positionIterations + velocityIterations), platformActor->getGlobalPose().p.x,
			platformActor->getLinearVelocity().x, bodies, poses);
		if(profile)
			fprintf(profile, "%d,%.9g,%.9g,%.9g,%.9g,%d,%d,%d,%d,%d,%d,%d\n", step + 1, profiler.wallMilliseconds(), profiler.solveWallMilliseconds(),
				double(profiler.prepareTime.load()) * 1e-6, double(profiler.solveTime.load()) * 1e-6, profiler.islandCount.load(), profiler.rows.load(),
				profiler.iterations.load(), profiler.iterationLimits.load(), profiler.factors.load(), profiler.updates.load(), profiler.lineEvaluations.load());
	}
	if(profile)
		fclose(profile);
	PxSetProfilerCallback(NULL);
	scene->release();
	material->release();
	dispatcher->release();
	physics->release();
	foundation->release();
	return 0;
}
