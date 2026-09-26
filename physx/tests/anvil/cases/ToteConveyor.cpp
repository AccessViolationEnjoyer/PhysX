// Totes of heavy boxes on ten long conveyors, simulated by PhysX with PGS or Anvil.
#include "PxPhysicsAPI.h"
#include "ToteScene.h"
#include "NativeProfiler.h"
#include <cstdlib>
#include <cstring>

using namespace physx;

// Contact target velocity is actor[0] relative to actor[1]; only belt contacts are modified.
class BeltContacts : public PxContactModifyCallback
{
public:
	PxReal speed = PxReal(totes::beltSpeed);

	virtual void onContactModify(PxContactModifyPair* const pairs, PxU32 count) PX_OVERRIDE
	{
		for(PxU32 i = 0; i < count; ++i)
		{
			const bool beltFirst = pairs[i].shape[0]->getSimulationFilterData().word0 != 0;
			const PxVec3 velocity(beltFirst ? -speed : speed, 0.0f, 0.0f);
			const PxU32 contactCount = pairs[i].contacts.size();
			for(PxU32 j = 0; j < contactCount; ++j)
				pairs[i].contacts.setTargetVelocity(j, velocity);
		}
	}
};

static PxFilterFlags filterShader(PxFilterObjectAttributes, PxFilterData data0, PxFilterObjectAttributes, PxFilterData data1, PxPairFlags& flags, const void*, PxU32)
{
	flags = PxPairFlag::eCONTACT_DEFAULT;
	if(data0.word0 || data1.word0)
		flags |= PxPairFlag::eMODIFY_CONTACTS;
	return PxFilterFlag::eDEFAULT;
}

static PxVec3 vector(const double v[3])
{
	return PxVec3(PxReal(v[0]), PxReal(v[1]), PxReal(v[2]));
}

static void setContactOffset(PxRigidActor& actor, PxReal offset)
{
	PxShape* shapes[5];
	const PxU32 count = actor.getShapes(shapes, 5);
	for(PxU32 i = 0; i < count; ++i)
		shapes[i]->setContactOffset(offset);
}

int main(int argc, const char* const* argv)
{
	if(argc < 2)
	{
		printf("ToteConveyor output-prefix [steps=2000] [dt=.01] [pgs|anvil] [threads=8] [position=4] [velocity=1] [surface-regularization=1e-2] [stiffening-depth=2e-5] [tolerance=1e-8] [contact-offset=5e-4] [belt-ramp=1]\n");
		return 1;
	}
	const int steps = argc > 2 ? atoi(argv[2]) : 2000;
	const PxReal timestep = argc > 3 ? PxReal(atof(argv[3])) : 0.01f;
	const bool anvil = argc > 4 && std::strcmp(argv[4], "anvil") == 0;
	const PxU32 threads = argc > 5 ? PxU32(atoi(argv[5])) : 8;
	const PxU32 positionIterations = argc > 6 ? PxU32(atoi(argv[6])) : 4;
	const PxU32 velocityIterations = argc > 7 ? PxU32(atoi(argv[7])) : 1;
	// Seconds for the belts to reach full speed from rest; 0 starts them at full speed.
	const double ramp = argc > 12 ? atof(argv[12]) : cases::beltRamp;
	// Each shape's contact offset; a pair makes contacts within the sum of the two.
	const PxReal contactOffset = PxReal(argc > 11 ? atof(argv[11]) : totes::contactOffset);
	if(steps * double(timestep) > totes::maximumDuration())
	{
		printf("The totes reach the end of the belts after %.3g s\n", totes::maximumDuration());
		return 1;
	}
	PxDefaultAllocator allocator;
	PxDefaultErrorCallback errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(threads);
	BeltContacts belts;
	PxSceneDesc description(physics->getTolerancesScale());
	description.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	description.cpuDispatcher = dispatcher;
	description.filterShader = filterShader;
	description.contactModifyCallback = &belts;
	description.solverType = anvil ? PxSolverType::eANVIL : PxSolverType::ePGS;
	description.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
	description.anvilSurfaceRegularization = argc > 8 ? PxReal(atof(argv[8])) : description.anvilSurfaceRegularization;
	description.anvilStiffeningDepth = argc > 9 ? PxReal(atof(argv[9])) : description.anvilStiffeningDepth;
	description.anvilTolerance = argc > 10 ? PxReal(atof(argv[10])) : description.anvilTolerance;
	NativeSolverProfiler profiler;
	if(anvil)
		PxSetProfilerCallback(&profiler);
	PxScene* scene = physics->createScene(description);
	PxMaterial* material = physics->createMaterial(PxReal(totes::friction), PxReal(totes::friction), 0.0f);
	for(int lane = 0; lane < totes::conveyorCount; ++lane)
	{
		PxRigidStatic* belt = physics->createRigidStatic(PxTransform(PxVec3(0.0f,
			PxReal(totes::conveyorTop - 0.5 * totes::beltThickness), PxReal(totes::laneZ(lane)))));
		PxShape* shape = physics->createShape(PxBoxGeometry(PxReal(0.5 * totes::conveyorLength),
			PxReal(0.5 * totes::beltThickness), PxReal(0.5 * totes::conveyorWidth)), *material);
		shape->setSimulationFilterData(PxFilterData(1, 0, 0, 0));
		belt->attachShape(*shape);
		shape->release();
		setContactOffset(*belt, contactOffset);
		scene->addActor(*belt);
	}
	totes::Part parts[5];
	totes::toteParts(parts);
	std::vector<PxRigidDynamic*> actors;
	for(int i = 0; i < totes::toteCount; ++i)
	{
		double position[3];
		totes::totePosition(i, position);
		PxRigidDynamic* tote = physics->createRigidDynamic(PxTransform(vector(position)));
		for(int part = 0; part < 5; ++part)
		{
			PxShape* shape = physics->createShape(PxBoxGeometry(vector(parts[part].halfExtents)), *material);
			shape->setLocalPose(PxTransform(vector(parts[part].centre)));
			tote->attachShape(*shape);
			shape->release();
		}
		// Uniform density over the five parts.
		PxRigidBodyExt::setMassAndUpdateInertia(*tote, PxReal(totes::toteMass));
		actors.push_back(tote);
		for(int box = 0; box < totes::boxesPerTote; ++box)
		{
			double offset[3];
			totes::boxOffset(box, offset);
			const PxVec3 centre = vector(position) + vector(offset);
			PxRigidDynamic* actor = PxCreateDynamic(*physics, PxTransform(centre), PxBoxGeometry(PxReal(0.5 * totes::boxLength),
				PxReal(0.5 * totes::boxHeight), PxReal(0.5 * totes::boxWidth)), *material, 1.0f);
			PxRigidBodyExt::setMassAndUpdateInertia(*actor, PxReal(totes::boxMass));
			actors.push_back(actor);
		}
	}
	for(size_t i = 0; i < actors.size(); ++i)
	{
		actors[i]->setSolverIterationCounts(positionIterations, velocityIterations);
		actors[i]->setSleepThreshold(0.0f);
		setContactOffset(*actors[i], contactOffset);
		scene->addActor(*actors[i]);
	}
	totes::Recorder recorder(argv[1]);
	if(!recorder.isValid())
		return 1;
	printf("%s: %d totes of %d boxes on %d conveyors, %.6g s steps, %u workers, contact offset %.3g m, PGS iterations %u+%u\n", anvil ? "Anvil" : "PGS",
		totes::toteCount, totes::boxesPerTote, totes::conveyorCount, double(timestep), threads, double(contactOffset), positionIterations, velocityIterations);
	std::vector<totes::State> states(static_cast<size_t>(totes::bodyCount));
	// Anvil's profile zones exist only in profile and checked builds.
	FILE* profile = anvil ? cases::openOutput(std::string(argv[1]) + "-profile.csv") : NULL;
	if(profile)
		fprintf(profile, "step,island_wall_ms,solve_wall_ms,prepare_cpu_ms,solve_cpu_ms,islands,rows,iterations,factorizations,rank_updates,line_evaluations\n");
	for(int step = 0; step < steps; ++step)
	{
		profiler.reset();
		const cases::Clock::time_point start = cases::Clock::now();
		belts.speed = PxReal(totes::beltSpeedAt((step + 1) * double(timestep), ramp));
		scene->simulate(timestep);
		scene->fetchResults(true);
		const double stepMs = cases::elapsed(start);
		PxSimulationStatistics statistics;
		scene->getSimulationStatistics(statistics);
		for(int i = 0; i < totes::bodyCount; ++i)
		{
			const PxTransform pose = actors[size_t(i)]->getGlobalPose();
			const PxVec3 velocity = actors[size_t(i)]->getLinearVelocity();
			totes::State& state = states[size_t(i)];
			for(int axis = 0; axis < 3; ++axis)
			{
				state.position[axis] = pose.p[axis];
				state.velocity[axis] = velocity[axis];
			}
			state.rotation[0] = pose.q.w;
			state.rotation[1] = pose.q.x;
			state.rotation[2] = pose.q.y;
			state.rotation[3] = pose.q.z;
		}
		recorder.record(step + 1, (step + 1) * double(timestep), stepMs, int(statistics.nbDiscreteContactPairsWithContacts),
			anvil ? profiler.rows.load() : -1, anvil ? profiler.iterations.load() : int(positionIterations + velocityIterations), states);
		if(profile)
			fprintf(profile, "%d,%.9g,%.9g,%.9g,%.9g,%d,%d,%d,%d,%d,%d\n", step + 1, profiler.wallMilliseconds(), profiler.solveWallMilliseconds(),
				double(profiler.prepareTime.load()) * 1e-6, double(profiler.solveTime.load()) * 1e-6, profiler.islandCount.load(),
				profiler.rows.load(), profiler.iterations.load(), profiler.factors.load(), profiler.updates.load(), profiler.lineEvaluations.load());
	}
	if(profile)
		fclose(profile);
	recorder.writeFinal(states);
	PxSetProfilerCallback(NULL);
	scene->release();
	material->release();
	dispatcher->release();
	physics->release();
	foundation->release();
	return 0;
}
