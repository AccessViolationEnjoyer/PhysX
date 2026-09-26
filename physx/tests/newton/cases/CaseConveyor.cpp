// Individual cases on ten long conveyors, simulated by PhysX with PGS or Newton.
#include "PxPhysicsAPI.h"
#include "CaseScene.h"
#include "NativeProfiler.h"
#include <cstdlib>
#include <cstring>

using namespace physx;

// Contact target velocity is actor[0] relative to actor[1]; only belt contacts are modified.
class BeltContacts : public PxContactModifyCallback
{
public:
	virtual void onContactModify(PxContactModifyPair* const pairs, PxU32 count) PX_OVERRIDE
	{
		for(PxU32 i = 0; i < count; ++i)
		{
			const bool beltFirst = pairs[i].shape[0]->getSimulationFilterData().word0 != 0;
			const PxVec3 velocity(beltFirst ? -PxReal(cases::beltSpeed) : PxReal(cases::beltSpeed), 0.0f, 0.0f);
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

int main(int argc, const char* const* argv)
{
	if(argc < 2)
	{
		printf("CaseConveyor output-prefix [steps=2000] [dt=.01] [pgs|newton] [threads=8] [position=4] [velocity=1] [surface-regularization=1e-2] [stiffening-depth=2e-5] [tolerance=1e-8]\n");
		return 1;
	}
	const int steps = argc > 2 ? atoi(argv[2]) : 2000;
	const PxReal timestep = argc > 3 ? PxReal(atof(argv[3])) : 0.01f;
	const bool newton = argc > 4 && std::strcmp(argv[4], "newton") == 0;
	const PxU32 threads = argc > 5 ? PxU32(atoi(argv[5])) : 8;
	const PxU32 positionIterations = argc > 6 ? PxU32(atoi(argv[6])) : 4;
	const PxU32 velocityIterations = argc > 7 ? PxU32(atoi(argv[7])) : 1;
	if(steps * double(timestep) > cases::maximumDuration())
	{
		printf("The cases reach the end of the belts after %.3g s\n", cases::maximumDuration());
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
	description.solverType = newton ? PxSolverType::eNEWTON : PxSolverType::ePGS;
	description.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
	description.newtonSurfaceRegularization = argc > 8 ? PxReal(atof(argv[8])) : description.newtonSurfaceRegularization;
	description.newtonStiffeningDepth = argc > 9 ? PxReal(atof(argv[9])) : description.newtonStiffeningDepth;
	description.newtonTolerance = argc > 10 ? PxReal(atof(argv[10])) : description.newtonTolerance;
	NativeNewtonProfiler profiler;
	if(newton)
		PxSetProfilerCallback(&profiler);
	PxScene* scene = physics->createScene(description);
	PxMaterial* material = physics->createMaterial(PxReal(cases::friction), PxReal(cases::friction), 0.0f);
	for(int lane = 0; lane < cases::conveyorCount; ++lane)
	{
		PxRigidStatic* belt = physics->createRigidStatic(PxTransform(PxVec3(0.0f,
			PxReal(cases::conveyorTop - 0.5 * cases::beltThickness), PxReal(cases::laneZ(lane)))));
		PxShape* shape = physics->createShape(PxBoxGeometry(PxReal(0.5 * cases::conveyorLength),
			PxReal(0.5 * cases::beltThickness), PxReal(0.5 * cases::conveyorWidth)), *material);
		shape->setSimulationFilterData(PxFilterData(1, 0, 0, 0));
		belt->attachShape(*shape);
		shape->release();
		scene->addActor(*belt);
	}
	std::vector<PxRigidDynamic*> actors;
	for(int i = 0; i < cases::caseCount; ++i)
	{
		double position[3];
		cases::casePosition(i, position);
		PxRigidDynamic* actor = PxCreateDynamic(*physics, PxTransform(PxVec3(PxReal(position[0]), PxReal(position[1]), PxReal(position[2]))),
			PxBoxGeometry(PxVec3(PxReal(0.5 * cases::caseSize))), *material, 1.0f);
		PxRigidBodyExt::setMassAndUpdateInertia(*actor, PxReal(cases::caseMass));
		actor->setSolverIterationCounts(positionIterations, velocityIterations);
		actor->setSleepThreshold(0.0f);
		scene->addActor(*actor);
		actors.push_back(actor);
	}
	cases::Recorder recorder(argv[1]);
	if(!recorder.isValid())
		return 1;
	printf("%s: %d cases on %d conveyors, %.6g s steps, %u workers, PGS iterations %u+%u\n", newton ? "Newton" : "PGS",
		cases::caseCount, cases::conveyorCount, double(timestep), threads, positionIterations, velocityIterations);
	std::vector<cases::State> states(static_cast<size_t>(cases::caseCount));
	// Newton's profile zones exist only in profile and checked builds.
	FILE* profile = newton ? cases::openOutput(std::string(argv[1]) + "-profile.csv") : NULL;
	if(profile)
		fprintf(profile, "step,island_wall_ms,solve_wall_ms,prepare_cpu_ms,solve_cpu_ms,islands,rows,iterations,factorizations,rank_updates,line_evaluations\n");
	for(int step = 0; step < steps; ++step)
	{
		profiler.reset();
		const cases::Clock::time_point start = cases::Clock::now();
		scene->simulate(timestep);
		scene->fetchResults(true);
		const double stepMs = cases::elapsed(start);
		PxSimulationStatistics statistics;
		scene->getSimulationStatistics(statistics);
		for(int i = 0; i < cases::caseCount; ++i)
		{
			const PxVec3 position = actors[size_t(i)]->getGlobalPose().p;
			const PxVec3 velocity = actors[size_t(i)]->getLinearVelocity();
			for(int axis = 0; axis < 3; ++axis)
			{
				states[size_t(i)].position[axis] = position[axis];
				states[size_t(i)].velocity[axis] = velocity[axis];
			}
		}
		recorder.record(step + 1, (step + 1) * double(timestep), stepMs, int(statistics.nbDiscreteContactPairsWithContacts),
			newton ? profiler.rows.load() : -1, newton ? profiler.iterations.load() : int(positionIterations + velocityIterations), states);
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
