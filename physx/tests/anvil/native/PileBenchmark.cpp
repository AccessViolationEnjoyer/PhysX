#include "PxPhysicsAPI.h"
#include "NativeProfiler.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace physx;

int main(int argc, char** argv)
{
	if(argc < 2)
	{
		printf("AnvilNativePile output.csv [steps=200] [anvil|pgs] [workers=8] [width=10] [depth=10] [layers=5] [regularization=1e-4] [surface-regularization=1e-2] [stiffening-depth=2e-5]\n");
		return 1;
	}
	const int steps = argc > 2 ? std::atoi(argv[2]) : 200;
	const PxSolverType::Enum solver = argc > 3 && std::strcmp(argv[3], "pgs") == 0 ? PxSolverType::ePGS : PxSolverType::eANVIL;
	const PxU32 workers = argc > 4 ? PxU32(std::atoi(argv[4])) : 8;
	const int width = argc > 5 ? std::atoi(argv[5]) : 10;
	const int depth = argc > 6 ? std::atoi(argv[6]) : 10;
	const int layers = argc > 7 ? std::atoi(argv[7]) : 5;
	FILE* output = std::fopen(argv[1], "w");
	if(!output || steps < 1 || width < 1 || depth < 1 || layers < 1)
		return 1;
	PxDefaultAllocator allocator;
	PxDefaultErrorCallback errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(workers);
	PxSceneDesc desc(physics->getTolerancesScale());
	desc.cpuDispatcher = dispatcher;
	desc.filterShader = PxDefaultSimulationFilterShader;
	desc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	desc.solverType = solver;
	desc.anvilRegularization = argc > 8 ? PxReal(std::atof(argv[8])) : desc.anvilRegularization;
	desc.anvilSurfaceRegularization = argc > 9 ? PxReal(std::atof(argv[9])) : desc.anvilSurfaceRegularization;
	desc.anvilStiffeningDepth = argc > 10 ? PxReal(std::atof(argv[10])) : desc.anvilStiffeningDepth;
	desc.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
	PxScene* scene = physics->createScene(desc);
	if(!scene)
		return 1;
	PxMaterial* material = physics->createMaterial(.5f, .5f, 0.0f);
	scene->addActor(*PxCreatePlane(*physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *material));
	std::vector<PxRigidDynamic*> bodies;
	for(int layer = 0; layer < layers; ++layer)
	{
		// Same staggered 0.3 m cubes as the maintained MuJoCo pile scene.
		// The stagger joins the columns through dynamic-body contacts.
		const PxReal shift = PxReal(layer % 2) * .075f;
		for(int z = 0; z < depth; ++z)
			for(int x = 0; x < width; ++x)
			{
				const PxVec3 position(PxReal((x - (width - 1) * .5) * .3) + shift,
					.15f + PxReal(layer) * .3f, PxReal((z - (depth - 1) * .5) * .3) + shift);
				PxRigidDynamic* body = PxCreateDynamic(*physics, PxTransform(position), PxBoxGeometry(PxVec3(.15f)), *material, 1.0f);
				PxRigidBodyExt::setMassAndUpdateInertia(*body, 1.0f);
				body->setAngularDamping(0.0f);
				body->setSleepThreshold(0.0f);
				body->setSolverIterationCounts(16, 2);
				scene->addActor(*body);
				bodies.push_back(body);
			}
	}
	NativeSolverProfiler profiler;
	if(solver == PxSolverType::eANVIL)
		PxSetProfilerCallback(&profiler);
	std::fprintf(output, "step,step_ms,island_wall_ms,solve_wall_ms,prepare_cpu_ms,solve_cpu_ms,islands,rows,iterations,iteration_limits,contact_pairs,minimum_y,maximum_speed\n");
	printf("%s pile: %dx%dx%d, %zu bodies, dt=0.01, workers=%u, regularization=%.9g, normal offsets\n",
		solver == PxSolverType::eANVIL ? "Anvil" : "PGS", width, depth, layers, bodies.size(), workers, double(desc.anvilRegularization));
	bool finite = true;
	for(int frame = 0; frame < steps; ++frame)
	{
		profiler.reset();
		const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		scene->simulate(.01f);
		scene->fetchResults(true);
		const double stepMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		double minimumY = 1e30, maximumSpeed = 0.0;
		for(size_t i = 0; i < bodies.size(); ++i)
		{
			finite = finite && bodies[i]->getGlobalPose().isFinite() && bodies[i]->getLinearVelocity().isFinite();
			minimumY = std::min(minimumY, double(bodies[i]->getGlobalPose().p.y));
			maximumSpeed = std::max(maximumSpeed, double(bodies[i]->getLinearVelocity().magnitude()));
		}
		PxSimulationStatistics statistics;
		scene->getSimulationStatistics(statistics);
		std::fprintf(output, "%d,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d,%d,%d,%u,%.9g,%.9g\n", frame + 1,
			stepMs, profiler.wallMilliseconds(), profiler.solveWallMilliseconds(),
			profiler.prepareTime.load() * 1e-6, profiler.solveTime.load() * 1e-6,
			profiler.islandCount.load(), profiler.rows.load(), profiler.iterations.load(), profiler.iterationLimits.load(),
			statistics.nbDiscreteContactPairsWithContacts, minimumY, maximumSpeed);
	}
	std::fclose(output);
	PxSetProfilerCallback(NULL);
	scene->release();
	material->release();
	dispatcher->release();
	physics->release();
	foundation->release();
	return finite ? 0 : 1;
}
