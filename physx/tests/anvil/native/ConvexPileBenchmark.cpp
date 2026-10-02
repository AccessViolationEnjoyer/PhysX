// PEEL's "PileOfLargeConvexes" (ConvexPileScene.h): 5x5 columns of 20 hulls collapse into a pile.
// The rest follows the other benchmark scenes (friction 0.5, 0.5 mm contact offset, speculative
// CCD). MujocoConvexPile.cpp runs the same scene in MuJoCo.
#include "PxPhysicsAPI.h"
#include "ConvexPileScene.h"
#include "NativeProfiler.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace physx;

static void setContactOffset(PxRigidActor& actor, PxReal offset)
{
	PxShape* shape = NULL;
	for(PxU32 i = 0; actor.getShapes(&shape, 1, i); ++i)
		shape->setContactOffset(offset);
}

int main(int argc, char** argv)
{
	if(argc < 2)
	{
		printf("AnvilConvexPile output.csv [steps=600] [anvil|pgs] [workers=8] [columns=5] [layers=20]\n");
		return 1;
	}
	const int steps = argc > 2 ? std::atoi(argv[2]) : 600;
	const PxSolverType::Enum solver = argc > 3 && std::strcmp(argv[3], "pgs") == 0 ? PxSolverType::ePGS : PxSolverType::eANVIL;
	const PxU32 workers = argc > 4 ? PxU32(std::atoi(argv[4])) : 8;
	const int columns = argc > 5 ? std::atoi(argv[5]) : 5;
	const int layers = argc > 6 ? std::atoi(argv[6]) : 20;
	FILE* output = std::fopen(argv[1], "w");
	if(!output || steps < 1 || columns < 1 || layers < 1)
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
	desc.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
	PxScene* scene = physics->createScene(desc);
	if(!scene)
		return 1;
	PxMaterial* material = physics->createMaterial(.5f, .5f, 0.0f);
	// The 0.5 mm contact offset of the other scenes, on every shape, in place of PhysX's 2 cm.
	const PxReal contactOffset = 5e-4f;
	PxRigidStatic* ground = PxCreatePlane(*physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *material);
	setContactOffset(*ground, contactOffset);
	scene->addActor(*ground);

	PxVec3 points[convexPile::pointCount];
	convexPile::hullPoints(&points[0].x);
	PxConvexMeshDesc convexDesc;
	convexDesc.points.count = convexPile::pointCount;
	convexDesc.points.stride = sizeof(PxVec3);
	convexDesc.points.data = points;
	convexDesc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
	const PxCookingParams cookingParams(physics->getTolerancesScale());
	PxConvexMesh* hull = PxCreateConvexMesh(cookingParams, convexDesc, physics->getPhysicsInsertionCallback());
	if(!hull)
		return 1;

	std::vector<PxRigidDynamic*> bodies;
	for(int layer = 0; layer < layers; ++layer)
	{
		for(int row = 0; row < columns; ++row)
		{
			for(int column = 0; column < columns; ++column)
			{
				double place[3];
				convexPile::position(column, row, layer, columns, place);
				const PxVec3 position = PxVec3(PxReal(place[0]), PxReal(place[1]), PxReal(place[2]));
				PxRigidDynamic* body = PxCreateDynamic(*physics, PxTransform(position), PxConvexMeshGeometry(hull), *material, 1.0f);
				// Contacts a step ahead of fast bodies: the offset stays the resting precision.
				body->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
				setContactOffset(*body, contactOffset);
				PxRigidBodyExt::setMassAndUpdateInertia(*body, PxReal(convexPile::mass));
				body->setAngularDamping(0.0f);
				body->setSleepThreshold(0.0f);
				body->setSolverIterationCounts(16, 2);
				scene->addActor(*body);
				bodies.push_back(body);
			}
		}
	}
	NativeSolverProfiler profiler;
	if(solver == PxSolverType::eANVIL)
		PxSetProfilerCallback(&profiler);
	// The profile columns are filled in profile builds (PX_PROFILE), where the SDK's zones exist.
	std::fprintf(output, "step,step_ms,island_wall_ms,solve_wall_ms,prepare_cpu_ms,solve_cpu_ms,islands,rows,iterations,iteration_limits,"
		"narrow_phase_wall_ms,narrow_phase_cpu_ms,broad_phase_wall_ms,broad_phase_cpu_ms,contact_pairs,minimum_y,maximum_y,mean_y,maximum_speed,state_hash\n");
	printf("%s convex pile: %dx%dx%d, %zu hulls of %u vertices, dt=0.01, workers=%u\n",
		solver == PxSolverType::eANVIL ? "Anvil" : "PGS", columns, columns, layers, bodies.size(), hull->getNbVertices(), workers);
	bool finite = true;
	for(int frame = 0; frame < steps; ++frame)
	{
		profiler.reset();
		const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		scene->simulate(.01f);
		scene->fetchResults(true);
		const double stepMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		double minimumY = 1e30, maximumY = -1e30, sumY = 0.0, maximumSpeed = 0.0;
		// FNV-1a over every pose and velocity bit, for exact determinism checks.
		unsigned long long hash = 14695981039346656037ull;
		for(size_t i = 0; i < bodies.size(); ++i)
		{
			const PxTransform pose = bodies[i]->getGlobalPose();
			const PxVec3 linear = bodies[i]->getLinearVelocity(), angular = bodies[i]->getAngularVelocity();
			finite = finite && pose.isFinite() && linear.isFinite();
			minimumY = std::min(minimumY, double(pose.p.y));
			maximumY = std::max(maximumY, double(pose.p.y));
			sumY += double(pose.p.y);
			maximumSpeed = std::max(maximumSpeed, double(linear.magnitude()));
			const float values[13] = { pose.p.x, pose.p.y, pose.p.z, pose.q.x, pose.q.y, pose.q.z, pose.q.w, linear.x, linear.y, linear.z, angular.x, angular.y, angular.z };
			const unsigned char* bytes = reinterpret_cast<const unsigned char*>(values);
			for(size_t b = 0; b < sizeof(values); ++b)
			{
				hash = (hash ^ bytes[b]) * 1099511628211ull;
			}
		}
		PxSimulationStatistics statistics;
		scene->getSimulationStatistics(statistics);
		std::fprintf(output, "%d,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%u,%.9g,%.9g,%.9g,%.9g,%016llx\n", frame + 1,
			stepMs, profiler.wallMilliseconds(), profiler.solveWallMilliseconds(),
			profiler.prepareTime.load() * 1e-6, profiler.solveTime.load() * 1e-6,
			profiler.islandCount.load(), profiler.rows.load(), profiler.iterations.load(), profiler.iterationLimits.load(),
			profiler.narrowPhase.wallMilliseconds(), profiler.narrowPhase.cpuMilliseconds(), profiler.broadPhase.wallMilliseconds(), profiler.broadPhase.cpuMilliseconds(),
			statistics.nbDiscreteContactPairsWithContacts, minimumY, maximumY, sumY / double(bodies.size()), maximumSpeed, hash);
	}
	std::fclose(output);
	PxSetProfilerCallback(NULL);
	scene->release();
	hull->release();
	material->release();
	dispatcher->release();
	physics->release();
	foundation->release();
	return finite ? 0 : 1;
}
