#include "PxPhysicsAPI.h"
#include "NativeProfiler.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace physx;

static int failures = 0;
static NativeSolverProfiler profiler;

static void check(bool value, const char* message)
{
	if(!value)
	{
		printf("FAIL %s\n", message);
		++failures;
	}
}

class TestErrors : public PxErrorCallback
{
public:
	int count;
	bool expected;
	TestErrors() : count(0), expected(false) {}
	virtual void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) PX_OVERRIDE
	{
		if(code == PxErrorCode::eDEBUG_INFO || code == PxErrorCode::eDEBUG_WARNING || code == PxErrorCode::ePERF_WARNING)
			return;
		++count;
		if(!expected)
			printf("PHYSX ERROR %s (%s:%d)\n", message, file, line);
	}
};

struct TestContext
{
	PxPhysics& physics;
	PxDefaultCpuDispatcher& dispatcher;
	PxSolverType::Enum solver;
	PxReal regularization;

	PxScene* scene(PxVec3 gravity = PxVec3(0.0f, -9.81f, 0.0f))
	{
		PxSceneDesc desc(physics.getTolerancesScale());
		desc.cpuDispatcher = &dispatcher;
		desc.filterShader = PxDefaultSimulationFilterShader;
		desc.gravity = gravity;
		desc.solverType = solver;
		desc.anvilRegularization = regularization;
		desc.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
		return physics.createScene(desc);
	}

	PxRigidDynamic* box(PxScene& scene, PxMaterial& material, const PxTransform& pose, PxVec3 halfSize = PxVec3(0.5f), PxReal mass = 1.0f)
	{
		PxRigidDynamic* body = PxCreateDynamic(physics, pose, PxBoxGeometry(halfSize), material, 1.0f);
		PxRigidBodyExt::setMassAndUpdateInertia(*body, mass);
		body->setLinearDamping(0.0f);
		body->setAngularDamping(0.0f);
		body->setSleepThreshold(0.0f);
		body->setSolverIterationCounts(16, 2);
		scene.addActor(*body);
		return body;
	}
};

static void step(PxScene& scene, int count = 1, PxReal dt = 0.01f)
{
	for(int i = 0; i < count; ++i)
	{
		scene.simulate(dt);
		scene.fetchResults(true);
	}
}

static void freeFall(TestContext& context, PxMaterial& material)
{
	PxScene* scene = context.scene();
	PxRigidDynamic* body = context.box(*scene, material, PxTransform(PxVec3(0.0f, 10.0f, 0.0f)));
	step(*scene, 20);
	const double expectedVelocity = -9.81 * .2;
	const double expectedPosition = 10.0 - 9.81 * .01 * .01 * 20.0 * 21.0 / 2.0;
	printf("free_fall y=%.9g vy=%.9g\n", double(body->getGlobalPose().p.y), double(body->getLinearVelocity().y));
	check(std::abs(body->getLinearVelocity().y - expectedVelocity) < 2e-5, "free fall velocity applies gravity once");
	check(std::abs(body->getGlobalPose().p.y - expectedPosition) < 2e-5, "free fall pose integration");
	scene->release();
}

static void spring(TestContext& context, PxMaterial& material, bool capped)
{
	PxScene* scene = context.scene(PxVec3(0.0f));
	PxRigidDynamic* body = context.box(*scene, material, PxTransform(PxVec3(1.0f, 0.0f, 0.0f)));
	PxD6Joint* joint = PxD6JointCreate(context.physics, NULL, PxTransform(PxIdentity), body, PxTransform(PxIdentity));
	joint->setMotion(PxD6Axis::eX, PxD6Motion::eFREE);
	joint->setConstraintFlag(PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);
	const double stiffness = 100000.0;
	const double damping = 2.0 * std::sqrt(stiffness);
	joint->setDrive(PxD6Drive::eX, PxD6JointDrive(PxReal(stiffness), PxReal(damping), capped ? 1.0f : PX_MAX_F32, false));
	double x = 1.0, velocity = 0.0, maxError = 0.0;
	for(int i = 0; i < 100; ++i)
	{
		const double delta = (-.01 * stiffness * x - (.01 * damping + .0001 * stiffness) * velocity) /
			(1.0 + .01 * damping + .0001 * stiffness);
		velocity += capped ? std::max(-.01, std::min(.01, delta)) : delta;
		x += .01 * velocity;
		step(*scene);
		maxError = std::max(maxError, std::abs(double(body->getGlobalPose().p.x) - x));
		check(body->getGlobalPose().isFinite(), "spring finite pose");
	}
	printf("%s max_position_error=%.9g x=%.9g vx=%.9g\n", capped ? "bounded_spring" : "critical_spring",
		maxError, double(body->getGlobalPose().p.x), double(body->getLinearVelocity().x));
	check(maxError < 3e-4, "spring matches implicit integration including force cap");
	scene->release();
}

static void limits(TestContext& context, PxMaterial& material)
{
	PxScene* scene = context.scene(PxVec3(0.0f));
	PxRigidDynamic* body = context.box(*scene, material, PxTransform(PxIdentity));
	PxD6Joint* joint = PxD6JointCreate(context.physics, NULL, PxTransform(PxIdentity), body, PxTransform(PxIdentity));
	joint->setMotion(PxD6Axis::eX, PxD6Motion::eLIMITED);
	joint->setLinearLimit(PxD6Axis::eX, PxJointLinearLimitPair(context.physics.getTolerancesScale(), -0.2f, 0.2f));
	body->setLinearVelocity(PxVec3(2.0f, 0.0f, 0.0f));
	double maximum = 0.0;
	for(int i = 0; i < 100; ++i)
	{
		step(*scene);
		maximum = std::max(maximum, double(body->getGlobalPose().p.x));
	}
	printf("hard_limit maximum_x=%.9g\n", maximum);
	check(maximum <= .201, "hard prismatic limit remains bounded");
	scene->release();
}

static void restingBox(TestContext& context, PxMaterial& material)
{
	PxScene* scene = context.scene();
	scene->addActor(*PxCreatePlane(context.physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), material));
	PxRigidDynamic* body = context.box(*scene, material, PxTransform(PxVec3(0.0f, .49f, 0.0f)));
	step(*scene, 300);
	printf("resting_box y=%.9g vy=%.9g\n", double(body->getGlobalPose().p.y), double(body->getLinearVelocity().y));
	check(std::abs(body->getGlobalPose().p.y - .5) < .003, "resting box maintains support");
	check(std::abs(body->getLinearVelocity().y) < .005, "resting box settles without bias energy");
	scene->release();
}

static void massRatio(TestContext& context, PxMaterial& material)
{
	PxScene* scene = context.scene();
	scene->addActor(*PxCreatePlane(context.physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), material));
	PxRigidDynamic* light = context.box(*scene, material, PxTransform(PxVec3(0.0f, .5f, 0.0f)));
	PxRigidDynamic* heavy = context.box(*scene, material, PxTransform(PxVec3(0.0f, 1.5f, 0.0f)), PxVec3(.5f), 10000.0f);
	step(*scene, 300);
	printf("mass_ratio_10000 light_y=%.9g heavy_y=%.9g heavy_vy=%.9g\n", double(light->getGlobalPose().p.y),
		double(heavy->getGlobalPose().p.y), double(heavy->getLinearVelocity().y));
	check(light->getGlobalPose().isFinite() && heavy->getGlobalPose().isFinite(), "mass ratio finite");
	if(context.solver == PxSolverType::eANVIL)
	{
		check(light->getGlobalPose().p.y > .48f && heavy->getGlobalPose().p.y > 1.46f, "Anvil supports heavy body on light body");
		check(std::abs(heavy->getLinearVelocity().y) < .03, "mass ratio reaches supported equilibrium");
	}
	scene->release();
}

static void breakage(TestContext& context, PxMaterial& material)
{
	PxScene* scene = context.scene();
	PxRigidDynamic* body = context.box(*scene, material, PxTransform(PxIdentity), PxVec3(.5f), 2.0f);
	PxFixedJoint* joint = PxFixedJointCreate(context.physics, NULL, PxTransform(PxIdentity), body, PxTransform(PxIdentity));
	joint->setBreakForce(1.0f, PX_MAX_F32);
	step(*scene);
	check(bool(joint->getConstraintFlags() & PxConstraintFlag::eBROKEN), "joint impulse writeback triggers breakage");
	printf("breakage broken=%d\n", (joint->getConstraintFlags() & PxConstraintFlag::eBROKEN) ? 1 : 0);
	scene->release();
}

static void locksAndWarmStart(TestContext& context, PxMaterial& material)
{
	PxScene* scene = context.scene();
	PxRigidDynamic* body = context.box(*scene, material,
		PxTransform(PxVec3(0.0f, 2.0f, 0.0f), PxQuat(.7f, PxVec3(1.0f, 2.0f, 3.0f).getNormalized())), PxVec3(.2f, .4f, .8f));
	body->setRigidDynamicLockFlags(PxRigidDynamicLockFlag::eLOCK_LINEAR_Y | PxRigidDynamicLockFlag::eLOCK_ANGULAR_X);
	const PxReal y = body->getGlobalPose().p.y;
	for(int i = 0; i < 100; ++i)
	{
		body->addTorque(PxVec3(5.0f, 2.0f, 0.0f));
		step(*scene);
		check(std::abs(body->getGlobalPose().p.y - y) < 1e-6f, "locked translation remains fixed");
		if(context.solver == PxSolverType::eANVIL)
			check(std::abs(body->getAngularVelocity().x) < 1e-5f, "rotated anisotropic angular lock");
	}
	body->release();
	step(*scene);
	body = context.box(*scene, material, PxTransform(PxVec3(0.0f, 20.0f, 0.0f)), PxVec3(.5f), 3.0f);
	step(*scene);
	check(std::abs(body->getLinearVelocity().y + .0981f) < 1e-5f, "recycled node does not inherit warm impulse");
	body->setGlobalPose(PxTransform(PxVec3(0.0f, 40.0f, 0.0f)));
	body->setLinearVelocity(PxVec3(0.0f));
	step(*scene);
	check(std::abs(body->getLinearVelocity().y + .0981f) < 1e-5f, "teleport resets warm start");
	printf("locks_and_warm_start y=%.9g vy=%.9g\n", double(body->getGlobalPose().p.y), double(body->getLinearVelocity().y));
	scene->release();
}

static void cable(TestContext& context, PxMaterial& material)
{
	PxScene* scene = context.scene();
	const PxU32 count = 20;
	PxRigidDynamic* previous = NULL;
	PxRigidDynamic* tip = NULL;
	for(PxU32 i = 0; i < count; ++i)
	{
		PxRigidDynamic* body = PxCreateDynamic(context.physics, PxTransform(PxVec3(.05f * i, 2.0f, 0.0f)),
			PxSphereGeometry(.025f), material, 1.0f);
		PxRigidBodyExt::setMassAndUpdateInertia(*body, .1f);
		body->setAngularDamping(0.0f);
		body->setSleepThreshold(0.0f);
		body->setSolverIterationCounts(16, 2);
		scene->addActor(*body);
		PxD6Joint* joint = PxD6JointCreate(context.physics, previous,
			previous ? PxTransform(PxVec3(.025f, 0.0f, 0.0f)) : body->getGlobalPose(),
			body, previous ? PxTransform(PxVec3(-.025f, 0.0f, 0.0f)) : PxTransform(PxIdentity));
		if(previous)
		{
			joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
			joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
			joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
			joint->setAngularDriveConfig(PxD6AngularDriveConfig::eSLERP);
			joint->setDrive(PxD6Drive::eSLERP, PxD6JointDrive(5729.57795f, 1145.91559f, PX_MAX_F32, false));
		}
		previous = tip = body;
	}
	step(*scene, 300);
	printf("stiff_cable tip=(%.9g,%.9g,%.9g)\n", double(tip->getGlobalPose().p.x), double(tip->getGlobalPose().p.y), double(tip->getGlobalPose().p.z));
	check(tip->getGlobalPose().isFinite(), "stiff cable finite");
	if(context.solver == PxSolverType::eANVIL)
	{
		// D6 SLERP uses quaternion-vector error and half-angle Jacobian axes. Its
		// small-angle force-drive stiffness is k/4. Sum the bending of each hinge
		// in this discrete, gravity-loaded cantilever (the first body is fixed).
		double bendingSum = 0.0;
		for(PxU32 n = 1; n < count; ++n)
			bendingSum += double(n) * n * (double(n) - 0.5);
		const double expectedSag = .1 * 9.81 * .05 * .05 * bendingSum / (2.0 * (5729.57795 / 4.0));
		printf("stiff_cable expected_small_angle_sag=%.9g measured_sag=%.9g\n", expectedSag, 2.0 - tip->getGlobalPose().p.y);
		check(std::abs(2.0 - tip->getGlobalPose().p.y - expectedSag) < .001,
			"stiff cable matches the discrete cantilever deflection");
	}
	scene->release();
}

int main(int argc, char** argv)
{
	const char* selection = argc > 1 ? argv[1] : "all";
	const PxSolverType::Enum solver = argc > 2 && std::strcmp(argv[2], "pgs") == 0 ? PxSolverType::ePGS :
		argc > 2 && std::strcmp(argv[2], "tgs") == 0 ? PxSolverType::eTGS : PxSolverType::eANVIL;
	const PxU32 threads = argc > 3 ? PxU32(std::atoi(argv[3])) : 8;
	PxDefaultAllocator allocator;
	TestErrors errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	PxInitExtensions(*physics, NULL);
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(threads);
	PxMaterial* material = physics->createMaterial(.5f, .5f, 0.0f);
	TestContext context = {*physics, *dispatcher, solver, argc > 4 ? PxReal(std::atof(argv[4])) : 1e-8f};
	PxSetProfilerCallback(&profiler);
	const bool all = std::strcmp(selection, "all") == 0;
	if(all || std::strcmp(selection, "free") == 0)
	{
		freeFall(context, *material);
	}
	if(all || std::strcmp(selection, "spring") == 0)
	{
		spring(context, *material, false);
	}
	if(all || std::strcmp(selection, "cap") == 0)
	{
		spring(context, *material, true);
	}
	if(all || std::strcmp(selection, "limit") == 0)
	{
		limits(context, *material);
	}
	if(all || std::strcmp(selection, "box") == 0)
	{
		restingBox(context, *material);
	}
	if(all || std::strcmp(selection, "mass") == 0)
	{
		massRatio(context, *material);
	}
	if(all || std::strcmp(selection, "break") == 0)
	{
		breakage(context, *material);
	}
	if(all || std::strcmp(selection, "warm") == 0)
	{
		locksAndWarmStart(context, *material);
	}
	if(all || std::strcmp(selection, "cable") == 0)
	{
		cable(context, *material);
	}
	PxSetProfilerCallback(NULL);
	const int islandCount = profiler.islandCount.load();
	const double inverseIslandCount = islandCount ? 1.0 / islandCount : 0.0;
	printf("Anvil profile islands=%d mean_iterations=%.9g min_iterations=%d max_iterations=%d mean_island_ms=%.9g mean_prepare_ms=%.9g mean_solve_ms=%.9g iteration_limits=%d\n",
		islandCount, profiler.iterations.load() * inverseIslandCount,
		islandCount ? profiler.minimumIterations.load() : 0, profiler.maximumIterations.load(),
		double(profiler.islandTime.load()) * 1e-6 * inverseIslandCount,
		double(profiler.prepareTime.load()) * 1e-6 * inverseIslandCount,
		double(profiler.solveTime.load()) * 1e-6 * inverseIslandCount, profiler.iterationLimits.load());
	material->release();
	dispatcher->release();
	PxCloseExtensions();
	physics->release();
	foundation->release();
	printf("Native tests: failures=%d engine_errors=%d\n", failures, errors.count);
	return failures || errors.count ? 1 : 0;
}
