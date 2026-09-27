// Cantilevered rods of cubes joined by fixed joints, stressing mass ratios. Cube 0 is kinematic
// and holds the rod horizontally; cube i weighs 0.1 * 2^i kg, so every joint carries a heavier
// rod beyond it through a lighter cube. Rods of 10 and 20 cubes settle under gravity, and a
// 100 kg cube is dropped onto the tip of a 10-cube rod. Each scene reports the tip deflection,
// the largest joint separation and rotation, and the remaining motion.
//
// usage: AnvilFixedJointTests [anvil|pgs] [joint-regularization]   (checks apply to Anvil at its
// default joint regularization; PGS and other regularizations only report)
#include "PxPhysicsAPI.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace physx;

static int failures = 0;

static void check(bool value, const char* message)
{
	if(!value)
	{
		printf("FAIL %s\n", message);
		++failures;
	}
}

static const PxReal cubeSize = 0.1f;
static const PxReal rodHeight = 1.0f;
static const PxReal timestep = 0.01f;

struct RodMetrics
{
	double tipDeflection = 0.0;   // Tip cube below its unloaded height.
	double jointGap = 0.0;        // Largest separation between joined frames.
	double jointAngle = 0.0;      // Largest relative rotation of joined frames, radians.
	double speed = 0.0;           // Fastest cube.
};

struct Rod
{
	PxScene* scene;
	std::vector<PxRigidDynamic*> cubes;
	std::vector<PxFixedJoint*> joints;
	// Step time, excluding the first step.
	double totalMs = 0.0;
	int stepCount = 0;

	Rod(PxPhysics& physics, PxCpuDispatcher& dispatcher, PxMaterial& material, PxSolverType::Enum solver, int count, PxReal jointRegularization)
	{
		PxSceneDesc description(physics.getTolerancesScale());
		if(jointRegularization > 0.0f)
			description.anvilJointRegularization = jointRegularization;
		description.gravity = PxVec3(0.0f, -9.81f, 0.0f);
		description.cpuDispatcher = &dispatcher;
		description.filterShader = PxDefaultSimulationFilterShader;
		description.solverType = solver;
		scene = physics.createScene(description);
		const PxBoxGeometry box(PxVec3(0.5f * cubeSize));
		for(int i = 0; i < count; ++i)
		{
			PxRigidDynamic* cube = PxCreateDynamic(physics, PxTransform(PxVec3(i * cubeSize, rodHeight, 0.0f)), box, material, 1.0f);
			PxRigidBodyExt::setMassAndUpdateInertia(*cube, 0.1f * std::pow(2.0f, PxReal(i)));
			cube->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, i == 0);
			cube->setSleepThreshold(0.0f);
			cube->setSolverIterationCounts(16, 2);
			scene->addActor(*cube);
			if(i > 0)
			{
				// Joined cubes share a face; joints disable collision between them.
				joints.push_back(PxFixedJointCreate(physics, cubes.back(), PxTransform(PxVec3(0.5f * cubeSize, 0.0f, 0.0f)),
					cube, PxTransform(PxVec3(-0.5f * cubeSize, 0.0f, 0.0f))));
			}
			cubes.push_back(cube);
		}
	}

	~Rod()
	{
		scene->release();
	}

	void step(int count)
	{
		for(int i = 0; i < count; ++i)
		{
			const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
			scene->simulate(timestep);
			scene->fetchResults(true);
			if(stepCount++ > 0)
				totalMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		}
	}

	double meanStepUs() const
	{
		return stepCount > 1 ? 1e3 * totalMs / (stepCount - 1) : 0.0;
	}

	RodMetrics measure() const
	{
		RodMetrics metrics;
		metrics.tipDeflection = rodHeight - cubes.back()->getGlobalPose().p.y;
		const PxTransform frame0(PxVec3(0.5f * cubeSize, 0.0f, 0.0f)), frame1(PxVec3(-0.5f * cubeSize, 0.0f, 0.0f));
		for(size_t i = 1; i < cubes.size(); ++i)
		{
			const PxTransform joint0 = cubes[i - 1]->getGlobalPose() * frame0;
			const PxTransform joint1 = cubes[i]->getGlobalPose() * frame1;
			metrics.jointGap = std::max(metrics.jointGap, double((joint0.p - joint1.p).magnitude()));
			metrics.jointAngle = std::max(metrics.jointAngle, double((joint0.q.getConjugate() * joint1.q).getAngle()));
			metrics.speed = std::max(metrics.speed, double(cubes[i]->getLinearVelocity().magnitude()));
		}
		return metrics;
	}
};

static void report(const char* name, const RodMetrics& settled, const RodMetrics& worst, double meanStepUs)
{
	printf("%s mean_step_us=%.2f tip_deflection_mm=%.4f max_joint_gap_mm=%.4f max_joint_angle_deg=%.4f final_speed_mm_s=%.4f"
		" worst_tip_deflection_mm=%.4f worst_joint_gap_mm=%.4f\n", name, meanStepUs, settled.tipDeflection * 1e3, settled.jointGap * 1e3,
		settled.jointAngle * 180.0 / PxPi, settled.speed * 1e3, worst.tipDeflection * 1e3, worst.jointGap * 1e3);
}

static void combine(RodMetrics& worst, const RodMetrics& current)
{
	worst.tipDeflection = std::max(worst.tipDeflection, current.tipDeflection);
	worst.jointGap = std::max(worst.jointGap, current.jointGap);
	worst.jointAngle = std::max(worst.jointAngle, current.jointAngle);
	worst.speed = std::max(worst.speed, current.speed);
}

// A rod settles under its own weight for three seconds.
static RodMetrics testRod(PxPhysics& physics, PxCpuDispatcher& dispatcher, PxMaterial& material, PxSolverType::Enum solver, int count, PxReal jointRegularization, bool checked)
{
	Rod rod(physics, dispatcher, material, solver, count, jointRegularization);
	RodMetrics worst;
	for(int step = 0; step < 300; ++step)
	{
		rod.step(1);
		combine(worst, rod.measure());
	}
	const RodMetrics settled = rod.measure();
	char name[64];
	snprintf(name, sizeof(name), "fixed_rod_%d", count);
	report(name, settled, worst, rod.meanStepUs());
	if(checked)
	{
		// The 20-cube root joint carries about 100 t through a 0.2 kg cube, so it may yield
		// by the joint regularization times that load ratio.
		const double tipLimit = count <= 10 ? 1.0e-5 : 5.0e-3;
		check(settled.tipDeflection < tipLimit, count <= 10 ? "10-cube rod holds its tip within 0.01 mm" : "20-cube rod holds its tip within 5 mm");
		check(worst.jointGap < (count <= 10 ? 1.0e-5 : 5.0e-5), "fixed joints stay closed");
		check(settled.jointAngle < (count <= 10 ? 1.0e-4 : 2.0e-3), "fixed joints stay aligned");
		check(settled.speed < 1.0e-3, "rod settles");
	}
	return settled;
}

// A 100 kg cube falls 0.5 m onto the tip of a settled 10-cube rod.
static void testDrop(PxPhysics& physics, PxCpuDispatcher& dispatcher, PxMaterial& material, PxSolverType::Enum solver, PxReal jointRegularization, bool checked)
{
	Rod rod(physics, dispatcher, material, solver, 10, jointRegularization);
	rod.step(300);
	const RodMetrics unloaded = rod.measure();
	const PxVec3 tip = rod.cubes.back()->getGlobalPose().p;
	const PxReal dropHeight = 0.5f;
	PxRigidDynamic* weight = PxCreateDynamic(physics, PxTransform(tip + PxVec3(0.0f, cubeSize + dropHeight, 0.0f)),
		PxBoxGeometry(PxVec3(0.5f * cubeSize)), material, 1.0f);
	PxRigidBodyExt::setMassAndUpdateInertia(*weight, 100.0f);
	weight->setSleepThreshold(0.0f);
	weight->setSolverIterationCounts(16, 2);
	rod.scene->addActor(*weight);
	RodMetrics worst;
	for(int step = 0; step < 300; ++step)
	{
		rod.step(1);
		combine(worst, rod.measure());
	}
	const RodMetrics settled = rod.measure();
	const PxVec3 onTip = rod.cubes.back()->getGlobalPose().transformInv(weight->getGlobalPose().p);
	printf("fixed_rod_drop mean_step_us=%.2f unloaded_tip_deflection_mm=%.4f peak_tip_deflection_mm=%.4f settled_tip_deflection_mm=%.4f"
		" deflection_change_mm=%.4f max_joint_gap_mm=%.4f max_joint_angle_deg=%.4f worst_joint_gap_mm=%.4f final_speed_mm_s=%.4f"
		" weight_offset_mm=%.3f,%.3f,%.3f\n", rod.meanStepUs(), unloaded.tipDeflection * 1e3, worst.tipDeflection * 1e3, settled.tipDeflection * 1e3,
		(settled.tipDeflection - unloaded.tipDeflection) * 1e3, settled.jointGap * 1e3, settled.jointAngle * 180.0 / PxPi,
		worst.jointGap * 1e3, settled.speed * 1e3, onTip.x * 1e3, onTip.y * 1e3, onTip.z * 1e3);
	if(checked)
	{
		check(worst.tipDeflection < 1.0e-4, "10-cube rod deflects less than 0.1 mm under the dropped 100 kg cube");
		check(settled.tipDeflection < 5.0e-5, "10-cube rod carries the 100 kg cube within 0.05 mm");
		check(worst.jointGap < 1.0e-5, "fixed joints stay closed through the impact");
		check(std::abs(onTip.x) < 5.0e-3 && std::abs(onTip.z) < 5.0e-3 && std::abs(onTip.y - cubeSize) < 1.0e-3, "dropped cube rests on the tip");
		check(settled.speed < 1.0e-3 && weight->getLinearVelocity().magnitude() < 1.0e-3f, "rod and dropped cube settle");
	}
}

int main(int argc, char** argv)
{
	const bool pgs = argc > 1 && std::strcmp(argv[1], "pgs") == 0;
	const PxReal jointRegularization = argc > 2 ? PxReal(std::atof(argv[2])) : 0.0f;
	const bool checked = !pgs && jointRegularization == 0.0f;
	const PxSolverType::Enum solver = pgs ? PxSolverType::ePGS : PxSolverType::eANVIL;
	PxDefaultAllocator allocator;
	PxDefaultErrorCallback errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	PxInitExtensions(*physics, NULL);
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(1);
	PxMaterial* material = physics->createMaterial(0.5f, 0.5f, 0.0f);
	testRod(*physics, *dispatcher, *material, solver, 10, jointRegularization, checked);
	testRod(*physics, *dispatcher, *material, solver, 20, jointRegularization, checked);
	testDrop(*physics, *dispatcher, *material, solver, jointRegularization, checked);
	material->release();
	dispatcher->release();
	PxCloseExtensions();
	physics->release();
	foundation->release();
	printf("Fixed joint rods (%s): %d failures\n", pgs ? "PGS" : "Anvil", failures);
	return failures ? 1 : 0;
}
