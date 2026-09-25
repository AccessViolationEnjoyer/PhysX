#include "PxPhysicsAPI.h"
#include "NativeProfiler.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

using namespace physx;

// Sliding scenes for the pyramid-friction dilatancy correction. Each sliding contact
// should stay on its support surface: the hop is the largest gap that opens between a
// body's lowest corner and the surface it slides on, and the normal speed is the
// largest relative speed across that surface.
namespace
{
const PxReal TIMESTEP = 0.01f;
const PxReal GRAVITY = 9.81f;
int failures = 0;
NativeNewtonProfiler profiler;

void check(bool condition, const char* message)
{
	if(!condition)
	{
		std::printf("FAIL %s\n", message);
		++failures;
	}
}

struct Surface
{
	// A support plane, optionally attached to a moving body.
	const PxRigidActor* body;
	PxVec3 localPoint;
	PxVec3 localNormal;

	void world(PxVec3& point, PxVec3& normal) const
	{
		const PxTransform pose = body ? body->getGlobalPose() : PxTransform(PxIdentity);
		point = pose.transform(localPoint);
		normal = pose.q.rotate(localNormal);
	}

	PxVec3 velocityAt(const PxVec3& point) const
	{
		if(!body || !body->is<PxRigidDynamic>())
		{
			return PxVec3(0.0f);
		}
		const PxRigidDynamic& dynamic = *body->is<PxRigidDynamic>();
		const PxVec3 arm = point - dynamic.getGlobalPose().p;
		return dynamic.getLinearVelocity() + dynamic.getAngularVelocity().cross(arm);
	}
};

struct Slider
{
	PxRigidDynamic* body;
	PxTransform shapePose; // Box shape relative to the body.
	PxVec3 halfExtents;
	Surface surface;
};

struct Metrics
{
	double maximumGap = 0.0;
	double minimumGap = 0.0;
	double maximumNormalSpeed = 0.0;
	double stepMs = 0.0;
	double solveMs = 0.0;
	double iterations = 0.0;
	double factorizations = 0.0;
	int frames = 0;

	void sample(const Slider& slider)
	{
		PxVec3 point, normal;
		slider.surface.world(point, normal);
		const PxTransform pose = slider.body->getGlobalPose() * slider.shapePose;
		double lowest = PX_MAX_F64;
		for(int corner = 0; corner < 8; ++corner)
		{
			const PxVec3 local((corner & 1 ? 1.0f : -1.0f) * slider.halfExtents.x, (corner & 2 ? 1.0f : -1.0f) * slider.halfExtents.y,
				(corner & 4 ? 1.0f : -1.0f) * slider.halfExtents.z);
			lowest = std::min(lowest, double(normal.dot(pose.transform(local) - point)));
		}
		maximumGap = std::max(maximumGap, lowest);
		minimumGap = std::min(minimumGap, lowest);
		const PxVec3 relative = slider.body->getLinearVelocity() - slider.surface.velocityAt(pose.p);
		maximumNormalSpeed = std::max(maximumNormalSpeed, std::abs(double(normal.dot(relative))));
	}
};

struct Context
{
	PxPhysics& physics;
	PxDefaultCpuDispatcher& dispatcher;

	PxScene* createScene()
	{
		PxSceneDesc desc(physics.getTolerancesScale());
		desc.cpuDispatcher = &dispatcher;
		desc.filterShader = PxDefaultSimulationFilterShader;
		desc.gravity = PxVec3(0.0f, -GRAVITY, 0.0f);
		desc.solverType = PxSolverType::eNEWTON;
		desc.newtonMaxIterations = 100;
		desc.newtonTolerance = 1e-8f;
		desc.newtonRegularization = 1e-4f;
		desc.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
		PxScene* scene = physics.createScene(desc);
		check(scene != NULL, "sliding scene creation");
		return scene;
	}

	PxRigidDynamic* box(PxScene& scene, PxMaterial& material, const PxVec3& halfExtents, const PxTransform& pose, PxReal mass)
	{
		PxRigidDynamic* actor = PxCreateDynamic(physics, pose, PxBoxGeometry(halfExtents), material, 1.0f);
		PxRigidBodyExt::setMassAndUpdateInertia(*actor, mass);
		actor->setLinearDamping(0.0f);
		actor->setAngularDamping(0.0f);
		actor->setSleepThreshold(0.0f);
		scene.addActor(*actor);
		return actor;
	}
};

// An open container: a base plus four walls on one body, with a 2 x 2 x layers load.
struct Container
{
	PxRigidDynamic* body;
	PxVec3 baseHalf;
	std::vector<PxRigidDynamic*> load;
	PxVec3 loadHalf;
};

Container createContainer(Context& context, PxScene& scene, PxMaterial& material, const PxTransform& pose, int layers)
{
	const PxReal wall = 0.02f, height = 0.25f;
	const PxVec3 inner(0.45f, 0.0f, 0.35f);
	Container container;
	container.baseHalf = PxVec3(inner.x + 2.0f * wall, wall, inner.z + 2.0f * wall);
	container.body = context.physics.createRigidDynamic(pose);
	const auto attach = [&](const PxVec3& half, const PxVec3& offset)
	{
		PxShape* shape = context.physics.createShape(PxBoxGeometry(half), material);
		shape->setLocalPose(PxTransform(offset));
		container.body->attachShape(*shape);
		shape->release();
	};
	attach(container.baseHalf, PxVec3(0.0f, container.baseHalf.y, 0.0f));
	const PxReal wallY = 2.0f * wall + height;
	attach(PxVec3(wall, height, inner.z + 2.0f * wall), PxVec3(inner.x + wall, wallY, 0.0f));
	attach(PxVec3(wall, height, inner.z + 2.0f * wall), PxVec3(-inner.x - wall, wallY, 0.0f));
	attach(PxVec3(inner.x, height, wall), PxVec3(0.0f, wallY, inner.z + wall));
	attach(PxVec3(inner.x, height, wall), PxVec3(0.0f, wallY, -inner.z - wall));
	PxRigidBodyExt::setMassAndUpdateInertia(*container.body, 10.0f);
	container.body->setLinearDamping(0.0f);
	container.body->setAngularDamping(0.0f);
	container.body->setSleepThreshold(0.0f);
	scene.addActor(*container.body);
	// Boxes leave clearance to the walls so they can slide inside the container.
	container.loadHalf = PxVec3(0.2f, 0.08f, 0.15f);
	for(int layer = 0; layer < layers; ++layer)
	{
		for(int i = 0; i < 4; ++i)
		{
			const PxVec3 offset((i & 1 ? 1.0f : -1.0f) * 0.22f, 2.0f * container.baseHalf.y + container.loadHalf.y * (2 * layer + 1), (i & 2 ? 1.0f : -1.0f) * 0.17f);
			container.load.push_back(context.box(scene, material, container.loadHalf, pose * PxTransform(offset), 1.0f));
		}
	}
	return container;
}

// Container floor as a support surface for the bottom load layer.
Surface floorOf(const Container& container)
{
	const Surface surface = { container.body, PxVec3(0.0f, 2.0f * container.baseHalf.y, 0.0f), PxVec3(0.0f, 1.0f, 0.0f) };
	return surface;
}

Metrics run(PxScene& scene, std::vector<Slider>& sliders, int frames, int settleFrames, const std::function<void(int)>& beforeStep = nullptr)
{
	Metrics metrics;
	double totalMs = 0.0;
	for(int frame = 0; frame < frames; ++frame)
	{
		if(beforeStep)
		{
			beforeStep(frame);
		}
		profiler.reset();
		const auto start = std::chrono::steady_clock::now();
		scene.simulate(TIMESTEP);
		scene.fetchResults(true);
		totalMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		metrics.solveMs += double(profiler.solveTime.load()) * 1e-6;
		metrics.iterations += profiler.iterations.load();
		metrics.factorizations += profiler.factors.load();
		if(frame >= settleFrames)
		{
			for(const Slider& slider : sliders)
			{
				metrics.sample(slider);
			}
		}
	}
	metrics.frames = frames;
	metrics.stepMs = totalMs / frames;
	metrics.solveMs /= frames;
	metrics.iterations /= frames;
	metrics.factorizations /= frames;
	return metrics;
}

void report(const char* name, const Metrics& metrics, double extra, const char* extraName)
{
	std::printf("%s max_gap_mm=%.4f min_gap_mm=%.4f max_normal_speed_mm_s=%.3f %s=%.5f solve_ms=%.4f step_ms=%.4f iterations=%.2f factorizations=%.2f\n", name, metrics.maximumGap * 1e3,
		metrics.minimumGap * 1e3, metrics.maximumNormalSpeed * 1e3, extraName, extra, metrics.solveMs, metrics.stepMs, metrics.iterations, metrics.factorizations);
}

// Tolerances allow contact compliance, not a visible hop.
void checkContact(const char* name, const Metrics& metrics)
{
	char message[256];
	std::snprintf(message, sizeof(message), "%s: sliding bodies stay on their support (gap below 0.5 mm)", name);
	check(metrics.maximumGap < 0.5e-3, message);
	std::snprintf(message, sizeof(message), "%s: sliding bodies do not bounce (normal speed below 20 mm/s)", name);
	check(metrics.maximumNormalSpeed < 20e-3, message);
	std::snprintf(message, sizeof(message), "%s: sliding bodies do not sink (penetration below 2 mm)", name);
	check(metrics.minimumGap > -2e-3, message);
}

const PxReal RAMP_ANGLE = 25.0f * PxPi / 180.0f;
const PxReal RAMP_FRICTION = 0.3f;

PxQuat rampRotation()
{
	// The surface normal leans toward +x, so bodies slide in +x.
	return PxQuat(-RAMP_ANGLE, PxVec3(0.0f, 0.0f, 1.0f));
}

void addRamp(Context& context, PxScene& scene, PxMaterial& material)
{
	const PxQuat rotation = rampRotation();
	const PxVec3 half(20.0f, 0.5f, 4.0f);
	PxRigidStatic* ramp = PxCreateStatic(context.physics, PxTransform(rotation.rotate(PxVec3(8.0f, -half.y, 0.0f)), rotation), PxBoxGeometry(half), material);
	scene.addActor(*ramp);
}

// Boxes of different sizes and masses slide down a 25 degree ramp in separate lanes.
void rampBoxes(Context& context)
{
	PxScene* scene = context.createScene();
	PxMaterial* material = context.physics.createMaterial(RAMP_FRICTION, RAMP_FRICTION, 0.0f);
	addRamp(context, *scene, *material);
	const PxQuat rotation = rampRotation();
	const PxVec3 normal = rotation.rotate(PxVec3(0.0f, 1.0f, 0.0f));
	const PxVec3 tangent = rotation.rotate(PxVec3(1.0f, 0.0f, 0.0f));
	const PxVec3 sizes[] = { PxVec3(0.15f, 0.1f, 0.15f), PxVec3(0.25f, 0.2f, 0.2f), PxVec3(0.3f, 0.15f, 0.25f), PxVec3(0.1f, 0.1f, 0.1f), PxVec3(0.4f, 0.3f, 0.3f), PxVec3(0.2f, 0.25f, 0.2f) };
	std::vector<Slider> sliders;
	for(int i = 0; i < 6; ++i)
	{
		const PxVec3 position = normal * sizes[i].y + PxVec3(0.0f, 0.0f, -3.0f + 1.2f * i);
		PxRigidDynamic* box = context.box(*scene, *material, sizes[i], PxTransform(position, rotation), PxReal(1 + i));
		const Slider slider = { box, PxTransform(PxIdentity), sizes[i], { NULL, PxVec3(0.0f), normal } };
		sliders.push_back(slider);
	}
	Metrics metrics = run(*scene, sliders, 50, 5);
	double early = 0.0;
	for(const Slider& slider : sliders)
		early += tangent.dot(slider.body->getLinearVelocity());
	const Metrics later = run(*scene, sliders, 150, 0);
	double late = 0.0;
	for(const Slider& slider : sliders)
		late += tangent.dot(slider.body->getLinearVelocity());
	const double acceleration = (late - early) / 6.0 / (150.0 * TIMESTEP);
	const double expected = GRAVITY * (std::sin(RAMP_ANGLE) - RAMP_FRICTION * std::cos(RAMP_ANGLE));
	metrics.maximumGap = std::max(metrics.maximumGap, later.maximumGap);
	metrics.minimumGap = std::min(metrics.minimumGap, later.minimumGap);
	metrics.maximumNormalSpeed = std::max(metrics.maximumNormalSpeed, later.maximumNormalSpeed);
	metrics.stepMs = (metrics.stepMs * 50 + later.stepMs * 150) / 200;
	metrics.solveMs = (metrics.solveMs * 50 + later.solveMs * 150) / 200;
	metrics.iterations = (metrics.iterations * 50 + later.iterations * 150) / 200;
	metrics.factorizations = (metrics.factorizations * 50 + later.factorizations * 150) / 200;
	report("ramp_boxes", metrics, acceleration / expected, "acceleration_ratio");
	checkContact("ramp_boxes", metrics);
	check(std::abs(acceleration / expected - 1.0) < 0.03, "ramp_boxes: sliding acceleration matches Coulomb friction");
	scene->release();
	material->release();
}

// A loaded container slides down the ramp. Its load moves with it.
void rampContainer(Context& context)
{
	PxScene* scene = context.createScene();
	PxMaterial* material = context.physics.createMaterial(RAMP_FRICTION, RAMP_FRICTION, 0.0f);
	addRamp(context, *scene, *material);
	const PxQuat rotation = rampRotation();
	const PxVec3 normal = rotation.rotate(PxVec3(0.0f, 1.0f, 0.0f));
	const PxVec3 tangent = rotation.rotate(PxVec3(1.0f, 0.0f, 0.0f));
	const Container container = createContainer(context, *scene, *material, PxTransform(PxVec3(0.0f), rotation), 2);
	std::vector<Slider> sliders;
	const Slider base = { container.body, PxTransform(PxVec3(0.0f, container.baseHalf.y, 0.0f)), container.baseHalf, { NULL, PxVec3(0.0f), normal } };
	sliders.push_back(base);
	for(int i = 0; i < 4; ++i)
	{
		const Slider load = { container.load[i], PxTransform(PxIdentity), container.loadHalf, floorOf(container) };
		sliders.push_back(load);
	}
	const Metrics metrics = run(*scene, sliders, 200, 20);
	const double speed = tangent.dot(container.body->getLinearVelocity());
	const double expected = GRAVITY * (std::sin(RAMP_ANGLE) - RAMP_FRICTION * std::cos(RAMP_ANGLE)) * 200.0 * TIMESTEP;
	report("ramp_container", metrics, speed / expected, "speed_ratio");
	checkContact("ramp_container", metrics);
	check(std::abs(speed / expected - 1.0) < 0.03, "ramp_container: loaded container slides at the Coulomb rate");
	scene->release();
	material->release();
}

// A kinematic pusher drives a loaded container along a flat floor. The load slides
// inside the container while it accelerates.
void pushedContainer(Context& context)
{
	PxScene* scene = context.createScene();
	PxMaterial* material = context.physics.createMaterial(0.5f, 0.5f, 0.0f);
	scene->addActor(*PxCreatePlane(context.physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *material));
	const Container container = createContainer(context, *scene, *material, PxTransform(PxIdentity), 2);
	const PxVec3 pusherHalf(0.05f, 0.15f, 0.3f);
	const PxReal pusherStart = -container.baseHalf.x - pusherHalf.x - 0.01f;
	PxRigidDynamic* pusher = context.physics.createRigidDynamic(PxTransform(PxVec3(pusherStart, pusherHalf.y + 0.01f, 0.0f)));
	PxShape* shape = context.physics.createShape(PxBoxGeometry(pusherHalf), *material);
	pusher->attachShape(*shape);
	shape->release();
	pusher->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
	scene->addActor(*pusher);
	std::vector<Slider> sliders;
	const Slider base = { container.body, PxTransform(PxVec3(0.0f, container.baseHalf.y, 0.0f)), container.baseHalf, { NULL, PxVec3(0.0f), PxVec3(0.0f, 1.0f, 0.0f) } };
	sliders.push_back(base);
	for(int i = 0; i < 4; ++i)
	{
		const Slider load = { container.load[i], PxTransform(PxIdentity), container.loadHalf, floorOf(container) };
		sliders.push_back(load);
	}
	// Ramp the pusher to 1.5 m/s over 0.3 s, then hold that speed.
	PxReal x = pusherStart;
	const auto push = [&](int frame)
	{
		const PxReal speed = 1.5f * std::min(1.0f, PxReal(frame) / 30.0f);
		x += speed * TIMESTEP;
		pusher->setKinematicTarget(PxTransform(PxVec3(x, pusherHalf.y + 0.01f, 0.0f)));
	};
	const Metrics metrics = run(*scene, sliders, 250, 20, push);
	const double speed = container.body->getLinearVelocity().x;
	report("pushed_container", metrics, speed / 1.5, "speed_ratio");
	checkContact("pushed_container", metrics);
	check(std::abs(speed / 1.5 - 1.0) < 0.05, "pushed_container: container follows the pusher");
	scene->release();
	material->release();
}
}

int main(int argc, char** argv)
{
	const char* selection = argc > 1 ? argv[1] : "all";
	PxDefaultAllocator allocator;
	PxDefaultErrorCallback errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	PxSetProfilerCallback(&profiler);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(4);
	Context context = { *physics, *dispatcher };
	const bool all = std::strcmp(selection, "all") == 0;
	const int repeats = argc > 2 ? std::max(1, std::atoi(argv[2])) : 1;
	for(int repeat = 0; repeat < repeats; ++repeat)
	{
		if(all || std::strcmp(selection, "ramp_boxes") == 0)
			rampBoxes(context);
		if(all || std::strcmp(selection, "ramp_container") == 0)
			rampContainer(context);
		if(all || std::strcmp(selection, "pushed_container") == 0)
			pushedContainer(context);
	}
	PxSetProfilerCallback(NULL);
	dispatcher->release();
	physics->release();
	foundation->release();
	std::printf("Sliding tests: failures=%d\n", failures);
	return failures ? 1 : 0;
}
