#include "PxPhysicsAPI.h"
#include "extensions/PxCustomGeometryExt.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

using namespace physx;

namespace
{
const PxReal TIMESTEP = 0.01f;

struct FrictionCase
{
	enum Enum
	{
		eFLAT,
		eINCLINE,
		eCONVEYOR,
		eSTATIC,
		eCYLINDER,
		eCAPSULE,
		eSTACK,
		eCOUNT
	};
};

const char* const CASE_NAMES[FrictionCase::eCOUNT] =
{
	"flat", "incline", "conveyor", "static", "cylinder", "capsule", "stack"
};

struct ComparisonSettings
{
	PxSolverType::Enum solver;
	const char* solverName;
	PxReal speed;
	PxU32 steps;
	PxU32 threads;
	PxReal regularization;
	PxReal staticFriction;
	PxReal dynamicFriction;
	PxU32 positionIterations;
	PxU32 velocityIterations;
};

class ComparisonErrors : public PxErrorCallback
{
public:
	std::atomic<PxU32> count;

	ComparisonErrors() : count(0) {}

	virtual void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) PX_OVERRIDE
	{
		if(code == PxErrorCode::eDEBUG_INFO || code == PxErrorCode::eDEBUG_WARNING || code == PxErrorCode::ePERF_WARNING)
			return;
		++count;
		std::fprintf(stderr, "PHYSX ERROR %s (%s:%d)\n", message, file, line);
	}
};

class ConveyorModifier : public PxContactModifyCallback
{
public:
	const PxRigidActor* floor;
	PxVec3 velocity;

	ConveyorModifier() : floor(NULL), velocity(0.0f) {}

	virtual void onContactModify(PxContactModifyPair* const pairs, PxU32 count) PX_OVERRIDE
	{
		if(velocity.isZero())
		{
			return;
		}
		for(PxU32 pairIndex = 0; pairIndex < count; ++pairIndex)
		{
			PxContactModifyPair& pair = pairs[pairIndex];
			if(pair.actor[0] != floor && pair.actor[1] != floor)
			{
				continue;
			}
			const PxVec3 target = pair.actor[0] == floor ? -velocity : velocity;
			const PxU32 contactCount = pair.contacts.size();
			for(PxU32 point = 0; point < contactCount; ++point)
			{
				pair.contacts.setTargetVelocity(point, target);
			}
		}
	}
};

class ImpulseRecorder : public PxSimulationEventCallback
{
public:
	const PxRigidActor* floor;
	PxVec3 normal;
	PxVec3 tangent;
	std::mutex mutex;
	PxU32 points;
	PxU32 anchors;
	double normalImpulse;
	double tangentImpulse;
	double absoluteTangentImpulse;
	bool finite;

	ImpulseRecorder() : floor(NULL), normal(0.0f, 1.0f, 0.0f), tangent(1.0f, 0.0f, 0.0f)
	{
		reset();
	}

	void reset()
	{
		points = anchors = 0;
		normalImpulse = tangentImpulse = absoluteTangentImpulse = 0.0;
		finite = true;
	}

	virtual void onConstraintBreak(PxConstraintInfo*, PxU32) PX_OVERRIDE {}
	virtual void onWake(PxActor**, PxU32) PX_OVERRIDE {}
	virtual void onSleep(PxActor**, PxU32) PX_OVERRIDE {}
	virtual void onTrigger(PxTriggerPair*, PxU32) PX_OVERRIDE {}
	virtual void onAdvance(const PxRigidBody* const*, const PxTransform*, const PxU32) PX_OVERRIDE {}

	virtual void onContact(const PxContactPairHeader& header, const PxContactPair* pairs, PxU32 count) PX_OVERRIDE
	{
		if(header.actors[0] != floor && header.actors[1] != floor)
			return;
		std::lock_guard<std::mutex> lock(mutex);
		const PxReal sign = header.actors[0] == floor ? -1.0f : 1.0f;
		for(PxU32 pairIndex = 0; pairIndex < count; ++pairIndex)
		{
			const PxContactPair& pair = pairs[pairIndex];
			// Native contact and patch counts are bytes; these arrays hold every
			// reported point and both friction anchors of every possible patch.
			PxContactPairPoint contactPoints[256];
			const PxU32 pointCount = pair.extractContacts(contactPoints, 256);
			points += pointCount;
			for(PxU32 point = 0; point < pointCount; ++point)
			{
				const PxVec3 impulse = sign * contactPoints[point].impulse;
				finite = finite && impulse.isFinite();
				normalImpulse += double(impulse.dot(normal));
			}
			if(pair.frictionPatches)
			{
				PxContactPairFrictionAnchor frictionAnchors[512];
				const PxU32 anchorCount = pair.extractFrictionAnchors(frictionAnchors, 512);
				anchors += anchorCount;
				for(PxU32 anchor = 0; anchor < anchorCount; ++anchor)
				{
					const PxVec3 impulse = sign * frictionAnchors[anchor].impulse;
					finite = finite && impulse.isFinite();
					tangentImpulse += double(impulse.dot(tangent));
					absoluteTangentImpulse += double(impulse.magnitude());
				}
			}
		}
	}
};

PxFilterFlags comparisonFilter(PxFilterObjectAttributes, PxFilterData, PxFilterObjectAttributes, PxFilterData, PxPairFlags& flags, const void*, PxU32)
{
	flags = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eMODIFY_CONTACTS | PxPairFlag::eNOTIFY_TOUCH_FOUND |
		PxPairFlag::eNOTIFY_TOUCH_PERSISTS | PxPairFlag::eNOTIFY_CONTACT_POINTS;
	return PxFilterFlag::eDEFAULT;
}

struct ComparisonContext
{
	PxPhysics& physics;
	PxDefaultCpuDispatcher& dispatcher;
	const ComparisonSettings& settings;
	FILE* trajectory;

	PxScene* scene(ConveyorModifier& modifier, ImpulseRecorder& recorder)
	{
		PxSceneDesc desc(physics.getTolerancesScale());
		desc.cpuDispatcher = &dispatcher;
		desc.filterShader = comparisonFilter;
		desc.contactModifyCallback = &modifier;
		desc.simulationEventCallback = &recorder;
		desc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
		desc.solverType = settings.solver;
		desc.newtonMaxIterations = 100;
		desc.newtonTolerance = 1e-8f;
		desc.newtonRegularization = settings.regularization;
		desc.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
		return physics.createScene(desc);
	}

	PxRigidDynamic* body(PxScene& targetScene, PxMaterial& material, const PxGeometry& geometry, const PxTransform& pose, PxReal mass)
	{
		PxRigidDynamic* actor = PxCreateDynamic(physics, pose, geometry, material, 1.0f);
		if(!actor)
		{
			return NULL;
		}
		if(!PxRigidBodyExt::setMassAndUpdateInertia(*actor, mass))
		{
			actor->release();
			return NULL;
		}
		actor->setLinearDamping(0.0f);
		actor->setAngularDamping(0.0f);
		actor->setSleepThreshold(0.0f);
		actor->setSolverIterationCounts(settings.positionIterations, settings.velocityIterations);
		targetScene.addActor(*actor);
		return actor;
	}
};

struct ComparisonSummary
{
	double maximumRise;
	double maximumUpwardSpeed;
	double minimumClearance;
	double maximumClearance;
	double maximumStaticDrift;
	double normalImpulse;
	double tangentImpulse;
	double absoluteTangentImpulse;
	PxU32 supportedSteps;

	ComparisonSummary() : maximumRise(0.0), maximumUpwardSpeed(0.0), minimumClearance(0.0),
		maximumClearance(0.0), maximumStaticDrift(0.0), normalImpulse(0.0), tangentImpulse(0.0),
		absoluteTangentImpulse(0.0), supportedSteps(0) {}
};

void lockRotation(PxRigidDynamic& actor)
{
	actor.setRigidDynamicLockFlags(PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y | PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
}

// The ideal shape's lowest support point, independent of the contact envelope.
double floorClearance(FrictionCase::Enum selection, const PxTransform& pose, const PxVec3& normal)
{
	double extent = 0.0;
	if(selection == FrictionCase::eCYLINDER)
	{
		const double axis = double(normal.dot(pose.q.rotate(PxVec3(0.0f, 0.0f, 1.0f))));
		extent = .75 * std::abs(axis) + .5 * std::sqrt(std::max(0.0, 1.0 - axis * axis));
	}
	else if(selection == FrictionCase::eCAPSULE)
	{
		const double axis = double(normal.dot(pose.q.rotate(PxVec3(1.0f, 0.0f, 0.0f))));
		extent = .5 + .75 * std::abs(axis);
	}
	else
	{
		const PxVec3 halfSize = selection == FrictionCase::eSTACK ? PxVec3(.6f, .2f, .5f) : PxVec3(.5f);
		extent = double(halfSize.x) * std::abs(double(normal.dot(pose.q.rotate(PxVec3(1.0f, 0.0f, 0.0f))))) +
			double(halfSize.y) * std::abs(double(normal.dot(pose.q.rotate(PxVec3(0.0f, 1.0f, 0.0f))))) +
			double(halfSize.z) * std::abs(double(normal.dot(pose.q.rotate(PxVec3(0.0f, 0.0f, 1.0f)))));
	}
	return double(normal.dot(pose.p)) - extent;
}

void recordTrajectory(ComparisonContext& context, FrictionCase::Enum selection, PxU32 frame, PxRigidDynamic* const* actors, PxU32 bodyCount, const PxVec3* initialPositions, const PxVec3& normal, const PxVec3& tangent, const ImpulseRecorder& recorder)
{
	if(!context.trajectory)
	{
		return;
	}
	for(PxU32 bodyIndex = 0; bodyIndex < bodyCount; ++bodyIndex)
	{
		const PxTransform pose = actors[bodyIndex]->getGlobalPose();
		const PxVec3 velocity = actors[bodyIndex]->getLinearVelocity();
		const PxVec3 angularVelocity = actors[bodyIndex]->getAngularVelocity();
		const PxVec3 displacement = pose.p - initialPositions[bodyIndex];
		std::fprintf(context.trajectory,
			"%s,%s,%.9g,%u,%.9g,%u,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
			"%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%u,%u\n",
			CASE_NAMES[selection], context.settings.solverName, double(context.settings.speed),
			frame, double(frame) * TIMESTEP, bodyIndex,
			double(pose.p.x), double(pose.p.y), double(pose.p.z),
			double(velocity.x), double(velocity.y), double(velocity.z),
			double(angularVelocity.x), double(angularVelocity.y), double(angularVelocity.z),
			double(displacement.dot(normal)), double(velocity.dot(normal)), double(displacement.dot(tangent)),
			double(velocity.dot(tangent)), bodyIndex == 0 ? floorClearance(selection, pose, normal) : 0.0,
			recorder.normalImpulse, recorder.tangentImpulse, recorder.points, recorder.anchors);
	}
}

bool runCase(ComparisonContext& context, FrictionCase::Enum selection)
{
	const ComparisonSettings& settings = context.settings;
	const PxQuat incline = selection == FrictionCase::eINCLINE ?
		PxQuat(-PxPi / 6.0f, PxVec3(0.0f, 0.0f, 1.0f)) : PxQuat(PxIdentity);
	const PxVec3 normal = incline.rotate(PxVec3(0.0f, 1.0f, 0.0f));
	const PxVec3 tangent = incline.rotate(PxVec3(1.0f, 0.0f, 0.0f));
	ConveyorModifier modifier;
	ImpulseRecorder recorder;
	recorder.normal = normal;
	recorder.tangent = tangent;

	PxMaterial* material = context.physics.createMaterial(settings.staticFriction, settings.dynamicFriction, 0.0f);
	if(!material)
		return false;
	PxScene* scene = context.scene(modifier, recorder);
	if(!scene)
	{
		material->release();
		return false;
	}
	PxRigidStatic* floor = PxCreatePlane(context.physics, PxPlane(normal, 0.0f), *material);
	if(!floor)
	{
		scene->release();
		material->release();
		return false;
	}
	scene->addActor(*floor);
	modifier.floor = recorder.floor = floor;
	if(selection == FrictionCase::eCONVEYOR)
		modifier.velocity = tangent * settings.speed;

	// Exact cylinder geometry keeps the rolling check independent of convex
	// tessellation. The callback must remain alive until the scene is released.
	PxCustomGeometryExt::CylinderCallbacks cylinderCallbacks(1.5f, .5f, 2);
	PxCustomGeometry cylinderGeometry(cylinderCallbacks);
	PxCapsuleGeometry capsuleGeometry(.5f, .75f);
	const PxVec3 halfSize = selection == FrictionCase::eSTACK ? PxVec3(.6f, .2f, .5f) : PxVec3(.5f);
	PxBoxGeometry boxGeometry(halfSize);
	const PxGeometry* geometry = &boxGeometry;
	PxQuat rotation = incline;
	if(selection == FrictionCase::eCYLINDER)
		geometry = &cylinderGeometry;
	if(selection == FrictionCase::eCAPSULE)
	{
		geometry = &capsuleGeometry;
		rotation = PxQuat(PxHalfPi, PxVec3(0.0f, 1.0f, 0.0f));
	}
	const PxReal height = selection == FrictionCase::eSTACK ? .2f : .5f;
	const PxReal mass = selection == FrictionCase::eSTACK ? 10.0f : 1.0f;
	PxRigidDynamic* actors[3] = {NULL, NULL, NULL};
	PxU32 bodyCount = 1;
	actors[0] = context.body(*scene, *material, *geometry, PxTransform(normal * height, rotation), mass);
	if(actors[0] && selection == FrictionCase::eSTACK)
	{
		// A 10 kg top box falls 25 cm onto a 1 kg middle box and a 10 kg base.
		// All three begin translating, exposing friction under a changing load.
		bodyCount = 3;
		actors[1] = context.body(*scene, *material, PxBoxGeometry(.4f, .25f, .4f),
			PxTransform(PxVec3(0.0f, .65f, 0.0f)), 1.0f);
		actors[2] = context.body(*scene, *material, PxBoxGeometry(.3f, .25f, .3f),
			PxTransform(PxVec3(0.0f, 1.4f, 0.0f)), 10.0f);
	}
	bool valid = true;
	for(PxU32 bodyIndex = 0; bodyIndex < bodyCount; ++bodyIndex)
		valid = valid && actors[bodyIndex] != NULL;
	if(!valid)
	{
		scene->release();
		material->release();
		return false;
	}
	const bool rolling = selection == FrictionCase::eCYLINDER || selection == FrictionCase::eCAPSULE;
	if(!rolling && selection != FrictionCase::eSTACK)
		lockRotation(*actors[0]);
	PxVec3 initialPositions[3];
	for(PxU32 bodyIndex = 0; bodyIndex < bodyCount; ++bodyIndex)
	{
		initialPositions[bodyIndex] = actors[bodyIndex]->getGlobalPose().p;
		if(selection != FrictionCase::eCONVEYOR && selection != FrictionCase::eSTATIC)
			actors[bodyIndex]->setLinearVelocity(tangent * settings.speed);
	}
	if(rolling)
		actors[0]->setAngularVelocity(PxVec3(0.0f, 0.0f, -settings.speed / .5f));

	ComparisonSummary summary;
	recordTrajectory(context, selection, 0, actors, bodyCount, initialPositions, normal, tangent, recorder);
	for(PxU32 frame = 1; frame <= settings.steps; ++frame)
	{
		if(selection == FrictionCase::eSTATIC)
		{
			// With distinct coefficients the force lies between the two Coulomb
			// bounds. Equal coefficients instead test half the static threshold.
			const PxReal coefficient = settings.staticFriction > settings.dynamicFriction ?
				.5f * (settings.staticFriction + settings.dynamicFriction) : .5f * settings.staticFriction;
			actors[0]->addForce(tangent * (mass * 9.81f * coefficient));
		}
		recorder.reset();
		scene->simulate(TIMESTEP);
		scene->fetchResults(true);
		valid = valid && recorder.finite;
		for(PxU32 bodyIndex = 0; bodyIndex < bodyCount; ++bodyIndex)
		{
			valid = valid && actors[bodyIndex]->getGlobalPose().isFinite() &&
				actors[bodyIndex]->getLinearVelocity().isFinite() && actors[bodyIndex]->getAngularVelocity().isFinite();
		}
		if(!valid)
			break;
		const PxTransform pose = actors[0]->getGlobalPose();
		const PxVec3 displacement = pose.p - initialPositions[0];
		const PxVec3 velocity = actors[0]->getLinearVelocity();
		const double clearance = floorClearance(selection, pose, normal);
		summary.maximumRise = std::max(summary.maximumRise, double(displacement.dot(normal)));
		summary.maximumUpwardSpeed = std::max(summary.maximumUpwardSpeed, double(velocity.dot(normal)));
		summary.minimumClearance = std::min(summary.minimumClearance, clearance);
		summary.maximumClearance = std::max(summary.maximumClearance, clearance);
		if(selection == FrictionCase::eSTATIC)
			summary.maximumStaticDrift = std::max(summary.maximumStaticDrift, std::abs(double(displacement.dot(tangent))));
		summary.normalImpulse += recorder.normalImpulse;
		summary.tangentImpulse += recorder.tangentImpulse;
		summary.absoluteTangentImpulse += recorder.absoluteTangentImpulse;
		summary.supportedSteps += recorder.normalImpulse > 0.0;
		recordTrajectory(context, selection, frame, actors, bodyCount, initialPositions, normal, tangent, recorder);
	}

	const PxVec3 displacement = actors[0]->getGlobalPose().p - initialPositions[0];
	const PxVec3 velocity = actors[0]->getLinearVelocity();
	const PxVec3 angularVelocity = actors[0]->getAngularVelocity();
	std::printf("%s solver=%s speed=%.9g steps=%u mu_static=%.9g mu_dynamic=%.9g regularization=%.9g "
		"rise=%.9g max_upward_speed=%.9g min_clearance=%.9g max_clearance=%.9g "
		"travel=%.9g final_speed=%.9g final_spin=%.9g static_drift=%.9g "
		"normal_impulse=%.9g tangent_impulse=%.9g absolute_tangent_impulse=%.9g supported_steps=%u finite=%d\n",
		CASE_NAMES[selection], settings.solverName, double(settings.speed), settings.steps,
		double(settings.staticFriction), double(settings.dynamicFriction), double(settings.regularization),
		summary.maximumRise, summary.maximumUpwardSpeed, summary.minimumClearance, summary.maximumClearance,
		double(displacement.dot(tangent)), double(velocity.dot(tangent)), double(angularVelocity.z),
		summary.maximumStaticDrift, summary.normalImpulse, summary.tangentImpulse,
		summary.absoluteTangentImpulse, summary.supportedSteps, valid ? 1 : 0);
	scene->release();
	material->release();
	return valid;
}
}

int main(int argc, char** argv)
{
	if(argc < 4)
	{
		std::fprintf(stderr, "Usage: NewtonFrictionComparison <all|flat|incline|conveyor|static|cylinder|capsule|stack> "
			"<pgs|newton> <trajectory.csv|-> [speed=.2] [steps=300] [threads=8] [regularization=1e-4] "
			"[staticFriction=.5] [dynamicFriction=.5] [positionIterations=98] [velocityIterations=2]\n");
		return 2;
	}
	const bool all = std::strcmp(argv[1], "all") == 0;
	int selected = -1;
	for(int index = 0; index < FrictionCase::eCOUNT; ++index)
	{
		if(std::strcmp(argv[1], CASE_NAMES[index]) == 0)
		{
			selected = index;
		}
	}
	if((!all && selected < 0) || (std::strcmp(argv[2], "pgs") != 0 && std::strcmp(argv[2], "newton") != 0))
	{
		return 2;
	}
	ComparisonSettings settings;
	settings.solver = std::strcmp(argv[2], "pgs") == 0 ? PxSolverType::ePGS : PxSolverType::eNEWTON;
	settings.solverName = argv[2];
	settings.speed = argc > 4 ? PxReal(std::atof(argv[4])) : .2f;
	settings.steps = argc > 5 ? PxU32(std::atoi(argv[5])) : 300;
	settings.threads = argc > 6 ? PxU32(std::atoi(argv[6])) : 8;
	settings.regularization = argc > 7 ? PxReal(std::atof(argv[7])) : 1e-4f;
	settings.staticFriction = argc > 8 ? PxReal(std::atof(argv[8])) : .5f;
	settings.dynamicFriction = argc > 9 ? PxReal(std::atof(argv[9])) : .5f;
	settings.positionIterations = argc > 10 ? PxU32(std::atoi(argv[10])) : 98;
	settings.velocityIterations = argc > 11 ? PxU32(std::atoi(argv[11])) : 2;
	if(!PxIsFinite(settings.speed) || !settings.steps || settings.steps > 1000000 || settings.threads > 128 || !PxIsFinite(settings.regularization) || settings.regularization <= 0.0f || !PxIsFinite(settings.staticFriction) || settings.staticFriction < 0.0f || !PxIsFinite(settings.dynamicFriction) || settings.dynamicFriction < 0.0f || !settings.positionIterations || settings.positionIterations > 255 || settings.velocityIterations > 255)
	{
		return 2;
	}
	FILE* trajectory = NULL;
	if(std::strcmp(argv[3], "-") != 0)
	{
		trajectory = std::fopen(argv[3], "w");
		if(!trajectory)
		{
			std::fprintf(stderr, "Cannot open trajectory file: %s\n", argv[3]);
			return 2;
		}
		std::fprintf(trajectory, "case,solver,initial_speed,step,time,body,x,y,z,vx,vy,vz,wx,wy,wz,"
			"normal_displacement,normal_speed,tangent_displacement,tangent_speed,base_floor_clearance,"
			"floor_normal_impulse,floor_tangent_impulse,floor_contact_points,floor_friction_anchors\n");
	}
	PxDefaultAllocator allocator;
	ComparisonErrors errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	if(!foundation)
	{
		if(trajectory)
		{
			std::fclose(trajectory);
		}
		return 1;
	}
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	if(!physics)
	{
		foundation->release();
		if(trajectory)
		{
			std::fclose(trajectory);
		}
		return 1;
	}
	const bool extensions = PxInitExtensions(*physics, NULL);
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(settings.threads);
	bool valid = extensions && dispatcher != NULL;
	if(valid)
	{
		ComparisonContext context = {*physics, *dispatcher, settings, trajectory};
		std::printf("configuration timestep=%.9g threads=%u position_iterations=%u velocity_iterations=%u "
			"newton_iterations=100 newton_tolerance=1e-8 friction_every_iteration=1 default_contact_offsets=1\n",
			double(TIMESTEP), settings.threads, settings.positionIterations, settings.velocityIterations);
		for(int index = 0; index < FrictionCase::eCOUNT; ++index)
		{
			if(all || selected == index)
			{
				valid = runCase(context, FrictionCase::Enum(index)) && valid;
			}
		}
	}
	if(dispatcher)
	{
		dispatcher->release();
	}
	if(extensions)
	{
		PxCloseExtensions();
	}
	physics->release();
	foundation->release();
	if(trajectory && std::fclose(trajectory) != 0)
	{
		valid = false;
	}
	return valid && errors.count.load() == 0 ? 0 : 1;
}

