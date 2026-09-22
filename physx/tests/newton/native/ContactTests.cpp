#include "PxPhysicsAPI.h"
#include "extensions/PxCustomGeometryExt.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

using namespace physx;

namespace
{
const PxReal TIMESTEP = 0.01f;
int failures = 0;

void check(bool condition, const char* message)
{
	if(!condition)
	{
		std::printf("FAIL %s\n", message);
		++failures;
	}
}

class ContactErrors : public PxErrorCallback
{
public:
	std::atomic<int> count;
	ContactErrors() : count(0) {}
	virtual void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) PX_OVERRIDE
	{
		if(code == PxErrorCode::eDEBUG_INFO || code == PxErrorCode::eDEBUG_WARNING || code == PxErrorCode::ePERF_WARNING)
			return;
		++count;
		std::printf("PHYSX ERROR %s (%s:%d)\n", message, file, line);
	}
};

class ContactModifier : public PxContactModifyCallback
{
public:
	enum Mode { eNONE, eCONVEYOR, eCAPPED, eDISABLED };
	Mode mode;
	const PxRigidActor* dynamicActor;
	PxReal targetSpeed;
	PxReal cap;
	std::atomic<PxU32> calls;
	std::atomic<PxU32> enabledPoints;
	std::atomic<PxU32> disabledPoints;
	ContactModifier() : mode(eNONE), dynamicActor(NULL), targetSpeed(0.0f), cap(0.01f), calls(0), enabledPoints(0), disabledPoints(0) {}
	virtual void onContactModify(PxContactModifyPair* const pairs, PxU32 count) PX_OVERRIDE
	{
		calls += count;
		for(PxU32 pairIndex = 0; pairIndex < count; ++pairIndex)
		{
			PxContactModifyPair& pair = pairs[pairIndex];
			const PxU32 contactCount = pair.contacts.size();
			for(PxU32 point = 0; point < contactCount; ++point)
			{
				if(mode == eCONVEYOR)
				{
					const PxReal sign = pair.actor[0] == dynamicActor ? 1.0f : -1.0f;
					pair.contacts.setTargetVelocity(point, PxVec3(sign * targetSpeed, 0.0f, 0.0f));
				}
				if(mode == eDISABLED || (mode == eCAPPED && point % 2 == 0))
				{
					pair.contacts.setMaxImpulse(point, 0.0f);
					++disabledPoints;
				}
				else if(mode == eCAPPED)
				{
					pair.contacts.setMaxImpulse(point, cap);
					++enabledPoints;
				}
			}
		}
	}
};

class ContactRecorder : public PxSimulationEventCallback
{
public:
	std::mutex mutex;
	PxU32 pairs, points, anchors, thresholdEvents;
	double normalImpulse, tangentImpulse, maximumPointImpulse;
	bool finite;
	ContactRecorder() { reset(); }
	void reset()
	{
		pairs = points = anchors = thresholdEvents = 0;
		normalImpulse = tangentImpulse = maximumPointImpulse = 0.0;
		finite = true;
	}
	virtual void onConstraintBreak(PxConstraintInfo*, PxU32) PX_OVERRIDE {}
	virtual void onWake(PxActor**, PxU32) PX_OVERRIDE {}
	virtual void onSleep(PxActor**, PxU32) PX_OVERRIDE {}
	virtual void onTrigger(PxTriggerPair*, PxU32) PX_OVERRIDE {}
	virtual void onAdvance(const PxRigidBody* const*, const PxTransform*, const PxU32) PX_OVERRIDE {}
	virtual void onContact(const PxContactPairHeader&, const PxContactPair* contactPairs, PxU32 count) PX_OVERRIDE
	{
		std::lock_guard<std::mutex> lock(mutex);
		pairs += count;
		for(PxU32 pairIndex = 0; pairIndex < count; ++pairIndex)
		{
			const PxContactPair& pair = contactPairs[pairIndex];
			if(pair.events & (PxPairFlag::eNOTIFY_THRESHOLD_FORCE_FOUND | PxPairFlag::eNOTIFY_THRESHOLD_FORCE_PERSISTS))
			{
				++thresholdEvents;
			}
			PxContactPairPoint contactPoints[64];
			const PxU32 pointCount = pair.extractContacts(contactPoints, 64);
			points += pointCount;
			for(PxU32 point = 0; point < pointCount; ++point)
			{
				finite = finite && contactPoints[point].position.isFinite() && contactPoints[point].impulse.isFinite();
				const double impulse = double(contactPoints[point].impulse.magnitude());
				normalImpulse += impulse;
				maximumPointImpulse = std::max(maximumPointImpulse, impulse);
			}
			if(pair.frictionPatches)
			{
				PxContactPairFrictionAnchor frictionAnchors[128];
				const PxU32 anchorCount = pair.extractFrictionAnchors(frictionAnchors, 128);
				anchors += anchorCount;
				for(PxU32 anchor = 0; anchor < anchorCount; ++anchor)
				{
					finite = finite && frictionAnchors[anchor].position.isFinite() && frictionAnchors[anchor].impulse.isFinite();
					tangentImpulse += std::abs(double(frictionAnchors[anchor].impulse.x));
				}
			}
		}
	}
};

PxFilterFlags contactFilter(PxFilterObjectAttributes, PxFilterData, PxFilterObjectAttributes, PxFilterData, PxPairFlags& flags, const void*, PxU32)
{
	flags = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eMODIFY_CONTACTS | PxPairFlag::eNOTIFY_TOUCH_FOUND |
		PxPairFlag::eNOTIFY_TOUCH_PERSISTS | PxPairFlag::eNOTIFY_CONTACT_POINTS |
		PxPairFlag::eNOTIFY_THRESHOLD_FORCE_FOUND | PxPairFlag::eNOTIFY_THRESHOLD_FORCE_PERSISTS;
	return PxFilterFlag::eDEFAULT;
}

struct ContactContext
{
	PxPhysics& physics;
	PxDefaultCpuDispatcher& dispatcher;
	PxSolverType::Enum solver;

	PxScene* createScene(ContactModifier& modifier, ContactRecorder& recorder, const PxVec3& gravity = PxVec3(0.0f, -9.81f, 0.0f))
	{
		PxSceneDesc desc(physics.getTolerancesScale());
		desc.cpuDispatcher = &dispatcher;
		desc.filterShader = contactFilter;
		desc.contactModifyCallback = &modifier;
		desc.simulationEventCallback = &recorder;
		desc.gravity = gravity;
		desc.solverType = solver;
		desc.newtonMaxIterations = 100;
		desc.newtonTolerance = 1e-8f;
		desc.newtonRegularization = 1e-4f;
		desc.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
		PxScene* scene = physics.createScene(desc);
		check(scene != NULL, "contact scene creation");
		return scene;
	}

	PxRigidDynamic* body(PxScene& scene, PxMaterial& material, const PxGeometry& geometry, const PxTransform& pose, PxReal mass = 1.0f)
	{
		check(geometry.getType() != PxGeometryType::eBOX || static_cast<const PxBoxGeometry&>(geometry).isValid(),
			"box fixture has three positive half extents");
		PxRigidDynamic* actor = PxCreateDynamic(physics, pose, geometry, material, 1.0f);
		PxRigidBodyExt::setMassAndUpdateInertia(*actor, mass);
		actor->setLinearDamping(0.0f);
		actor->setAngularDamping(0.0f);
		actor->setSleepThreshold(0.0f);
		actor->setSolverIterationCounts(16, 2);
		actor->setContactReportThreshold(1.0f);
		scene.addActor(*actor);
		return actor;
	}

	void floor(PxScene& scene, PxMaterial& material)
	{
		scene.addActor(*PxCreatePlane(physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), material));
	}
};

void step(PxScene& scene, ContactRecorder& recorder)
{
	recorder.reset();
	scene.simulate(TIMESTEP);
	scene.fetchResults(true);
	check(recorder.finite, "finite contact point and friction-anchor reporting");
}

void lockRotation(PxRigidDynamic& actor)
{
	actor.setRigidDynamicLockFlags(PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y | PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
}

void conveyor(ContactContext& context)
{
	ContactModifier modifier;
	ContactRecorder recorder;
	modifier.mode = ContactModifier::eCONVEYOR;
	modifier.targetSpeed = 1.0f;
	PxMaterial* material = context.physics.createMaterial(0.7f, 0.7f, 0.0f);
	PxScene* scene = context.createScene(modifier, recorder);
	if(!scene)
	{
		material->release();
		return;
	}
	context.floor(*scene, *material);
	PxRigidDynamic* actor = context.body(*scene, *material, PxBoxGeometry(PxVec3(0.5f)), PxTransform(PxVec3(0.0f, 0.5f, 0.0f)));
	lockRotation(*actor);
	modifier.dynamicActor = actor;
	double normalImpulse = 0.0;
	PxU32 reports = 0;
	PxU32 thresholds = 0;
	for(int frame = 0; frame < 250; ++frame)
	{
		step(*scene, recorder);
		thresholds += recorder.thresholdEvents;
		if(frame >= 150) { normalImpulse += recorder.normalImpulse; ++reports; }
	}
	const PxReal forwardSpeed = actor->getLinearVelocity().x;
	const PxReal forwardPosition = actor->getGlobalPose().p.x;
	check(std::abs(forwardSpeed - 1.0f) < 0.025f, "surface target velocity carries a body at conveyor speed");
	check(forwardPosition > 1.7f, "conveyor causes sustained translation");
	check(std::abs(actor->getGlobalPose().p.y - 0.5f) < 0.01f, "conveyor preserves normal support");
	check(std::abs(normalImpulse / reports - 9.81 * TIMESTEP) < 0.004, "normal impulse reporting balances gravity");
	check(thresholds > 0, "force threshold events are reported");
	modifier.targetSpeed = -1.0f;
	for(int frame = 0; frame < 250; ++frame)
		step(*scene, recorder);
	check(std::abs(actor->getLinearVelocity().x + 1.0f) < 0.025f, "changed target velocity reverses the conveyor");
	check(modifier.calls.load() >= 400, "contact modification callback participates throughout conveyor test");
	std::printf("conveyor forward_vx=%.9g reverse_vx=%.9g x=%.9g y=%.9g normal_impulse=%.9g thresholds=%u\n",
		double(forwardSpeed), double(actor->getLinearVelocity().x), double(forwardPosition), double(actor->getGlobalPose().p.y),
		normalImpulse / reports, thresholds);
	scene->release();
	material->release();
}

void normalCaps(ContactContext& context, bool disabled)
{
	ContactModifier modifier;
	ContactRecorder recorder;
	modifier.mode = disabled ? ContactModifier::eDISABLED : ContactModifier::eCAPPED;
	PxMaterial* material = context.physics.createMaterial(0.0f, 0.0f, 0.0f);
	PxScene* scene = context.createScene(modifier, recorder, PxVec3(0.0f));
	if(!scene)
	{
		material->release();
		return;
	}
	context.floor(*scene, *material);
	PxRigidDynamic* actor = context.body(*scene, *material, PxBoxGeometry(PxVec3(0.5f)), PxTransform(PxVec3(0.0f, 0.49f, 0.0f)));
	lockRotation(*actor);
	actor->setLinearVelocity(PxVec3(0.0f, -1.0f, 0.0f));
	modifier.dynamicActor = actor;
	step(*scene, recorder);
	check(modifier.calls.load() > 0 && modifier.disabledPoints.load() > 0, "contact modification disables requested points");
	check(recorder.maximumPointImpulse <= double(modifier.cap) + 1e-6, "reported point impulses respect the modified cap");
	check(std::abs(double(actor->getLinearVelocity().y) + 1.0 - recorder.normalImpulse) < 2e-5,
		"point impulse writeback agrees with body momentum change");
	if(disabled)
	{
		check(recorder.normalImpulse == 0.0, "zero cap produces zero contact impulse");
		check(std::abs(actor->getGlobalPose().p.y - 0.48f) < 2e-6f, "disabled contact has no hidden position correction");
	}
	else
	{
		check(modifier.enabledPoints.load() > 0, "mixed contact manifold retains nonzero capped points");
		check(recorder.normalImpulse > 0.0, "nonzero capped points still solve");
		check(recorder.normalImpulse <= double(modifier.cap) * modifier.enabledPoints.load() + 1e-6,
			"total impulse includes only enabled points");
	}
	std::printf("%s vy=%.9g y=%.9g impulse=%.9g maximum_point=%.9g enabled=%u disabled=%u\n",
		disabled ? "disabled_contacts" : "capped_contacts", double(actor->getLinearVelocity().y), double(actor->getGlobalPose().p.y),
		recorder.normalImpulse, recorder.maximumPointImpulse, modifier.enabledPoints.load(), modifier.disabledPoints.load());
	scene->release();
	material->release();
}

void compliant(ContactContext& context, bool accelerationSpring)
{
	ContactModifier modifier;
	ContactRecorder recorder;
	const double mass = 2.0;
	const double stiffness = 4000.0;
	const double effectiveMass = accelerationSpring ? 1.0 : mass;
	const double damping = 2.0 * std::sqrt(stiffness * effectiveMass);
	PxMaterial* material = context.physics.createMaterial(0.8f, 0.2f, PxReal(-stiffness));
	material->setDamping(PxReal(damping));
	if(accelerationSpring)
	{
		material->setFlag(PxMaterialFlag::eCOMPLIANT_ACCELERATION_SPRING, true);
	}
	PxScene* scene = context.createScene(modifier, recorder);
	if(!scene)
	{
		material->release();
		return;
	}
	context.floor(*scene, *material);
	PxRigidDynamic* actor = context.body(*scene, *material, PxSphereGeometry(0.5f), PxTransform(PxVec3(0.0f, 0.5f, 0.0f)), PxReal(mass));
	lockRotation(*actor);
	double position = 0.5;
	double velocity = 0.0;
	double maximumError = 0.0;
	double reportedImpulse = 0.0;
	for(int frame = 0; frame < 300; ++frame)
	{
		const double dt = double(TIMESTEP);
		velocity = (velocity - 9.81 * dt - dt * stiffness * (position - 0.5) / effectiveMass) /
			(1.0 + dt * damping / effectiveMass + dt * dt * stiffness / effectiveMass);
		position += dt * velocity;
		step(*scene, recorder);
		maximumError = std::max(maximumError, std::abs(double(actor->getGlobalPose().p.y) - position));
		if(frame >= 200)
			reportedImpulse += recorder.normalImpulse;
	}
	const double equilibrium = 0.5 - 9.81 * effectiveMass / stiffness;
	check(maximumError < 6e-4, "compliant contact matches the implicit spring-damper recurrence");
	check(std::abs(double(actor->getGlobalPose().p.y) - equilibrium) < 2e-5, "compliant contact reaches physical static compression");
	check(std::abs(actor->getLinearVelocity().y) < 1e-4f, "critically damped contact settles");
	check(std::abs(reportedImpulse / 100.0 - mass * 9.81 * TIMESTEP) < 0.002, "compliant contact reports supporting impulse");
	const double initialX = actor->getGlobalPose().p.x;
	double maximumDrift = 0.0;
	for(int frame = 0; frame < 100; ++frame)
	{
		actor->addForce(PxVec3(PxReal(0.1 * mass * 9.81), 0.0f, 0.0f));
		step(*scene, recorder);
		maximumDrift = std::max(maximumDrift, std::abs(double(actor->getGlobalPose().p.x) - initialX));
	}
	check(maximumDrift < 0.002, "friction holds a compliant contact under a sub-dynamic-friction load");
	actor->setLinearVelocity(PxVec3(0.5f, 0.0f, 0.0f));
	double dynamicFrictionRatio = -1.0;
	for(int frame = 0; frame < 4; ++frame)
	{
		const PxReal speedBefore = actor->getLinearVelocity().x;
		step(*scene, recorder);
		if(frame > 0 && recorder.normalImpulse > 1e-8)
			dynamicFrictionRatio = mass * (speedBefore - actor->getLinearVelocity().x) / recorder.normalImpulse;
	}
	check(std::abs(dynamicFrictionRatio - 0.2) < 0.002,
		"sliding compliant contact uses dynamic friction");
	std::printf("%s y=%.9g expected_y=%.9g max_position_error=%.9g impulse=%.9g friction_drift=%.9g dynamic_ratio=%.9g\n",
		accelerationSpring ? "compliant_acceleration" : "compliant_force", double(actor->getGlobalPose().p.y), equilibrium,
		maximumError, reportedImpulse / 100.0, maximumDrift, dynamicFrictionRatio);
	scene->release();
	material->release();
}

void frictionTransition(ContactContext& context)
{
	ContactModifier modifier;
	ContactRecorder recorder;
	PxMaterial* material = context.physics.createMaterial(0.8f, 0.2f, 0.0f);
	PxScene* scene = context.createScene(modifier, recorder);
	if(!scene)
	{
		material->release();
		return;
	}
	context.floor(*scene, *material);
	PxRigidDynamic* actor = context.body(*scene, *material, PxBoxGeometry(PxVec3(0.5f)), PxTransform(PxVec3(0.0f, 0.5f, 0.0f)));
	lockRotation(*actor);
	for(int frame = 0; frame < 100; ++frame)
	{
		step(*scene, recorder);
	}
	const double initialX = actor->getGlobalPose().p.x;
	for(int frame = 0; frame < 100; ++frame)
	{
		actor->addForce(PxVec3(4.905f, 0.0f, 0.0f));
		step(*scene, recorder);
	}
	const double staticDrift = double(actor->getGlobalPose().p.x) - initialX;
	check(std::abs(staticDrift) < 0.002 && std::abs(actor->getLinearVelocity().x) < 0.005f,
		"static friction holds a load above the dynamic threshold");
	const double initialSpeed = actor->getLinearVelocity().x;
	double maximumHeight = actor->getGlobalPose().p.y;
	double maximumVerticalSpeed = 0.0;
	for(int frame = 0; frame < 20; ++frame)
	{
		actor->addForce(PxVec3(9.81f, 0.0f, 0.0f));
		step(*scene, recorder);
		maximumHeight = std::max(maximumHeight, double(actor->getGlobalPose().p.y));
		maximumVerticalSpeed = std::max(maximumVerticalSpeed, std::abs(double(actor->getLinearVelocity().y)));
	}
	const double slidingSpeed = actor->getLinearVelocity().x;
	const double expectedSpeed = initialSpeed + (9.81 - 0.8 * 9.81) * double(TIMESTEP) +
		(9.81 - 0.2 * 9.81) * double(TIMESTEP) * 19.0;
	check(std::abs(slidingSpeed - expectedSpeed) < 0.08,
		"sliding contact switches to dynamic friction");
	check(maximumHeight < 0.55 && maximumVerticalSpeed < 0.8,
		"friction pyramid lift remains bounded");
	for(int frame = 0; frame < 300; ++frame)
		step(*scene, recorder);
	check(std::abs(actor->getLinearVelocity().x) < 0.01f, "friction stops the body and returns to rest");
	std::printf("friction_transition static_drift=%.9g sliding_vx=%.9g expected=%.9g max_y=%.9g max_abs_vy=%.9g final_vx=%.9g\n",
		staticDrift, slidingSpeed, expectedSpeed, maximumHeight, maximumVerticalSpeed, double(actor->getLinearVelocity().x));
	scene->release();
	material->release();
}

void fastSliding(ContactContext& context)
{
	ContactModifier modifier;
	ContactRecorder recorder;
	PxMaterial* material = context.physics.createMaterial(0.5f, 0.5f, 0.0f);
	PxScene* scene = context.createScene(modifier, recorder);
	if(!scene)
	{
		material->release();
		return;
	}
	context.floor(*scene, *material);
	PxRigidDynamic* actor = context.body(*scene, *material, PxBoxGeometry(PxVec3(0.5f)), PxTransform(PxVec3(0.0f, 0.5f, 0.0f)));
	lockRotation(*actor);
	actor->setLinearVelocity(PxVec3(2.0f, 0.0f, 0.0f));

	const double dt = double(TIMESTEP);
	const double supportImpulse = 9.81 * dt;
	double expectedSpeed = 2.0, expectedDistance = 0.0;
	double minimumHeight = 0.5, maximumHeight = 0.5, maximumUpwardSpeed = 0.0;
	int expectedStopFrame = 0, actualStopFrame = 0;
	for(int frame = 1; frame <= 100; ++frame)
	{
		// Coulomb friction removes mu*g*dt before the semi-implicit position
		// update. Clamp at rest so the reference never reverses the body.
		expectedSpeed = std::max(0.0, expectedSpeed - 0.5 * supportImpulse);
		expectedDistance += dt * expectedSpeed;
		if(expectedSpeed == 0.0 && expectedStopFrame == 0)
		{
			expectedStopFrame = frame;
		}
		step(*scene, recorder);
		const PxVec3 velocity = actor->getLinearVelocity();
		const double height = actor->getGlobalPose().p.y;
		if(std::abs(velocity.x) < 0.001f && actualStopFrame == 0)
		{
			actualStopFrame = frame;
		}
		minimumHeight = std::min(minimumHeight, height);
		maximumHeight = std::max(maximumHeight, height);
		maximumUpwardSpeed = std::max(maximumUpwardSpeed, double(velocity.y));
	}
	const double distance = actor->getGlobalPose().p.x;
	check(maximumHeight < 0.55 && minimumHeight > 0.499 && maximumUpwardSpeed < 0.9,
		"fast sliding friction-pyramid lift remains bounded");
	check(std::abs(actualStopFrame - expectedStopFrame) <= 2,
		"fast sliding stops at the Coulomb reference timestep");
	check(std::abs(distance - expectedDistance) < 0.005 && std::abs(actor->getLinearVelocity().x) < 0.0001f,
		"fast sliding matches the Coulomb stopping distance and remains at rest");
	std::printf("fast_sliding min_y=%.9g max_y=%.9g max_upward_vy=%.9g "
		"stop_frame=%d expected_stop_frame=%d distance=%.9g expected_distance=%.9g\n",
		minimumHeight, maximumHeight, maximumUpwardSpeed, actualStopFrame, expectedStopFrame, distance, expectedDistance);
	scene->release();
	material->release();
}

void frictionPrecision(ContactContext& context)
{
	const PxReal scales[] = { 0.1f, 1.0f, 10.0f };
	const PxQuat rotations[] = { PxQuat(PxIdentity), PxQuat(0.73f, PxVec3(1.0f, 2.0f, 3.0f).getNormalized()),
		PxQuat(1.21f, PxVec3(-2.0f, 1.0f, 0.5f).getNormalized()) };
	for(PxU32 scaleIndex = 0; scaleIndex < 3; ++scaleIndex)
	{
		for(PxU32 rotationIndex = 0; rotationIndex < 3; ++rotationIndex)
		{
			const PxReal scale = scales[scaleIndex];
			const PxQuat& rotation = rotations[rotationIndex];
			const PxVec3 normal = rotation.rotate(PxVec3(0.0f, 1.0f, 0.0f)).getNormalized();
			const PxVec3 tangent = rotation.rotate(PxVec3(1.0f, 0.0f, 0.0f)).getNormalized();
			const PxVec3 origin = rotation.rotate(PxVec3(0.17f, 0.23f, -0.31f)) * scale;
			const PxReal gravity = 9.81f * scale;
			ContactModifier modifier;
			ContactRecorder recorder;
			PxMaterial* material = context.physics.createMaterial(0.8f, 0.2f, 0.0f);
			PxScene* scene = context.createScene(modifier, recorder, -normal * gravity);
			if(!scene)
			{
				material->release();
				return;
			}
			PxRigidStatic* floor = PxCreatePlane(context.physics, PxPlane(normal, -normal.dot(origin)), *material);
			scene->addActor(*floor);
			const PxQuat boxRotation = rotation * PxQuat(0.37f, PxVec3(0.0f, 1.0f, 0.0f));
			PxRigidDynamic* actor = context.body(*scene, *material, PxBoxGeometry(PxVec3(0.5f * scale)),
				PxTransform(origin + normal * (0.5f * scale), boxRotation));
			lockRotation(*actor);

			// Scale acceleration with length so the same timestep gives comparable motion.
			// Keep the normal contact/rest offsets and material policy at their usual values.
			for(int frame = 0; frame < 40; ++frame)
				step(*scene, recorder);
			const PxVec3 start = actor->getGlobalPose().p;
			double maximumDrift = 0.0;
			double maximumSpeed = 0.0;
			for(int frame = 0; frame < 60; ++frame)
			{
				// This load is above dynamic friction and below static friction.
				actor->addForce(tangent * (0.5f * gravity));
				step(*scene, recorder);
				const PxVec3 displacement = actor->getGlobalPose().p - start;
				const PxVec3 velocity = actor->getLinearVelocity();
				maximumDrift = std::max(maximumDrift, double((displacement - normal * normal.dot(displacement)).magnitude()));
				maximumSpeed = std::max(maximumSpeed, double((velocity - normal * normal.dot(velocity)).magnitude()));
			}
			check(maximumDrift < 0.0005 * scale && maximumSpeed < 0.002 * scale,
				"static friction holds across geometry scales and orientations");

			// This physical speed is below the anchor uncertainty divided by dt.
			// It must still enter the velocity solve and be stopped by an actual friction impulse.
			actor->setLinearVelocity(tangent * (1e-5f * scale));
			step(*scene, recorder);
			const PxReal slowSpeed = tangent.dot(actor->getLinearVelocity());
			check(std::abs(slowSpeed) < 5e-7f * scale, "small physical tangential velocity is still solved");

			// A resolved sliding speed must remain physical motion with pyramid-friction deceleration.
			actor->setLinearVelocity(tangent * (0.5f * scale));
			const PxVec3 slidingStart = actor->getGlobalPose().p;
			double staticFrictionRatio = -1.0, dynamicFrictionRatio = -1.0;
			for(int frame = 0; frame < 4; ++frame)
			{
				const PxReal speedBefore = tangent.dot(actor->getLinearVelocity());
				step(*scene, recorder);
				const PxReal speedAfter = tangent.dot(actor->getLinearVelocity());
				if(recorder.normalImpulse > 1e-8 * scale)
				{
					const double ratio = (speedBefore - speedAfter) / recorder.normalImpulse;
					if(frame == 0)
						staticFrictionRatio = ratio;
					else
						dynamicFrictionRatio = ratio;
				}
			}
			const PxReal slidingSpeed = tangent.dot(actor->getLinearVelocity());
			const PxReal slidingDistance = tangent.dot(actor->getGlobalPose().p - slidingStart);
			const PxReal supportHeight = normal.dot(actor->getGlobalPose().p - origin);
			check(std::abs(staticFrictionRatio - 0.8) < 0.002 && std::abs(dynamicFrictionRatio - 0.2) < 0.002,
				"sliding contact switches from static to dynamic friction");
			check(slidingSpeed > 0.0f && slidingDistance > 0.005f * scale,
				"resolved sliding motion remains physical");
			check(supportHeight > 0.49f * scale && supportHeight < 0.56f * scale,
				"rotated sliding contact remains close to its support surface");
			std::printf("friction_precision scale=%.9g rotation=%u drift=%.9g static_speed=%.9g slow_speed=%.9g slide_speed=%.9g static_ratio=%.9g dynamic_ratio=%.9g\n",
				double(scale), rotationIndex, maximumDrift, maximumSpeed, double(slowSpeed),
				double(slidingSpeed), staticFrictionRatio, dynamicFrictionRatio);
			scene->release();
			material->release();
		}
	}
}
void rolling(ContactContext& context, bool cylinder)
{
	ContactModifier modifier;
	ContactRecorder recorder;
	PxMaterial* material = context.physics.createMaterial(0.5f, 0.5f, 0.0f);
	PxScene* scene = context.createScene(modifier, recorder);
	if(!scene)
	{
		material->release();
		return;
	}
	context.floor(*scene, *material);
	PxCustomGeometryExt::CylinderCallbacks cylinderCallbacks(1.5f, 0.5f, 2);
	PxCustomGeometry cylinderGeometry(cylinderCallbacks);
	PxCapsuleGeometry capsuleGeometry(0.5f, 0.75f);
	const PxGeometry& geometry = cylinder ? static_cast<const PxGeometry&>(cylinderGeometry) : static_cast<const PxGeometry&>(capsuleGeometry);
	const PxQuat rotation = cylinder ? PxQuat(PxIdentity) : PxQuat(PxHalfPi, PxVec3(0.0f, 1.0f, 0.0f));
	PxRigidDynamic* actor = context.body(*scene, *material, geometry, PxTransform(PxVec3(0.0f, 0.5f, 0.0f), rotation));
	actor->setLinearVelocity(PxVec3(1.0f, 0.0f, 0.0f));
	actor->setAngularVelocity(PxVec3(0.0f, 0.0f, -2.0f));
	double minimumHeight = 1.0;
	double maximumHeight = 0.0;
	double maximumVerticalSpeed = 0.0;
	for(int frame = 0; frame < 200; ++frame)
	{
		step(*scene, recorder);
		const PxTransform pose = actor->getGlobalPose();
		check(pose.isFinite() && actor->getLinearVelocity().isFinite() && actor->getAngularVelocity().isFinite(), "rolling body remains finite");
		minimumHeight = std::min(minimumHeight, double(pose.p.y));
		maximumHeight = std::max(maximumHeight, double(pose.p.y));
		maximumVerticalSpeed = std::max(maximumVerticalSpeed, std::abs(double(actor->getLinearVelocity().y)));
	}
	check(minimumHeight > 0.485 && maximumHeight < 0.515 && maximumVerticalSpeed < 0.08,
		"rolling curved contact has no systematic lift or vertical impulses");
	check(actor->getLinearVelocity().x > 0.85f && actor->getLinearVelocity().x < 1.05f,
		"pure rolling retains translational speed without spurious braking or energy gain");
	check(std::abs(actor->getLinearVelocity().z) < 0.03f, "rolling contact does not drift sideways");
	std::printf("rolling_%s vx=%.9g wz=%.9g min_y=%.9g max_y=%.9g max_abs_vy=%.9g\n", cylinder ? "cylinder" : "capsule",
		double(actor->getLinearVelocity().x), double(actor->getAngularVelocity().z), minimumHeight, maximumHeight, maximumVerticalSpeed);
	scene->release();
	material->release();
}
}

int main(int argc, char** argv)
{
	const char* selection = argc > 1 ? argv[1] : "all";
	const PxSolverType::Enum solver = argc > 2 && std::strcmp(argv[2], "pgs") == 0 ? PxSolverType::ePGS :
		argc > 2 && std::strcmp(argv[2], "tgs") == 0 ? PxSolverType::eTGS : PxSolverType::eNEWTON;
	PxDefaultAllocator allocator;
	ContactErrors errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	PxInitExtensions(*physics, NULL);
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(8);
	ContactContext context = {*physics, *dispatcher, solver};
	const bool all = std::strcmp(selection, "all") == 0;
	if(all || std::strcmp(selection, "conveyor") == 0)
	{
		conveyor(context);
	}
	if(all || std::strcmp(selection, "caps") == 0)
	{
		normalCaps(context, false);
	}
	if(all || std::strcmp(selection, "disabled") == 0)
	{
		normalCaps(context, true);
	}
	if(all || std::strcmp(selection, "compliant") == 0)
	{
		compliant(context, false);
	}
	if(all || std::strcmp(selection, "acceleration") == 0)
	{
		compliant(context, true);
	}
	if(all || std::strcmp(selection, "friction") == 0)
	{
		frictionTransition(context);
	}
	if(all || std::strcmp(selection, "fast_sliding") == 0)
	{
		fastSliding(context);
	}
	if(all || std::strcmp(selection, "friction_precision") == 0)
	{
		frictionPrecision(context);
	}
	if(all || std::strcmp(selection, "capsule") == 0)
	{
		rolling(context, false);
	}
	if(all || std::strcmp(selection, "cylinder") == 0)
	{
		rolling(context, true);
	}
	dispatcher->release();
	PxCloseExtensions();
	physics->release();
	foundation->release();
	std::printf("Native contact tests: failures=%d engine_errors=%d\n", failures, errors.count.load());
	return failures || errors.count.load() ? 1 : 0;
}
