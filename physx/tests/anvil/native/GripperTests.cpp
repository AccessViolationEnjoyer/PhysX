// A two-finger gripper lifts a row of four boxes held only by friction. Each finger is a
// dynamic plate on a prismatic D6 joint (only X free) to a kinematic arm, closed by a
// force-limited linear drive. The boxes sit side by side on the ground; the fingers close on
// the outer faces of the outer boxes, squeezing the row together, and the arm lifts 0.3 m.
// Friction must carry every box: each outer face carries two boxes and the next faces in carry
// one, so the grip must reach 2 m (g + a) / mu while the arm accelerates upwards at a.
// The same gripper also lifts a single curved body, 1 kg and 0.1 m across, held at single
// points or along lines: a sphere, and a cylinder (a 64-sided convex hull) lying with its axis
// along z, gripped on its curved side. Each finger then carries half its weight, so the grip
// must reach m (g + a) / (2 mu). Curved bodies are held for 5 s to show any creep.
//
// usage: AnvilGripperTests [anvil|pgs] [grip-force] [hold-seconds=1] [position-iterations=16]
//                          [shake-amplitude=0] [shake-frequency=1] [boxes|sphere|cylinder]
//   Without a grip force, Anvil is checked at a 150 N grip and just below and above the
//   required grip, for the row of boxes and each curved body; PGS only reports. With one, a
//   single run is reported. The iteration count applies to PGS; every body also takes two
//   velocity iterations. A shake moves the arm up and down sinusoidally through the hold,
//   starting from rest.
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

static const PxReal timestep = 0.01f;
static PxU32 positionIterations = 16;
static PxReal shakeAmplitude = 0.0f, shakeFrequency = 1.0f;
static const PxReal gravity = 9.81f;
static const PxReal friction = 0.5f;
static const PxReal boxSize = 0.1f;
static const PxReal boxMass = 1.0f;
static const int boxCount = 4;
static const PxReal fingerHalf[3] = { 0.01f, 0.04f, 0.06f };
static const PxReal fingerMass = 0.5f;
static const PxReal opening = 0.03f;      // Each finger's initial gap to its box.
static const PxReal armHeight = 0.3f;
static const PxReal lift = 0.3f;
// Every shape's contact offset: pairs make contacts within 1 mm, like MuJoCo's margin.
static const PxReal contactOffset = 0.0005f;
// Phases in steps: settle open, close, lift, then hold.
static const int closeStep = 100, liftStep = 200, liftSteps = 150;
// Curved bodies: radius, the cylinder's half length and its number of sides.
static const PxReal curvedRadius = 0.05f, cylinderHalfLength = 0.05f;
static const int cylinderSides = 64;
static const PxReal curvedHoldSeconds = 5.0f;

struct GripObject
{
	enum Enum
	{
		eBOXES,
		eSPHERE,
		eCYLINDER
	};
};

static const char* objectName(GripObject::Enum object)
{
	return object == GripObject::eBOXES ? "boxes" : object == GripObject::eSPHERE ? "sphere" : "cylinder";
}

// A cylinder with its axis along z, as the convex hull of its two end polygons.
static PxConvexMesh* createCylinder(PxPhysics& physics)
{
	PxVec3 points[2 * cylinderSides];
	for(int i = 0; i < cylinderSides; ++i)
	{
		const PxReal angle = 2.0f * PxPi * PxReal(i) / PxReal(cylinderSides);
		const PxReal x = curvedRadius * PxCos(angle), y = curvedRadius * PxSin(angle);
		points[2 * i] = PxVec3(x, y, -cylinderHalfLength);
		points[2 * i + 1] = PxVec3(x, y, cylinderHalfLength);
	}
	PxConvexMeshDesc description;
	description.points.count = 2 * cylinderSides;
	description.points.stride = sizeof(PxVec3);
	description.points.data = points;
	description.flags = PxConvexFlag::eCOMPUTE_CONVEX;
	return PxCreateConvexMesh(PxCookingParams(physics.getTolerancesScale()), description, physics.getPhysicsInsertionCallback());
}

static PxConvexMesh* cylinderMesh = NULL;

static void setContactOffset(PxRigidActor& actor)
{
	PxShape* shape;
	for(PxU32 i = 0; i < actor.getNbShapes(); ++i)
	{
		actor.getShapes(&shape, 1, i);
		shape->setContactOffset(contactOffset);
	}
}

// Arm height at a step: a smooth 0.3 m rise over the lift phase.
static PxReal armLift(int step)
{
	const PxReal t = PxClamp(PxReal(step - liftStep) / liftSteps, 0.0f, 1.0f);
	const PxReal hold = PxMax(0.0f, PxReal(step - liftStep - liftSteps) * timestep);
	return lift * t * t * (3.0f - 2.0f * t) + shakeAmplitude * (1.0f - PxCos(2.0f * PxPi * shakeFrequency * hold));
}

// The grip each finger needs at the lift's peak acceleration, 6 h / T^2: two boxes of the row
// on each outer face, or half a single curved body.
static PxReal requiredGrip(GripObject::Enum object)
{
	const PxReal duration = liftSteps * timestep;
	const PxReal carried = object == GripObject::eBOXES ? 2.0f * boxMass / friction : 0.5f * boxMass / friction;
	return carried * (gravity + 6.0f * lift / (duration * duration));
}

struct GripResult
{
	PxReal minimumLift = PX_MAX_F32;   // Lowest box's rise.
	PxReal maximumSlip = 0.0f;         // Largest vertical slip against the fingers since closing.
	PxReal maximumTilt = 0.0f;
	PxReal maximumOverlap = 0.0f;      // Neighbouring boxes squeezed into each other.
	PxReal fingerGap = 0.0f;           // Distance between the fingers' inner faces.
	PxReal speed = 0.0f;               // Fastest box at the end.
	PxReal creep = 0.0f;               // Fastest box against the fingers at the end.
	double meanStepUs = 0.0;
};

static GripResult simulate(PxPhysics& physics, PxCpuDispatcher& dispatcher, bool pgs, PxReal gripForce, PxReal holdSeconds, GripObject::Enum object = GripObject::eBOXES)
{
	PxSceneDesc description(physics.getTolerancesScale());
	description.gravity = PxVec3(0.0f, -gravity, 0.0f);
	description.cpuDispatcher = &dispatcher;
	description.filterShader = PxDefaultSimulationFilterShader;
	description.solverType = pgs ? PxSolverType::ePGS : PxSolverType::eANVIL;
	description.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
	PxScene* scene = physics.createScene(description);
	PxMaterial* material = physics.createMaterial(friction, friction, 0.0f);
	PxRigidStatic* ground = PxCreatePlane(physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *material);
	setContactOffset(*ground);
	scene->addActor(*ground);

	// The held bodies: the row of boxes, or one curved body; each rests with its centre at
	// restHeight and spans width along x.
	std::vector<PxRigidDynamic*> boxes;
	const bool row = object == GripObject::eBOXES;
	const int bodyCount = row ? boxCount : 1;
	const PxReal restHeight = row ? 0.5f * boxSize : curvedRadius;
	const PxReal width = row ? boxCount * boxSize : 2.0f * curvedRadius;
	for(int i = 0; i < bodyCount; ++i)
	{
		const PxReal x = row ? (i - 0.5f * (boxCount - 1)) * boxSize : 0.0f;
		const PxTransform pose(PxVec3(x, restHeight, 0.0f));
		PxRigidDynamic* box = object == GripObject::eBOXES ? PxCreateDynamic(physics, pose, PxBoxGeometry(PxVec3(0.5f * boxSize)), *material, 1.0f) :
			object == GripObject::eSPHERE ? PxCreateDynamic(physics, pose, PxSphereGeometry(curvedRadius), *material, 1.0f) :
			PxCreateDynamic(physics, pose, PxConvexMeshGeometry(cylinderMesh), *material, 1.0f);
		// Contacts a step ahead of fast bodies: the offset stays the resting precision.
		box->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		PxRigidBodyExt::setMassAndUpdateInertia(*box, boxMass);
		box->setSleepThreshold(0.0f);
		box->setSolverIterationCounts(positionIterations, 2);
		setContactOffset(*box);
		scene->addActor(*box);
		boxes.push_back(box);
	}

	// The arm has no shape; the fingers hang from it at the boxes' mid-height.
	PxRigidDynamic* arm = physics.createRigidDynamic(PxTransform(PxVec3(0.0f, armHeight, 0.0f)));
	// Contacts a step ahead of fast bodies: the offset stays the resting precision.
	arm->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
	arm->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
	scene->addActor(*arm);
	PxRigidDynamic* fingers[2];
	PxD6Joint* joints[2];
	const PxReal fingerX = 0.5f * width + opening + fingerHalf[0];
	for(int side = 0; side < 2; ++side)
	{
		const PxReal sign = side ? 1.0f : -1.0f;
		const PxVec3 position(sign * fingerX, restHeight, 0.0f);
		fingers[side] = PxCreateDynamic(physics, PxTransform(position), PxBoxGeometry(PxVec3(fingerHalf[0], fingerHalf[1], fingerHalf[2])), *material, 1.0f);
		// Contacts a step ahead of fast bodies: the offset stays the resting precision.
		fingers[side]->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		PxRigidBodyExt::setMassAndUpdateInertia(*fingers[side], fingerMass);
		fingers[side]->setSleepThreshold(0.0f);
		fingers[side]->setSolverIterationCounts(positionIterations, 2);
		setContactOffset(*fingers[side]);
		scene->addActor(*fingers[side]);
		joints[side] = PxD6JointCreate(physics, arm, PxTransform(position - arm->getGlobalPose().p), fingers[side], PxTransform(PxIdentity));
		joints[side]->setMotion(PxD6Axis::eX, PxD6Motion::eFREE);
		joints[side]->setConstraintFlag(PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);
		// Stiff enough to hold the fingers open; closing drives past the boxes, so the force limit sets the grip.
		joints[side]->setDrive(PxD6Drive::eX, PxD6JointDrive(5000.0f, 200.0f, gripForce, false));
		joints[side]->setDrivePosition(PxTransform(PxIdentity));
	}

	GripResult result;
	const int endStep = liftStep + liftSteps + int(holdSeconds / timestep);
	double totalMs = 0.0;
	std::vector<PxReal> closedOffset(bodyCount);
	for(int step = 0; step < endStep; ++step)
	{
		if(step == closeStep)
		{
			for(int side = 0; side < 2; ++side)
				joints[side]->setDrivePosition(PxTransform(PxVec3((side ? -1.0f : 1.0f) * (opening + 0.05f), 0.0f, 0.0f)));
		}
		arm->setKinematicTarget(PxTransform(PxVec3(0.0f, armHeight + armLift(step + 1), 0.0f)));
		const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		scene->simulate(timestep);
		scene->fetchResults(true);
		if(step > 0)
			totalMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		const PxReal fingerY = 0.5f * (fingers[0]->getGlobalPose().p.y + fingers[1]->getGlobalPose().p.y);
		if(step + 1 == liftStep)
		{
			for(int i = 0; i < bodyCount; ++i)
				closedOffset[i] = boxes[i]->getGlobalPose().p.y - fingerY;
		}
		if(step + 1 >= liftStep)
		{
			for(int i = 0; i < bodyCount; ++i)
				result.maximumSlip = std::max(result.maximumSlip, std::abs(boxes[i]->getGlobalPose().p.y - fingerY - closedOffset[i]));
		}
		for(int i = 0; i < bodyCount; ++i)
		{
			result.maximumTilt = std::max(result.maximumTilt, boxes[i]->getGlobalPose().q.getAngle());
			if(i + 1 < bodyCount)
				result.maximumOverlap = std::max(result.maximumOverlap, boxSize - (boxes[i + 1]->getGlobalPose().p.x - boxes[i]->getGlobalPose().p.x));
		}
	}
	const PxReal fingerSpeed = 0.5f * (fingers[0]->getLinearVelocity().y + fingers[1]->getLinearVelocity().y);
	for(int i = 0; i < bodyCount; ++i)
	{
		result.minimumLift = std::min(result.minimumLift, boxes[i]->getGlobalPose().p.y - restHeight);
		result.speed = std::max(result.speed, boxes[i]->getLinearVelocity().magnitude());
		result.creep = std::max(result.creep, std::abs(boxes[i]->getLinearVelocity().y - fingerSpeed));
	}
	result.fingerGap = fingers[1]->getGlobalPose().p.x - fingers[0]->getGlobalPose().p.x - 2.0f * fingerHalf[0];
	result.meanStepUs = 1e3 * totalMs / (endStep - 1);
	printf("gripper %s iterations=%u grip_n=%.1f mean_step_us=%.2f hold_s=%.1f creep_mm_s=%.4f minimum_lift_mm=%.3f maximum_slip_mm=%.4f maximum_tilt_deg=%.4f"
		" row_width_mm=%.4f maximum_box_overlap_mm=%.4f final_speed_mm_s=%.4f object=%s\n", pgs ? "PGS" : "Anvil", positionIterations, double(gripForce), result.meanStepUs,
		double(holdSeconds), result.creep * 1e3, result.minimumLift * 1e3, result.maximumSlip * 1e3, result.maximumTilt * 180.0 / PxPi,
		result.fingerGap * 1e3, result.maximumOverlap * 1e3, result.speed * 1e3, objectName(object));
	scene->release();
	material->release();
	return result;
}

int main(int argc, char** argv)
{
	const bool pgs = argc > 1 && std::strcmp(argv[1], "pgs") == 0;
	const PxReal holdSeconds = argc > 3 ? PxReal(std::atof(argv[3])) : 1.0f;
	positionIterations = argc > 4 ? PxU32(std::atoi(argv[4])) : 16;
	shakeAmplitude = argc > 5 ? PxReal(std::atof(argv[5])) : 0.0f;
	shakeFrequency = argc > 6 ? PxReal(std::atof(argv[6])) : 1.0f;
	PxDefaultAllocator allocator;
	PxDefaultErrorCallback errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	PxInitExtensions(*physics, NULL);
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(1);
	cylinderMesh = createCylinder(*physics);
	const char* only = argc > 7 ? argv[7] : NULL;
	if(argc > 2)
	{
		const GripObject::Enum object = only && std::strcmp(only, "sphere") == 0 ? GripObject::eSPHERE : only && std::strcmp(only, "cylinder") == 0 ? GripObject::eCYLINDER : GripObject::eBOXES;
		simulate(*physics, *dispatcher, pgs, PxReal(std::atof(argv[2])), holdSeconds, object);
	}
	else
	{
		const GripResult firm = simulate(*physics, *dispatcher, pgs, 150.0f, holdSeconds);
		// The row needs 2 m (g + a) / mu per finger during the lift (42.4 N here).
		const PxReal required = requiredGrip(GripObject::eBOXES);
		printf("gripper required_grip_n=%.2f\n", double(required));
		const GripResult weak = simulate(*physics, *dispatcher, pgs, 0.9f * required, holdSeconds);
		const GripResult enough = simulate(*physics, *dispatcher, pgs, 1.1f * required, holdSeconds);
		if(!pgs)
		{
			check(firm.minimumLift > lift - 0.005f, "a 150 N grip lifts every box with the gripper");
			check(firm.maximumSlip < 0.002f, "boxes slip less than 2 mm against the fingers");
			check(firm.maximumTilt < 0.5f * PxPi / 180.0f, "boxes stay level");
			check(std::abs(firm.fingerGap - boxCount * boxSize) < 0.001f, "the fingers hold the row at its width");
			check(firm.speed < 0.001f, "the lifted row comes to rest");
			check(weak.minimumLift < 0.01f, "a grip 10% below the friction requirement drops the row");
			check(enough.minimumLift > 0.25f && enough.maximumTilt < 0.5f * PxPi / 180.0f, "a grip 10% above the friction requirement lifts the row");
		}
		// Curved bodies, held at points (sphere) or along lines (cylinder); each finger needs
		// m (g + a) / (2 mu) (5.3 N here).
		const GripObject::Enum curved[2] = { GripObject::eSPHERE, GripObject::eCYLINDER };
		for(int i = 0; i < 2; ++i)
		{
			const GripObject::Enum object = curved[i];
			const PxReal curvedRequired = requiredGrip(object);
			const GripResult curvedFirm = simulate(*physics, *dispatcher, pgs, 150.0f, curvedHoldSeconds, object);
			const GripResult curvedWeak = simulate(*physics, *dispatcher, pgs, 0.9f * curvedRequired, curvedHoldSeconds, object);
			const GripResult curvedEnough = simulate(*physics, *dispatcher, pgs, 1.1f * curvedRequired, curvedHoldSeconds, object);
			if(!pgs)
			{
				check(curvedFirm.minimumLift > lift - 0.005f, object == GripObject::eSPHERE ? "a 150 N grip lifts the sphere" : "a 150 N grip lifts the cylinder");
				check(curvedFirm.maximumSlip < 0.002f, object == GripObject::eSPHERE ? "the sphere slips less than 2 mm" : "the cylinder slips less than 2 mm");
				check(curvedFirm.speed < 0.001f, object == GripObject::eSPHERE ? "the lifted sphere comes to rest" : "the lifted cylinder comes to rest");
				check(curvedWeak.minimumLift < 0.01f, object == GripObject::eSPHERE ? "a grip 10% below the requirement drops the sphere" : "a grip 10% below the requirement drops the cylinder");
				check(curvedEnough.minimumLift > lift - 0.005f && curvedEnough.maximumSlip < 0.002f,
					object == GripObject::eSPHERE ? "a grip 10% above the requirement holds the sphere" : "a grip 10% above the requirement holds the cylinder");
			}
		}
	}
	cylinderMesh->release();
	dispatcher->release();
	PxCloseExtensions();
	physics->release();
	foundation->release();
	printf("Gripper (%s): %d failures\n", pgs ? "PGS" : "Anvil", failures);
	return failures ? 1 : 0;
}
