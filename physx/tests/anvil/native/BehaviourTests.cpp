#include "PxPhysicsAPI.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace physx;

// Physical behaviour checks against analytic expectations:
// - bouncing_balls: restitution sets the rebound speed and height.
// - split_conveyors: a case straddling two belts at different speeds turns at the rate Coulomb
//   friction predicts for uniform pressure and does not drift sideways.
// - sheet_stack: 100 sheets of 1 mm and 1 g stay stacked without penetration or drift.
// usage: BehaviourTests [all|bouncing_balls|split_conveyors|sheet_stack] [anvil|pgs] [threads=1]
namespace
{
const PxReal TIMESTEP = 0.01f;
const PxReal GRAVITY = 9.81f;
// Contact offset per shape; benchmarks never use PhysX's 2 cm default.
const PxReal CONTACT_OFFSET = 0.0005f;
int failures = 0;

void check(bool condition, const char* message)
{
	if(!condition)
	{
		std::printf("FAIL %s\n", message);
		++failures;
	}
}

// Belt shapes carry word0 = 1 and their surface speed in mm/s in word1.
class BeltContacts : public PxContactModifyCallback
{
public:
	virtual void onContactModify(PxContactModifyPair* const pairs, PxU32 count) PX_OVERRIDE
	{
		for(PxU32 i = 0; i < count; ++i)
		{
			// Target velocity is actor[0] relative to actor[1].
			const bool beltFirst = pairs[i].shape[0]->getSimulationFilterData().word0 != 0;
			const PxFilterData belt = pairs[i].shape[beltFirst ? 0 : 1]->getSimulationFilterData();
			const PxReal speed = PxReal(belt.word1) * 1.0e-3f;
			const PxVec3 velocity(beltFirst ? -speed : speed, 0.0f, 0.0f);
			const PxU32 contactCount = pairs[i].contacts.size();
			for(PxU32 j = 0; j < contactCount; ++j)
			{
				pairs[i].contacts.setTargetVelocity(j, velocity);
			}
		}
	}
};

PxFilterFlags filterShader(PxFilterObjectAttributes, PxFilterData data0, PxFilterObjectAttributes, PxFilterData data1, PxPairFlags& flags, const void*, PxU32)
{
	flags = PxPairFlag::eCONTACT_DEFAULT;
	if(data0.word0 || data1.word0)
	{
		flags |= PxPairFlag::eMODIFY_CONTACTS;
	}
	return PxFilterFlag::eDEFAULT;
}

struct Context
{
	PxPhysics* physics;
	PxDefaultCpuDispatcher* dispatcher;
	BeltContacts belts;
	bool anvil;

	PxScene* createScene()
	{
		PxSceneDesc desc(physics->getTolerancesScale());
		desc.cpuDispatcher = dispatcher;
		desc.filterShader = filterShader;
		desc.contactModifyCallback = &belts;
		desc.gravity = PxVec3(0.0f, -GRAVITY, 0.0f);
		desc.solverType = anvil ? PxSolverType::eANVIL : PxSolverType::ePGS;
		desc.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
		PxScene* scene = physics->createScene(desc);
		check(scene != NULL, "scene creation");
		return scene;
	}

	PxShape* shape(const PxGeometry& geometry, PxMaterial& material, const PxTransform& localPose = PxTransform(PxIdentity))
	{
		PxShape* result = physics->createShape(geometry, material, true);
		result->setContactOffset(CONTACT_OFFSET);
		result->setRestOffset(0.0f);
		result->setLocalPose(localPose);
		return result;
	}

	void staticBox(PxScene& scene, PxMaterial& material, const PxVec3& center, const PxVec3& halfExtents, PxU32 beltSpeedMm = 0)
	{
		PxRigidStatic* actor = physics->createRigidStatic(PxTransform(center));
		PxShape* box = shape(PxBoxGeometry(halfExtents), material);
		if(beltSpeedMm)
		{
			box->setSimulationFilterData(PxFilterData(1, beltSpeedMm, 0, 0));
		}
		actor->attachShape(*box);
		box->release();
		scene.addActor(*actor);
	}

	PxRigidDynamic* dynamicBody(PxScene& scene, const PxTransform& pose)
	{
		PxRigidDynamic* actor = physics->createRigidDynamic(pose);
		actor->setLinearDamping(0.0f);
		actor->setAngularDamping(0.0f);
		actor->setSleepThreshold(0.0f);
		scene.addActor(*actor);
		return actor;
	}
};

void step(PxScene& scene)
{
	scene.simulate(TIMESTEP);
	scene.fetchResults(true);
}

// Balls dropped onto the floor. The first rebound leaves at the restitution times the impact
// speed and rises to restitution squared times the drop; later bounces never gain height, and
// the balls come to rest on the floor once rebounds fall below the bounce threshold. Without
// continuous collision detection the impact falls part-way through a step, so the apex may be
// off, and the ball may sink, by up to one step of travel at the impact speed.
void bouncingBalls(Context& context)
{
	PxScene* scene = context.createScene();
	PxMaterial* floorMaterial = context.physics->createMaterial(0.5f, 0.5f, 0.0f);
	context.staticBox(*scene, *floorMaterial, PxVec3(0.0f, -0.5f, 0.0f), PxVec3(2.0f, 0.5f, 2.0f));
	const PxReal radius = 0.05f, dropHeight = 1.0f;
	const PxReal restitutions[] = { 0.5f, 0.8f };
	const int ballCount = 2;
	PxRigidDynamic* balls[ballCount];
	PxMaterial* ballMaterials[ballCount];
	for(int i = 0; i < ballCount; ++i)
	{
		// The ball's restitution applies: the larger of the pair's values is used.
		ballMaterials[i] = context.physics->createMaterial(0.5f, 0.5f, restitutions[i]);
		ballMaterials[i]->setRestitutionCombineMode(PxCombineMode::eMAX);
		balls[i] = context.dynamicBody(*scene, PxTransform(PxVec3(-0.5f + 1.0f * PxReal(i), dropHeight, 0.0f)));
		PxShape* sphere = context.shape(PxSphereGeometry(radius), *ballMaterials[i]);
		balls[i]->attachShape(*sphere);
		sphere->release();
		PxRigidBodyExt::setMassAndUpdateInertia(*balls[i], 0.2f);
	}
	// A bounce is a sign change of the vertical velocity with a rebound faster than 5 cm/s.
	const int frames = 600;
	const double bounceSpeed = 0.05;
	double previousVelocity[ballCount], impactSpeed[ballCount], reboundSpeed[ballCount], flightApex[ballCount], firstApex[ballCount], lastApex[ballCount], lowest[ballCount];
	int bounces[ballCount];
	bool gained[ballCount];
	for(int i = 0; i < ballCount; ++i)
	{
		previousVelocity[i] = impactSpeed[i] = reboundSpeed[i] = flightApex[i] = firstApex[i] = lastApex[i] = 0.0;
		lowest[i] = dropHeight;
		bounces[i] = 0;
		gained[i] = false;
	}
	for(int frame = 0; frame < frames; ++frame)
	{
		step(*scene);
		for(int i = 0; i < ballCount; ++i)
		{
			const double velocity = balls[i]->getLinearVelocity().y;
			const double height = balls[i]->getGlobalPose().p.y;
			lowest[i] = std::min(lowest[i], height);
			if(previousVelocity[i] < 0.0 && velocity > bounceSpeed)
			{
				if(bounces[i] == 0)
				{
					impactSpeed[i] = -previousVelocity[i];
					reboundSpeed[i] = velocity;
				}
				else
				{
					// The flight that just ended: the first one sets the first apex, and no
					// flight may rise above the one before it.
					if(bounces[i] == 1)
					{
						firstApex[i] = flightApex[i];
					}
					else
					{
						gained[i] = gained[i] || flightApex[i] > lastApex[i] + 1.0e-3;
					}
					lastApex[i] = flightApex[i];
				}
				++bounces[i];
				flightApex[i] = height;
			}
			flightApex[i] = std::max(flightApex[i], height);
			previousVelocity[i] = velocity;
		}
	}
	for(int i = 0; i < ballCount; ++i)
	{
		if(bounces[i] == 1)
		{
			firstApex[i] = flightApex[i];
		}
	}
	char message[256];
	for(int i = 0; i < ballCount; ++i)
	{
		const double e = restitutions[i];
		const double firstApexExpected = radius + e * e * (dropHeight - radius);
		const double speedRatio = impactSpeed[i] > 0.0 ? reboundSpeed[i] / impactSpeed[i] : 0.0;
		const double finalSpeed = balls[i]->getLinearVelocity().magnitude();
		const double finalHeight = balls[i]->getGlobalPose().p.y;
		std::printf("bouncing_ball restitution=%.2f bounces=%d rebound_ratio=%.4f first_apex_m=%.4f expected_apex_m=%.4f max_penetration_mm=%.4f final_height_m=%.5f final_speed_mm_s=%.4f\n",
			e, bounces[i], speedRatio, firstApex[i], firstApexExpected, (radius - lowest[i]) * 1e3, finalHeight, finalSpeed * 1e3);
		std::snprintf(message, sizeof(message), "bouncing_ball %.2f: rebound speed is restitution times impact speed (within 5%%)", e);
		check(bounces[i] > 0 && std::abs(speedRatio / e - 1.0) < 0.05, message);
		std::snprintf(message, sizeof(message), "bouncing_ball %.2f: first rebound rises to restitution squared times the drop (within 5%% and a step's travel)", e);
		check(std::abs(firstApex[i] - firstApexExpected) < 0.05 * (firstApexExpected - radius) + impactSpeed[i] * TIMESTEP, message);
		std::snprintf(message, sizeof(message), "bouncing_ball %.2f: no bounce rises above the previous one", e);
		check(!gained[i], message);
		std::snprintf(message, sizeof(message), "bouncing_ball %.2f: ball comes to rest on the floor", e);
		check(finalSpeed < 1.0e-2 && std::abs(finalHeight - radius) < 1.0e-3, message);
		std::snprintf(message, sizeof(message), "bouncing_ball %.2f: ball never sinks more than a step's travel into the floor", e);
		check(radius - lowest[i] < impactSpeed[i] * TIMESTEP, message);
	}
	scene->release();
	floorMaterial->release();
	for(int i = 0; i < ballCount; ++i)
	{
		ballMaterials[i]->release();
	}
}

// A frictional load on a belt: position relative to the case's centre of mass (x, z), the belt's
// speed along x, and the share of the case's weight it carries.
struct BeltLoad
{
	double x;
	double z;
	double speed;
	double weight;
};

// The yaw rate at which Coulomb friction exerts no torque about the centre of mass. Each load's
// friction acts along its belt's velocity relative to the case at that point.
double expectedYawRate(const std::vector<BeltLoad>& loads, double velocityX, double velocityZ)
{
	double low = -20.0, high = 20.0;
	for(int iteration = 0; iteration < 200; ++iteration)
	{
		const double yaw = 0.5 * (low + high);
		double torque = 0.0;
		for(size_t i = 0; i < loads.size(); ++i)
		{
			const BeltLoad& load = loads[i];
			// The case's velocity there is v + w y x r = (vx + w z, 0, vz - w x).
			const double relativeX = load.speed - (velocityX + yaw * load.z);
			const double relativeZ = -(velocityZ - yaw * load.x);
			const double length = std::sqrt(relativeX * relativeX + relativeZ * relativeZ);
			if(length > 0.0)
			{
				torque += load.weight * (load.z * relativeX - load.x * relativeZ) / length;
			}
		}
		// Friction torque decreases as the yaw rate increases.
		if(torque > 0.0)
		{
			low = yaw;
		}
		else
		{
			high = yaw;
		}
	}
	return 0.5 * (low + high);
}

// Uniform pressure over the case's rotated footprint, sampled on a grid; the belt under each cell
// is the one on its side of the seam at z = 0.
void uniformLoads(const PxTransform& pose, const PxVec3& centre, const PxVec3& half, double fastSpeed, double slowSpeed, std::vector<BeltLoad>& loads)
{
	const int cells = 48;
	loads.clear();
	for(int i = 0; i < cells; ++i)
	{
		for(int j = 0; j < cells; ++j)
		{
			const PxVec3 local(half.x * (2.0f * (PxReal(i) + 0.5f) / PxReal(cells) - 1.0f), -half.y, half.z * (2.0f * (PxReal(j) + 0.5f) / PxReal(cells) - 1.0f));
			const PxVec3 world = pose.transform(local);
			const BeltLoad load = { double(world.x - centre.x), double(world.z - centre.z), world.z < 0.0f ? fastSpeed : slowSpeed, 1.0 };
			loads.push_back(load);
		}
	}
}

// Loads at the corners of each belt's overlap with the footprint, as box collision reports them:
// each belt carries half the weight, shared equally by its overlap's corners.
void cornerLoads(const PxTransform& pose, const PxVec3& centre, const PxVec3& half, double fastSpeed, double slowSpeed, std::vector<BeltLoad>& loads)
{
	PxVec3 corners[4];
	for(int i = 0; i < 4; ++i)
	{
		// Counterclockwise around the bottom face.
		const PxReal sx = (i == 0 || i == 3) ? -1.0f : 1.0f, sz = i < 2 ? -1.0f : 1.0f;
		corners[i] = pose.transform(PxVec3(sx * half.x, -half.y, sz * half.z));
	}
	loads.clear();
	for(int side = 0; side < 2; ++side)
	{
		// Clip the footprint to z < 0 (side 0) or z > 0 (side 1).
		std::vector<PxVec3> polygon;
		for(int i = 0; i < 4; ++i)
		{
			const PxVec3& a = corners[i];
			const PxVec3& b = corners[(i + 1) % 4];
			const bool insideA = side == 0 ? a.z < 0.0f : a.z > 0.0f;
			const bool insideB = side == 0 ? b.z < 0.0f : b.z > 0.0f;
			if(insideA)
			{
				polygon.push_back(a);
			}
			if(insideA != insideB)
			{
				polygon.push_back(a + (b - a) * (a.z / (a.z - b.z)));
			}
		}
		for(size_t i = 0; i < polygon.size(); ++i)
		{
			const BeltLoad load = { double(polygon[i].x - centre.x), double(polygon[i].z - centre.z), side == 0 ? fastSpeed : slowSpeed, 0.5 / double(polygon.size()) };
			loads.push_back(load);
		}
	}
}

// A case straddles two belts running at different speeds, half of its base on each. Both halves
// slide, so Coulomb friction turns the case at the rate where the friction torque about its
// centre vanishes, and the two belts' sideways forces cancel, so it must not drift across them.
// The analytic rate assumes uniform pressure over the base. The rate from the corners of each
// belt's contact area, where box collision places contact points, is reported for comparison.
void splitConveyors(Context& context)
{
	PxScene* scene = context.createScene();
	const PxReal friction = 0.5f;
	PxMaterial* material = context.physics->createMaterial(friction, friction, 0.0f);
	const PxReal fastSpeed = 0.6f, slowSpeed = 0.4f;
	const PxReal beltHalfWidth = 0.3f, beltHalfLength = 3.0f, beltHalfThickness = 0.1f;
	// The fast belt covers z < 0, the slow one z > 0; their tops are at y = 0.
	context.staticBox(*scene, *material, PxVec3(0.0f, -beltHalfThickness, -beltHalfWidth), PxVec3(beltHalfLength, beltHalfThickness, beltHalfWidth), PxU32(fastSpeed * 1000.0f + 0.5f));
	context.staticBox(*scene, *material, PxVec3(0.0f, -beltHalfThickness, beltHalfWidth), PxVec3(beltHalfLength, beltHalfThickness, beltHalfWidth), PxU32(slowSpeed * 1000.0f + 0.5f));
	const PxVec3 half(0.15f, 0.1f, 0.15f);
	const PxReal averageSpeed = 0.5f * (fastSpeed + slowSpeed);
	PxRigidDynamic* body = context.dynamicBody(*scene, PxTransform(PxVec3(-1.0f, half.y, 0.0f)));
	PxShape* box = context.shape(PxBoxGeometry(half), *material);
	body->attachShape(*box);
	box->release();
	PxRigidBodyExt::setMassAndUpdateInertia(*body, 10.0f);
	body->setLinearVelocity(PxVec3(averageSpeed, 0.0f, 0.0f));
	const int frames = 70, settleFrames = 20;
	double measured = 0.0, expected = 0.0, cornerExpected = 0.0, speedError = 0.0;
	std::vector<BeltLoad> loads;
	const PxReal startZ = body->getGlobalPose().p.z;
	for(int frame = 0; frame < frames; ++frame)
	{
		step(*scene);
		if(frame < settleFrames)
		{
			continue;
		}
		const PxTransform pose = body->getGlobalPose();
		const PxVec3 centre = pose.transform(body->getCMassLocalPose().p);
		const PxVec3 velocity = body->getLinearVelocity();
		uniformLoads(pose, centre, half, fastSpeed, slowSpeed, loads);
		expected += expectedYawRate(loads, velocity.x, velocity.z);
		cornerLoads(pose, centre, half, fastSpeed, slowSpeed, loads);
		cornerExpected += expectedYawRate(loads, velocity.x, velocity.z);
		measured += body->getAngularVelocity().y;
		speedError = std::max(speedError, std::abs(double(velocity.x) - averageSpeed) / averageSpeed);
	}
	measured /= frames - settleFrames;
	expected /= frames - settleFrames;
	cornerExpected /= frames - settleFrames;
	const PxTransform pose = body->getGlobalPose();
	const double drift = std::abs(double(pose.transform(body->getCMassLocalPose().p).z) - startZ);
	std::printf("split_conveyors yaw_rate_rad_s=%.5f expected_rad_s=%.5f ratio=%.4f corner_model_rad_s=%.5f corner_ratio=%.4f sideways_drift_mm=%.4f max_speed_error=%.4f yaw_deg=%.2f\n",
		measured, expected, measured / expected, cornerExpected, measured / cornerExpected, drift * 1e3, speedError, 2.0 * std::asin(std::min(1.0, std::abs(double(pose.q.y)))) * 180.0 / PxPi);
	check(std::abs(measured / expected - 1.0) < 0.1, "split_conveyors: case turns at the Coulomb friction rate for uniform pressure (within 10%)");
	check(drift < 1.0e-3, "split_conveyors: case does not drift sideways (below 1 mm)");
	check(speedError < 0.02, "split_conveyors: case moves at the belts' mean speed (within 2%)");
	scene->release();
	material->release();
}

// One hundred sheets 1 mm thick weighing 1 g each, stacked on the floor. Each interface should
// hold with at most a few micrometres of overlap, the stack should stop settling, and no sheet
// should creep sideways or tilt. The overlaps add up over 99 interfaces, so the stack is shorter
// than 100 mm by about a hundred times the overlap.
void sheetStack(Context& context)
{
	PxScene* scene = context.createScene();
	PxMaterial* material = context.physics->createMaterial(0.5f, 0.5f, 0.0f);
	context.staticBox(*scene, *material, PxVec3(0.0f, -0.5f, 0.0f), PxVec3(1.0f, 0.5f, 1.0f));
	const int sheetCount = 100;
	const PxVec3 half(0.1f, 0.0005f, 0.1f);
	std::vector<PxRigidDynamic*> sheets;
	for(int i = 0; i < sheetCount; ++i)
	{
		PxRigidDynamic* sheet = context.dynamicBody(*scene, PxTransform(PxVec3(0.0f, half.y * PxReal(2 * i + 1), 0.0f)));
		PxShape* shape = context.shape(PxBoxGeometry(half), *material);
		sheet->attachShape(*shape);
		shape->release();
		PxRigidBodyExt::setMassAndUpdateInertia(*sheet, 0.001f);
		sheets.push_back(sheet);
	}
	const int frames = 300, settleFrames = 50;
	double maximumOverlap = 0.0, maximumFloorPenetration = 0.0, maximumTilt = 0.0, maximumSpeed = 0.0;
	// Top of the stack when the last 100 frames begin, to measure further settling.
	double settlingStart = 0.0;
	for(int frame = 0; frame < frames; ++frame)
	{
		step(*scene);
		if(frame == frames - 101)
		{
			settlingStart = sheets.back()->getGlobalPose().p.y;
		}
		if(frame < settleFrames)
		{
			continue;
		}
		// Overlap: the next sheet's lowest corner below this sheet's highest corner.
		double previousTop = 0.0;
		for(int i = 0; i < sheetCount; ++i)
		{
			const PxTransform pose = sheets[size_t(i)]->getGlobalPose();
			double bottom = PX_MAX_F64, top = -PX_MAX_F64;
			for(int corner = 0; corner < 8; ++corner)
			{
				const PxVec3 local((corner & 1 ? 1.0f : -1.0f) * half.x, (corner & 2 ? 1.0f : -1.0f) * half.y, (corner & 4 ? 1.0f : -1.0f) * half.z);
				const double y = pose.transform(local).y;
				bottom = std::min(bottom, y);
				top = std::max(top, y);
			}
			if(i == 0)
			{
				maximumFloorPenetration = std::max(maximumFloorPenetration, -bottom);
			}
			else
			{
				maximumOverlap = std::max(maximumOverlap, previousTop - bottom);
			}
			previousTop = top;
			maximumTilt = std::max(maximumTilt, double(PxAcos(PxClamp(pose.q.rotate(PxVec3(0.0f, 1.0f, 0.0f)).y, -1.0f, 1.0f))));
			if(frame == frames - 1)
			{
				maximumSpeed = std::max(maximumSpeed, double(sheets[size_t(i)]->getLinearVelocity().magnitude()));
			}
		}
	}
	double maximumDrift = 0.0;
	for(int i = 0; i < sheetCount; ++i)
	{
		const PxVec3 p = sheets[size_t(i)]->getGlobalPose().p;
		maximumDrift = std::max(maximumDrift, double(PxVec2(p.x, p.z).magnitude()));
	}
	const double topHeight = sheets.back()->getGlobalPose().p.y + half.y;
	const double compression = 2.0 * half.y * sheetCount - topHeight;
	const double settling = std::abs(sheets.back()->getGlobalPose().p.y - settlingStart);
	std::printf("sheet_stack max_overlap_um=%.3f floor_penetration_um=%.3f compression_um=%.3f settling_last_second_um=%.4f max_drift_mm=%.5f max_tilt_deg=%.5f final_max_speed_mm_s=%.5f\n",
		maximumOverlap * 1e6, maximumFloorPenetration * 1e6, compression * 1e6, settling * 1e6, maximumDrift * 1e3, maximumTilt * 180.0 / PxPi, maximumSpeed * 1e3);
	check(maximumOverlap < 20.0e-6, "sheet_stack: sheets overlap by less than 20 um");
	check(maximumFloorPenetration < 20.0e-6, "sheet_stack: bottom sheet sinks less than 20 um into the floor");
	check(settling < 1.0e-6, "sheet_stack: stack stops settling (top moves less than 1 um over the last second)");
	check(maximumDrift < 0.1e-3, "sheet_stack: no sheet drifts more than 0.1 mm sideways");
	check(maximumTilt < 0.1 * PxPi / 180.0, "sheet_stack: no sheet tilts more than 0.1 degree");
	check(maximumSpeed < 1.0e-3, "sheet_stack: stack is at rest (below 1 mm/s)");
	scene->release();
	material->release();
}
}

int main(int argc, char** argv)
{
	const char* selection = argc > 1 ? argv[1] : "all";
	const bool anvil = argc < 3 || std::strcmp(argv[2], "pgs") != 0;
	const PxU32 threads = argc > 3 ? PxU32(std::atoi(argv[3])) : 1;
	PxDefaultAllocator allocator;
	PxDefaultErrorCallback errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(threads);
	Context context;
	context.physics = physics;
	context.dispatcher = dispatcher;
	context.anvil = anvil;
	const bool all = std::strcmp(selection, "all") == 0;
	if(all || std::strcmp(selection, "bouncing_balls") == 0)
	{
		bouncingBalls(context);
	}
	if(all || std::strcmp(selection, "split_conveyors") == 0)
	{
		splitConveyors(context);
	}
	if(all || std::strcmp(selection, "sheet_stack") == 0)
	{
		sheetStack(context);
	}
	dispatcher->release();
	physics->release();
	foundation->release();
	std::printf("Behaviour tests (%s): failures=%d\n", anvil ? "Anvil" : "PGS", failures);
	return failures ? 1 : 0;
}
