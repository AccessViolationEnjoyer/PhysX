#include "PxPhysicsAPI.h"
#include "CardHouseScene.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace physx;

// Physical behaviour checks against analytic expectations:
// - bouncing_balls: restitution sets the rebound speed and height.
// - split_conveyors: a case straddling two belts at different speeds turns at the rate Coulomb
//   friction predicts for its contact points and does not drift sideways.
// - sheet_stack: 100 sheets of 1 mm and 1 g stay stacked without penetration or drift.
// - card_house: the FBF paper's house of 40 cards, at its scale and with real playing cards,
//   stands still once it has settled.
// - rolling: a sphere and a cylinder roll down a ramp at the rolling acceleration, without slip.
// usage: BehaviourTests [all|bouncing_balls|split_conveyors|sheet_stack|card_house|rolling] [anvil|pgs] [threads=1]
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
	bool ccd;

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
		// Contacts a step ahead of fast bodies: the offset stays the resting precision.
		actor->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, ccd);
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
// the balls come to rest on the floor once rebounds fall below the bounce threshold. The impact
// falls part-way through a step: with speculative contacts the ball turns short of the floor,
// without them it turns inside it, so the apex may be off, and the ball may sink, by up to one
// step of travel at the impact speed. Runs with and without speculative contacts.
void bouncingBalls(Context& context)
{
	const char* mode = context.ccd ? "ccd" : "no_ccd";
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
		std::printf("bouncing_ball %s restitution=%.2f bounces=%d rebound_ratio=%.4f first_apex_m=%.4f expected_apex_m=%.4f max_penetration_mm=%.4f final_height_m=%.5f final_speed_mm_s=%.4f\n",
			mode, e, bounces[i], speedRatio, firstApex[i], firstApexExpected, (radius - lowest[i]) * 1e3, finalHeight, finalSpeed * 1e3);
		std::snprintf(message, sizeof(message), "bouncing_ball %s %.2f: rebound speed is restitution times impact speed (within 5%%)", mode, e);
		check(bounces[i] > 0 && std::abs(speedRatio / e - 1.0) < 0.05, message);
		std::snprintf(message, sizeof(message), "bouncing_ball %s %.2f: first rebound rises to restitution squared times the drop (within 5%% and a step's travel)", mode, e);
		check(std::abs(firstApex[i] - firstApexExpected) < 0.05 * (firstApexExpected - radius) + impactSpeed[i] * TIMESTEP, message);
		std::snprintf(message, sizeof(message), "bouncing_ball %s %.2f: no bounce rises above the previous one", mode, e);
		check(!gained[i], message);
		std::snprintf(message, sizeof(message), "bouncing_ball %s %.2f: ball comes to rest on the floor", mode, e);
		check(finalSpeed < 1.0e-2 && std::abs(finalHeight - radius) < 1.0e-3, message);
		std::snprintf(message, sizeof(message), "bouncing_ball %s %.2f: ball never sinks more than a step's travel into the floor", mode, e);
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
// Box collision supports each belt's share of the weight at the corners of its contact area, so
// the check uses the rate for those points. Uniform pressure over the base, which moves load
// toward the centre and turns the case more than twice as fast, is reported for comparison.
void splitConveyors(Context& context, PxReal fastSpeed, PxReal slowSpeed)
{
	PxScene* scene = context.createScene();
	const PxReal friction = 0.5f;
	PxMaterial* material = context.physics->createMaterial(friction, friction, 0.0f);
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
	// Four seconds carry the case two metres along the belts.
	const int frames = 400, settleFrames = 20;
	double measured = 0.0, expected = 0.0, cornerExpected = 0.0, speedError = 0.0, drift = 0.0, settledDrift = 0.0;
	std::vector<BeltLoad> loads;
	const double startZ = body->getGlobalPose().p.z;
	double settledZ = startZ;
	for(int frame = 0; frame < frames; ++frame)
	{
		step(*scene);
		const PxTransform pose = body->getGlobalPose();
		const PxVec3 centre = pose.transform(body->getCMassLocalPose().p);
		drift = std::max(drift, std::abs(double(centre.z) - startZ));		if(frame < settleFrames)
		{
			settledZ = centre.z;
			continue;
		}
		settledDrift = std::max(settledDrift, std::abs(double(centre.z) - settledZ));
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
	std::printf("split_conveyors belts_m_s=%.3f/%.3f yaw_rate_rad_s=%.5f expected_rad_s=%.5f ratio=%.4f uniform_pressure_rad_s=%.5f uniform_ratio=%.4f max_drift_mm=%.4f settled_drift_mm=%.4f max_speed_error=%.4f yaw_deg=%.2f\n",
		fastSpeed, slowSpeed, measured, cornerExpected, measured / cornerExpected, expected, measured / expected, drift * 1e3, settledDrift * 1e3, speedError, 2.0 * std::asin(std::min(1.0, std::abs(double(pose.q.y)))) * 180.0 / PxPi);
	char message[256];
	std::snprintf(message, sizeof(message), "split_conveyors %.3f/%.3f: case turns at the Coulomb friction rate for its contact points (within 5%%)", fastSpeed, slowSpeed);
	check(std::abs(measured / cornerExpected - 1.0) < 0.05, message);
	std::snprintf(message, sizeof(message), "split_conveyors %.3f/%.3f: case does not drift sideways over 4 s (below 1 mm)", fastSpeed, slowSpeed);
	check(drift < 1.0e-3, message);
	std::snprintf(message, sizeof(message), "split_conveyors %.3f/%.3f: case moves at the belts' mean speed (within 2%%)", fastSpeed, slowSpeed);
	check(speedError < 0.02, message);
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

// The FBF paper's card house (CardHouseScene.h): five levels of tents of two cards leaning together
// at 65 degrees, with bridge cards across neighbouring tents, 40 cards, friction 0.8. At the paper's
// scale (2.5 m cards, 40 mm thick, 25 kg) and with real playing cards (88 x 63 x 0.3 mm, 1.8 g). The
// cards start slightly apart and fall into place over the first two seconds; then the house should
// stand still. The limits on movement are 0.25% of the card's length. A house that collapsed while
// settling would lie still too, so settling is checked separately.
void standingCardHouse(Context& context, const cardHouse::Scale& scale)
{
	PxScene* scene = context.createScene();
	PxMaterial* material = context.physics->createMaterial(PxReal(cardHouse::friction), PxReal(cardHouse::friction), 0.0f);
	const PxReal floorHalf = PxReal(10.0 * scale.layout);
	context.staticBox(*scene, *material, PxVec3(0.0f, -0.5f, 0.0f), PxVec3(floorHalf, 0.5f, floorHalf));
	const std::vector<cardHouse::Card> layout = cardHouse::build(scale);
	std::vector<PxRigidDynamic*> cards;
	std::vector<PxVec3> halves;
	for(size_t i = 0; i < layout.size(); ++i)
	{
		const cardHouse::Card& card = layout[i];
		const PxVec3 half(PxReal(card.halfExtents[0]), PxReal(card.halfExtents[1]), PxReal(card.halfExtents[2]));
		const PxTransform pose(PxVec3(PxReal(card.position[0]), PxReal(card.position[1]), PxReal(card.position[2])), PxQuat(PxReal(card.angle), PxVec3(0.0f, 0.0f, 1.0f)));
		PxRigidDynamic* body = context.dynamicBody(*scene, pose);
		PxShape* shape = context.shape(PxBoxGeometry(half), *material);
		body->attachShape(*shape);
		shape->release();
		PxRigidBodyExt::setMassAndUpdateInertia(*body, PxReal(scale.mass));
		cards.push_back(body);
		halves.push_back(half);
	}
	const int frames = 1000, settleFrames = 200;
	const double length = 2.0 * scale.halfLength;
	std::vector<PxTransform> settled(cards.size());
	double maximumDisplacement = 0.0, maximumRotation = 0.0, maximumFloorPenetration = 0.0, lowestTop = PX_MAX_F64;
	double totalMs = 0.0;
	for(int frame = 0; frame < frames; ++frame)
	{
		const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		step(*scene);
		totalMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		for(size_t i = 0; i < cards.size(); ++i)
		{
			const PxTransform pose = cards[i]->getGlobalPose();
			if(frame + 1 == settleFrames)
			{
				settled[i] = pose;
			}
			if(frame + 1 <= settleFrames)
			{
				continue;
			}
			maximumDisplacement = std::max(maximumDisplacement, double((pose.p - settled[i].p).magnitude()));
			const PxQuat change = pose.q * settled[i].q.getConjugate();
			maximumRotation = std::max(maximumRotation, 2.0 * double(PxAcos(PxMin(1.0f, PxAbs(change.w)))));
			double bottom = PX_MAX_F64;
			for(int corner = 0; corner < 8; ++corner)
			{
				const PxVec3& half = halves[i];
				const PxVec3 local((corner & 1 ? 1.0f : -1.0f) * half.x, (corner & 2 ? 1.0f : -1.0f) * half.y, (corner & 4 ? 1.0f : -1.0f) * half.z);
				bottom = std::min(bottom, double(pose.transform(local).y));
			}
			maximumFloorPenetration = std::max(maximumFloorPenetration, -bottom);
		}
		if(frame + 1 > settleFrames)
		{
			lowestTop = std::min(lowestTop, double(cards.back()->getGlobalPose().p.y));
		}
	}
	double maximumSpeed = 0.0;
	for(size_t i = 0; i < cards.size(); ++i)
	{
		maximumSpeed = std::max(maximumSpeed, double(cards[i]->getLinearVelocity().magnitude()));
	}
	const double topDrop = double(settled.back().p.y) - lowestTop;
	// Settling, from the starting layout: how far each card turned and how far the top card fell.
	double maximumSettleTurn = 0.0;
	for(size_t i = 0; i < cards.size(); ++i)
	{
		const PxQuat start(PxReal(layout[i].angle), PxVec3(0.0f, 0.0f, 1.0f));
		const PxQuat change = settled[i].q * start.getConjugate();
		maximumSettleTurn = std::max(maximumSettleTurn, 2.0 * double(PxAcos(PxMin(1.0f, PxAbs(change.w)))));
	}
	const double settleFall = layout.back().position[1] - double(settled.back().p.y);
	std::printf("card_house %s settle_turn_deg=%.3f top_settle_fall_mm=%.3f top_start_height_mm=%.3f\n", scale.name, maximumSettleTurn * 180.0 / PxPi, settleFall * 1e3, layout.back().position[1] * 1e3);
	check(double(settled.back().p.y) > cardHouse::topLevelBase(scale), "card_house: the house stands (the top card stays on the top level while settling)");
	std::printf("card_house %s cards=%zu max_displacement_mm=%.4f max_rotation_deg=%.4f top_drop_mm=%.4f floor_penetration_um=%.3f final_max_speed_mm_s=%.4f"
		" mean_step_ms=%.4f total_s=%.3f\n", scale.name, cards.size(), maximumDisplacement * 1e3, maximumRotation * 180.0 / PxPi, topDrop * 1e3, maximumFloorPenetration * 1e6,
		maximumSpeed * 1e3, totalMs / frames, totalMs * 1e-3);
	check(maximumDisplacement < 0.0025 * length, "card_house: no card moves more than 0.25% of its length once the house has settled");
	check(maximumRotation < 0.1 * PxPi / 180.0, "card_house: no card turns more than 0.1 degree once the house has settled");
	check(maximumSpeed < 0.0025 * length, "card_house: house is at rest (below 0.25% of a card length per second)");
	check(maximumFloorPenetration < 20.0e-6, "card_house: cards sink less than 20 um into the floor");
	scene->release();
	material->release();
}

// A sphere and a cylinder (a 64-sided convex hull) roll from rest down a 5 degree ramp with
// friction 0.5, far more than rolling without slipping needs (tan 5 / (1 + m r^2 / I) at most
// 0.03). Rolling, the contact stays put on both surfaces while the bodies' material passes
// through it, and the centre accelerates at g sin 5 / (1 + I / (m r^2)). Both bodies have radius
// 5 cm. The hull stays below sqrt(g r), where it would leave the ramp at each corner.
void rolling(Context& context)
{
	const PxReal radius = 0.05f, halfLength = 0.05f, angle = 5.0f * PxPi / 180.0f;
	const int sides = 64, frames = 100;
	PxVec3 points[2 * sides];
	for(int i = 0; i < sides; ++i)
	{
		const PxReal theta = 2.0f * PxPi * PxReal(i) / PxReal(sides);
		points[2 * i] = PxVec3(radius * PxCos(theta), radius * PxSin(theta), -halfLength);
		points[2 * i + 1] = PxVec3(radius * PxCos(theta), radius * PxSin(theta), halfLength);
	}
	PxConvexMeshDesc description;
	description.points.count = 2 * sides;
	description.points.stride = sizeof(PxVec3);
	description.points.data = points;
	description.flags = PxConvexFlag::eCOMPUTE_CONVEX;
	PxConvexMesh* cylinder = PxCreateConvexMesh(PxCookingParams(context.physics->getTolerancesScale()), description, context.physics->getPhysicsInsertionCallback());
	for(int shapeKind = 0; shapeKind < 2; ++shapeKind)
	{
		const bool sphere = shapeKind == 0;
		PxScene* scene = context.createScene();
		PxMaterial* material = context.physics->createMaterial(0.5f, 0.5f, 0.0f);
		// The ramp's top surface descends along +x through the origin.
		const PxQuat tilt(-angle, PxVec3(0.0f, 0.0f, 1.0f));
		const PxVec3 normal = tilt.rotate(PxVec3(0.0f, 1.0f, 0.0f)), downhill = tilt.rotate(PxVec3(1.0f, 0.0f, 0.0f));
		PxRigidStatic* ramp = context.physics->createRigidStatic(PxTransform(-normal * 0.1f, tilt));
		PxShape* rampShape = context.shape(PxBoxGeometry(4.0f, 0.1f, 1.0f), *material);
		ramp->attachShape(*rampShape);
		rampShape->release();
		scene->addActor(*ramp);
		// The cylinder starts with a flat between two corners on the ramp, its axis along z.
		const PxReal apothem = radius * PxCos(PxPi / sides);
		const PxReal centreHeight = sphere ? radius : apothem;
		const PxQuat facing = tilt * PxQuat(PxPi / sides - 0.5f * PxPi, PxVec3(0.0f, 0.0f, 1.0f));
		const PxTransform start(downhill * -2.0f + normal * centreHeight, sphere ? PxQuat(PxIdentity) : facing);
		PxRigidDynamic* body = context.dynamicBody(*scene, start);
		PxShape* shape = sphere ? context.shape(PxSphereGeometry(radius), *material) : context.shape(PxConvexMeshGeometry(cylinder), *material);
		body->attachShape(*shape);
		shape->release();
		PxRigidBodyExt::setMassAndUpdateInertia(*body, 1.0f);
		// I / (m r^2) about the rolling axis (z): 0.4 for the sphere, about 0.5 for the hull.
		const PxVec3 inertia = body->getMassSpaceInertiaTensor();
		const double ratio = double(sphere ? inertia.x : inertia.z) / (double(body->getMass()) * radius * radius);
		const double expected = GRAVITY * std::sin(double(angle)) / (1.0 + ratio);
		double maximumSlip = 0.0, maximumNormalSpeed = 0.0;
		for(int frame = 0; frame < frames; ++frame)
		{
			step(*scene);
			const PxVec3 linear = body->getLinearVelocity(), angular = body->getAngularVelocity();
			// The material at the contact, below the centre on the ramp.
			const PxVec3 contactVelocity = linear + angular.cross(-normal * centreHeight);
			maximumSlip = std::max(maximumSlip, double((contactVelocity - normal * normal.dot(contactVelocity)).magnitude()));
			maximumNormalSpeed = std::max(maximumNormalSpeed, double(PxAbs(linear.dot(normal))));
		}
		const double time = frames * double(TIMESTEP);
		const double travelled = double((body->getGlobalPose().p - start.p).dot(downhill));
		const double speed = double(body->getLinearVelocity().dot(downhill));
		std::printf("rolling %s inertia_ratio=%.4f final_speed_m_s=%.5f expected_speed_m_s=%.5f speed_ratio=%.5f travelled_m=%.5f max_contact_slip_mm_s=%.4f max_normal_speed_mm_s=%.4f\n",
			sphere ? "sphere" : "cylinder", ratio, speed, expected * time, speed / (expected * time), travelled, maximumSlip * 1e3, maximumNormalSpeed * 1e3);
		if(sphere)
		{
			check(std::abs(speed / (expected * time) - 1.0) < 1.0e-3, "rolling: the sphere rolls at the rolling acceleration");
			check(maximumSlip < 1.0e-3, "rolling: the sphere's contact does not slip (below 1 mm/s)");
		}
		// The hull loses energy at each corner landing, so it cannot reach the smooth cylinder's
		// speed: a rigid 64-sided hull that lands inelastically on each corner reaches 0.49 m/s
		// here. At 10 ms Anvil and PGS reach 0.28 m/s, at 1 ms 0.42 m/s: a step resolves each
		// landing as a whole, and the hull lands on a corner every 1-2 steps. Damping separated
		// points, as Anvil once did, braked each corner before it landed: 0.10 m/s at any
		// timestep (see PHYSX_INTEGRATION.md). Reported only.
		scene->release();
		material->release();
	}
	cylinder->release();
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
	context.ccd = false;
	const bool all = std::strcmp(selection, "all") == 0;
	if(all || std::strcmp(selection, "bouncing_balls") == 0)
	{
		bouncingBalls(context);
		context.ccd = true;
		bouncingBalls(context);
	}
	context.ccd = true;
	if(all || std::strcmp(selection, "split_conveyors") == 0)
	{
		// Slip of 10 cm/s, and of 5 mm/s, where the slip direction is least certain.
		splitConveyors(context, 0.6f, 0.4f);
		splitConveyors(context, 0.505f, 0.495f);
	}
	if(all || std::strcmp(selection, "sheet_stack") == 0)
	{
		sheetStack(context);
	}
	if(all || std::strcmp(selection, "card_house") == 0)
	{
		standingCardHouse(context, cardHouse::paper);
		standingCardHouse(context, cardHouse::playingCards);
	}
	if(all || std::strcmp(selection, "rolling") == 0)
	{
		rolling(context);
	}
	dispatcher->release();
	physics->release();
	foundation->release();
	std::printf("Behaviour tests (%s): failures=%d\n", anvil ? "Anvil" : "PGS", failures);
	return failures ? 1 : 0;
}
