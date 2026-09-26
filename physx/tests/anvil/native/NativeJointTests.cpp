#include "PxPhysicsAPI.h"
#include <cmath>
#include <cstdio>
#include <cstring>

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

struct SceneFixture
{
	PxPhysics& physics;
	PxScene* scene;

	SceneFixture(PxPhysics& sdk, PxCpuDispatcher& dispatcher, PxSolverType::Enum solver) : physics(sdk)
	{
		PxSceneDesc settings(physics.getTolerancesScale());
		settings.gravity = PxVec3(0.0f);
		settings.cpuDispatcher = &dispatcher;
		settings.filterShader = PxDefaultSimulationFilterShader;
		settings.solverType = solver;
		scene = physics.createScene(settings);
	}

	~SceneFixture() { scene->release(); }

	PxRigidDynamic* body(const PxTransform& pose = PxTransform(PxIdentity))
	{
		PxRigidDynamic* result = physics.createRigidDynamic(pose);
		result->setMass(1.0f);
		result->setMassSpaceInertiaTensor(PxVec3(1.0f));
		result->setLinearDamping(0.0f);
		result->setAngularDamping(0.0f);
		result->setSleepThreshold(0.0f);
		result->setSolverIterationCounts(16, 2);
		scene->addActor(*result);
		return result;
	}

	void step(PxU32 count = 1)
	{
		for(PxU32 i = 0; i < count; ++i)
		{
			scene->simulate(0.01f);
			scene->fetchResults(true);
		}
	}
};

static void testNativeJoints(PxPhysics& physics, PxCpuDispatcher& dispatcher, PxSolverType::Enum solver)
{
	{
		SceneFixture fixture(physics, dispatcher, solver);
		PxRigidDynamic* body = fixture.body();
		PxPrismaticJoint* joint = PxPrismaticJointCreate(physics, NULL, PxTransform(PxIdentity), body, PxTransform(PxIdentity));
		joint->setLimit(PxJointLinearLimitPair(physics.getTolerancesScale(), -0.25f, 0.25f));
		joint->setPrismaticJointFlag(PxPrismaticJointFlag::eLIMIT_ENABLED, true);
		body->setLinearVelocity(PxVec3(1.0f, 2.0f, 3.0f));
		fixture.step(100);
		const PxVec3 position = body->getGlobalPose().p;
		check(position.x > 0.1f && position.x < 0.252f && std::abs(position.y) < 1.0e-4f &&
			std::abs(position.z) < 1.0e-4f, "native prismatic freedom and hard limit");
	}
	{
		SceneFixture fixture(physics, dispatcher, solver);
		PxRigidDynamic* body = fixture.body();
		PxRevoluteJoint* joint = PxRevoluteJointCreate(physics, NULL, PxTransform(PxIdentity), body, PxTransform(PxIdentity));
		joint->setLimit(PxJointAngularLimitPair(-0.3f, 0.3f));
		joint->setRevoluteJointFlag(PxRevoluteJointFlag::eLIMIT_ENABLED, true);
		body->setAngularVelocity(PxVec3(1.0f, 2.0f, 3.0f));
		fixture.step(100);
		check(joint->getAngle() > 0.1f && joint->getAngle() < 0.302f && body->getGlobalPose().p.magnitude() < 1.0e-4f,
			"native revolute freedom and hard limit");
	}
	{
		SceneFixture fixture(physics, dispatcher, solver);
		PxRigidDynamic* body = fixture.body();
		PxSphericalJointCreate(physics, NULL, PxTransform(PxIdentity), body, PxTransform(PxIdentity));
		body->setLinearVelocity(PxVec3(1.0f, 2.0f, 3.0f));
		body->setAngularVelocity(PxVec3(0.1f, 0.2f, 0.3f));
		fixture.step(100);
		check(body->getGlobalPose().p.magnitude() < 1.0e-4f && body->getGlobalPose().q.getAngle() > 0.2f,
			"native spherical joint locks translation and allows orientation changes");
	}
	{
		SceneFixture fixture(physics, dispatcher, solver);
		PxRigidDynamic* body = fixture.body(PxTransform(PxVec3(1.2f, 0.0f, 0.0f)));
		PxDistanceJoint* joint = PxDistanceJointCreate(physics, NULL, PxTransform(PxIdentity), body, PxTransform(PxIdentity));
		joint->setMinDistance(1.0f);
		joint->setMaxDistance(1.0f);
		joint->setTolerance(0.0f);
		joint->setStiffness(200.0f);
		joint->setDamping(30.0f);
		joint->setDistanceJointFlags(PxDistanceJointFlag::eMIN_DISTANCE_ENABLED | PxDistanceJointFlag::eMAX_DISTANCE_ENABLED |
			PxDistanceJointFlag::eSPRING_ENABLED);
		fixture.step(200);
		check(std::abs(body->getGlobalPose().p.x - 1.0f) < 1.0e-3f && body->getLinearVelocity().magnitude() < 1.0e-3f,
			"native distance spring converges to rest length");
	}
}

static void testNativeCouplers(PxPhysics& physics, PxCpuDispatcher& dispatcher, PxSolverType::Enum solver)
{
	{
		SceneFixture fixture(physics, dispatcher, solver);
		PxRigidDynamic* body0 = fixture.body();
		PxRigidDynamic* body1 = fixture.body();
		PxRevoluteJoint* hinge0 = PxRevoluteJointCreate(physics, NULL, PxTransform(PxIdentity), body0, PxTransform(PxIdentity));
		PxRevoluteJoint* hinge1 = PxRevoluteJointCreate(physics, NULL, PxTransform(PxIdentity), body1, PxTransform(PxIdentity));
		PxGearJoint* gear = PxGearJointCreate(physics, body0, PxTransform(PxIdentity), body1, PxTransform(PxIdentity));
		gear->setHinges(hinge0, hinge1);
		gear->setGearRatio(2.0f);
		body0->setAngularVelocity(PxVec3(1.0f, 0.0f, 0.0f));
		fixture.step(50);
		check(body0->getAngularVelocity().x > 0.1f && std::abs(2.0f * body0->getAngularVelocity().x +
			body1->getAngularVelocity().x) < 1.0e-4f, "native gear ratio couples two rotating bodies");
	}
	{
		SceneFixture fixture(physics, dispatcher, solver);
		PxRigidDynamic* body0 = fixture.body();
		PxRigidDynamic* body1 = fixture.body();
		PxRevoluteJoint* hinge = PxRevoluteJointCreate(physics, NULL, PxTransform(PxIdentity), body0, PxTransform(PxIdentity));
		PxPrismaticJoint* slider = PxPrismaticJointCreate(physics, NULL, PxTransform(PxIdentity), body1, PxTransform(PxIdentity));
		PxRackAndPinionJoint* rack = PxRackAndPinionJointCreate(physics, body0, PxTransform(PxIdentity), body1, PxTransform(PxIdentity));
		rack->setJoints(hinge, slider);
		rack->setRatio(2.0f);
		body0->setAngularVelocity(PxVec3(1.0f, 0.0f, 0.0f));
		fixture.step(50);
		check(body0->getAngularVelocity().x > 0.1f && std::abs(body0->getAngularVelocity().x -
			2.0f * body1->getLinearVelocity().x) < 1.0e-4f, "native rack and pinion couples angular and linear motion");
	}
}

struct CustomSpringData
{
	PxReal stiffness;
	PxReal damping;
	PxReal angle;
	bool angular;
};

static PxU32 prepareCustomSpring(Px1DConstraint* rows, PxVec3p& offset, PxU32,
	PxConstraintInvMassScale&, const void* constantBlock, const PxTransform& frame0, const PxTransform&,
	bool, PxVec3p& anchor0, PxVec3p& anchor1)
{
	const CustomSpringData& data = *static_cast<const CustomSpringData*>(constantBlock);
	offset = PxVec3(0.0f);
	anchor0 = anchor1 = frame0.p;
	Px1DConstraint& row = rows[0];
	if(data.angular)
		row.angular0 = PxVec3(1.0f, 0.0f, 0.0f);
	else
		row.linear0 = PxVec3(1.0f, 0.0f, 0.0f);
	row.geometricError = data.angular ? data.angle : frame0.p.x;
	row.flags = Px1DConstraintFlag::eSPRING | Px1DConstraintFlag::eOUTPUT_FORCE;
	row.mods.spring.stiffness = data.stiffness;
	row.mods.spring.damping = data.damping;
	return 1;
}

class CustomSpring : public PxConstraintConnector
{
public:
	CustomSpringData data;
	PxConstraint* constraint;

	CustomSpring(PxPhysics& physics, PxRigidDynamic& body, bool angular) : constraint(NULL)
	{
		data.stiffness = angular ? 1.0f : 1000.0f;
		data.damping = 2.0f * std::sqrt(data.stiffness);
		data.angle = 0.0f;
		data.angular = angular;
		const PxConstraintShaderTable shaders = { prepareCustomSpring, NULL, PxConstraintFlag::Enum(0) };
		constraint = physics.createConstraint(&body, NULL, *this, shaders, sizeof(data));
	}

	~CustomSpring() { if(constraint) constraint->release(); }
	virtual void* prepareData() PX_OVERRIDE { return &data; }
	virtual bool updatePvdProperties(pvdsdk::PvdDataStream&, const PxConstraint*, PxPvdUpdateType::Enum) const PX_OVERRIDE { return false; }
	virtual void updateOmniPvdProperties() const PX_OVERRIDE {}
	virtual void onConstraintRelease() PX_OVERRIDE { constraint = NULL; }
	virtual void onComShift(PxU32) PX_OVERRIDE {}
	virtual void onOriginShift(const PxVec3&) PX_OVERRIDE {}
	virtual void* getExternalReference(PxU32& type) PX_OVERRIDE { type = 0x10000; return this; }
	virtual PxBase* getSerializable() PX_OVERRIDE { return NULL; }
	virtual PxConstraintSolverPrep getPrep() const PX_OVERRIDE { return prepareCustomSpring; }
	virtual const void* getConstantBlock() const PX_OVERRIDE { return &data; }
};

static void testCustomSprings(PxPhysics& physics, PxCpuDispatcher& dispatcher, PxSolverType::Enum solver)
{
	{
		SceneFixture fixture(physics, dispatcher, solver);
		PxRigidDynamic* body = fixture.body(PxTransform(PxVec3(0.2f, 0.0f, 0.0f)));
		PxPrismaticJoint* joint = PxPrismaticJointCreate(physics, NULL, PxTransform(PxIdentity), body, PxTransform(PxIdentity));
		joint->setLimit(PxJointLinearLimitPair(physics.getTolerancesScale(), -0.25f, 0.25f));
		joint->setPrismaticJointFlag(PxPrismaticJointFlag::eLIMIT_ENABLED, true);
		CustomSpring spring(physics, *body, false);
		fixture.step(100);
		check(std::abs(body->getGlobalPose().p.x) < 1.0e-4f, "custom spring coexists with native prismatic limits");
		joint->release();
		body->setGlobalPose(PxTransform(PxVec3(0.2f, 0.0f, 0.0f)));
		body->setLinearVelocity(PxVec3(0.0f));
		fixture.step(100);
		check(std::abs(body->getGlobalPose().p.x) < 1.0e-4f, "custom constraint remains valid after independent native joint release");
	}
	{
		SceneFixture fixture(physics, dispatcher, solver);
		const double initialAngle = 10000.0 * PxPi / 180.0;
		PxRigidDynamic* body = fixture.body(PxTransform(PxVec3(0.0f), PxQuat(PxReal(initialAngle), PxVec3(1.0f, 0.0f, 0.0f))));
		PxRevoluteJoint* joint = PxRevoluteJointCreate(physics, NULL, PxTransform(PxIdentity), body, PxTransform(PxIdentity));
		CustomSpring spring(physics, *body, true);
		double angle = initialAngle;
		PxReal previousAngle = joint->getAngle();
		for(PxU32 i = 0; i < 1600; ++i)
		{
			spring.data.angle = PxReal(angle);
			spring.constraint->markDirty();
			fixture.step();
			const PxReal nextAngle = joint->getAngle();
			angle += std::remainder(double(nextAngle) - previousAngle, double(PxTwoPi));
			previousAngle = nextAngle;
		}
		check(std::abs(angle) < 0.001 && body->getAngularVelocity().magnitude() < 0.001f,
			"custom angular spring unwinds the full 10000 degree accumulated angle");

		joint->setDriveVelocity(0.5f);
		joint->setDriveForceLimit(10.0f);
		joint->setConstraintFlag(PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);
		joint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true);
		for(PxU32 i = 0; i < 20; ++i)
		{
			spring.data.angle = PxReal(angle);
			spring.constraint->markDirty();
			fixture.step();
			const PxReal nextAngle = joint->getAngle();
			angle += std::remainder(double(nextAngle) - previousAngle, double(PxTwoPi));
			previousAngle = nextAngle;
		}
		check(std::abs(body->getAngularVelocity().x - 0.5f) < 1.0e-3f,
			"native motor remains available alongside the independent spring");
	}
}

int main(int argc, char** argv)
{
	const PxSolverType::Enum solver = argc > 1 && std::strcmp(argv[1], "pgs") == 0 ? PxSolverType::ePGS : PxSolverType::eANVIL;
	PxDefaultAllocator allocator;
	PxDefaultErrorCallback errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	PxInitExtensions(*physics, NULL);
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(1);
	testNativeJoints(*physics, *dispatcher, solver);
	testNativeCouplers(*physics, *dispatcher, solver);
	testCustomSprings(*physics, *dispatcher, solver);
	dispatcher->release();
	PxCloseExtensions();
	physics->release();
	foundation->release();
	printf("Native joint scenes: %d failures\n", failures);
	return failures ? 1 : 0;
}
