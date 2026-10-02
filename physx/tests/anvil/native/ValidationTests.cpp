#include "PxPhysicsAPI.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
using namespace physx;
namespace
{
int failures = 0;
void check(bool value, const char* text)
{
	if(!value)
	{
		std::printf("FAIL %s\n", text);
		++failures;
	}
}
class ValidationErrors : public PxErrorCallback
{
	std::mutex mutex;
	const char* expectedText;
	PxErrorCode::Enum expectedCode;
	int observed;
public:
	int expectedTotal, unexpected;
	ValidationErrors() : expectedText(NULL), expectedCode(PxErrorCode::eNO_ERROR), observed(0), expectedTotal(0), unexpected(0) {}
	void begin(PxErrorCode::Enum code, const char* text)
	{
		std::lock_guard<std::mutex> lock(mutex);
		check(expectedText == NULL, "diagnostic expectations do not overlap");
		expectedCode = code;
		expectedText = text;
		observed = 0;
	}
	void end()
	{
		std::lock_guard<std::mutex> lock(mutex);
		check(expectedText && observed == 1, "exactly one primary diagnostic matches the rejected operation");
		expectedText = NULL;
	}
	virtual void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) PX_OVERRIDE
	{
		std::lock_guard<std::mutex> lock(mutex);
		if(expectedText && code == expectedCode && std::strstr(message, expectedText))
		{
			++observed;
			++expectedTotal;
			return;
		}
		if(code == PxErrorCode::eDEBUG_INFO || code == PxErrorCode::ePERF_WARNING)
		{
			return;
		}
		++unexpected;
		std::printf("UNEXPECTED %d %s (%s:%d)\n", int(code), message, file, line);
	}
};
PxSceneDesc descriptor(PxPhysics& physics, PxDefaultCpuDispatcher& dispatcher)
{
	PxSceneDesc desc(physics.getTolerancesScale());
	desc.cpuDispatcher = &dispatcher;
	desc.filterShader = PxDefaultSimulationFilterShader;
	desc.gravity = PxVec3(0.0f);
	desc.solverType = PxSolverType::eANVIL;
	desc.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
	return desc;
}
PxRigidDynamic* body(PxPhysics& physics, PxMaterial& material, PxReal x)
{
	PxRigidDynamic* result = PxCreateDynamic(physics, PxTransform(PxVec3(x, 10.0f, 0.0f)), PxBoxGeometry(PxVec3(0.25f)), material, 1.0f);
	// Contacts a step ahead of fast bodies: the offset stays the resting precision.
	result->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
	check(result != NULL, "rigid fixture created");
	if(result)
	{
		result->setLinearDamping(0.0f);
		result->setAngularDamping(0.0f);
		result->setSleepThreshold(0.0f);
	}
	return result;
}
void usable(PxScene& scene, PxRigidDynamic& probe, PxU32 count)
{
	check(scene.getSolverType() == PxSolverType::eANVIL, "valid scene retains Anvil selection");
	check(scene.getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC) == count, "rigid actor count is preserved");
	check(probe.getScene() == &scene, "original actor retains its scene");
	const PxReal oldX = probe.getGlobalPose().p.x;
	probe.setLinearVelocity(PxVec3(1.0f, 0.0f, 0.0f));
	scene.simulate(0.01f);
	check(scene.fetchResults(true), "original scene can still complete simulation");
	check(probe.getGlobalPose().isFinite() && probe.getLinearVelocity().isFinite(), "scene remains finite");
	check(std::abs(double(probe.getGlobalPose().p.x - oldX) - .01) < 1e-5, "scene still integrates original actor");
}
// Scene creation validates its descriptor in checked builds only (PX_CHECK_AND_RETURN_NULL); a
// release build trusts it. isValid() checks the Anvil settings in checked builds only too.
void reject(PxPhysics& physics, ValidationErrors& errors, const PxSceneDesc& desc, const char* label)
{
	check(!desc.isValid(), label);
#if PX_CHECKED
	const PxU32 count = physics.getNbScenes();
	errors.begin(PxErrorCode::eINVALID_PARAMETER, "Physics::createScene:");
	PxScene* scene = physics.createScene(desc);
	errors.end();
	check(scene == NULL, label);
	check(physics.getNbScenes() == count, "rejected scene leaves registry unchanged");
	if(scene)
	{
		scene->release();
	}
#else
	PX_UNUSED(physics);
	PX_UNUSED(errors);
#endif
}
void settings(PxPhysics& physics, PxDefaultCpuDispatcher& dispatcher, ValidationErrors& errors)
{
	PxSceneDesc defaults(physics.getTolerancesScale());
	check(PxSolverType::ePGS == 0 && PxSolverType::eTGS == 1 && PxSolverType::eANVIL == 2, "solver enum compatibility");
	check(defaults.solverType == PxSolverType::ePGS, "default solver remains PGS");
	check(defaults.anvilMaxIterations == 1000 && defaults.anvilTolerance == 1e-8f && defaults.anvilRegularization == 1e-4f, "Anvil defaults");
	defaults.cpuDispatcher = &dispatcher; defaults.filterShader = PxDefaultSimulationFilterShader;
	check(defaults.isValid(), "default PGS descriptor valid with required dispatcher and filter");
	PxScene* pgs = physics.createScene(defaults);
	check(pgs && pgs->getSolverType() == PxSolverType::ePGS, "default scene creates PGS");
	if(pgs)
	{
		pgs->release();
	}
	defaults.anvilMaxIterations = 0;
	defaults.anvilTolerance = -1.0f;
	defaults.anvilRegularization = -1.0f;
	check(defaults.isValid(), "unused Anvil fields do not invalidate PGS");
	defaults.solverType = PxSolverType::eTGS;
	check(defaults.isValid(), "unused Anvil fields do not invalidate TGS");
	const PxSceneDesc base = descriptor(physics, dispatcher);
	check(base.isValid(), "default Anvil descriptor valid");
#if PX_CHECKED
	const PxU32 iterations[] = { 0, 0x80000000u, 0xffffffffu };
	for(PxU32 i = 0; i < 3; ++i)
	{
		PxSceneDesc desc = base; desc.anvilMaxIterations = iterations[i];
		reject(physics, errors, desc, "invalid Anvil iteration count rejected");
	}
	const PxReal invalid[] = { 0.0f, -1.0f };
	for(PxU32 i = 0; i < 2; ++i)
	{
		PxSceneDesc desc = base; desc.anvilTolerance = invalid[i];
		reject(physics, errors, desc, "invalid Anvil tolerance rejected");
		desc = base; desc.anvilRegularization = invalid[i];
		reject(physics, errors, desc, "invalid Anvil regularization rejected");
	}
#endif
	PxSceneDesc desc = base; desc.solverType = static_cast<PxSolverType::Enum>(12345);
	reject(physics, errors, desc, "invalid solver enum rejected");
	desc = base; desc.flags |= PxSceneFlag::eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS;
	reject(physics, errors, desc, "TGS-only external forces flag rejected");
}
}int main()
{
	PxDefaultAllocator allocator;
	ValidationErrors errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	if(!foundation)
	{
		return 1;
	}
	foundation->setReportAllocationNames(true);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	if(!physics)
	{
		foundation->release();
		return 1;
	}
	const bool extensions = PxInitExtensions(*physics, NULL);
	check(extensions, "extensions initialized");
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(8);
	PxMaterial* material = physics->createMaterial(.5f, .5f, 0.0f);
	check(dispatcher && material, "dispatcher and material created");
	if(dispatcher && material)
	{
		PxScene* scene = physics->createScene(descriptor(*physics, *dispatcher));
		check(scene && scene->getSolverType() == PxSolverType::eANVIL, "valid Anvil scene created");
		PxRigidDynamic* probe = body(*physics, *material, 0.0f);
		if(scene && probe)
		{
			check(scene->addActor(*probe), "valid rigid actor inserted");
			settings(*physics, *dispatcher, errors); usable(*scene, *probe, 1);
		}
		if(probe)
		{
			probe->release();
		}
		if(scene)
		{
			scene->release();
		}
	}
	if(material)
	{
		material->release();
	}
	if(dispatcher)
	{
		dispatcher->release();
	}
	if(extensions)
	{
		PxCloseExtensions();
	}
	physics->release(); foundation->release();
	std::printf("Native SDK validation: failures=%d expected_diagnostics=%d unexpected_diagnostics=%d\n", failures, errors.expectedTotal, errors.unexpected);
	return failures || errors.unexpected ? 1 : 0;
}
