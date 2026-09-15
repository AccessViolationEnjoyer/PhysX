#include "PxPhysicsAPI.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
using namespace physx;
namespace
{
int failures = 0;
void check(bool value, const char* text)
{
	if(!value) { std::printf("FAIL %s\n", text); ++failures; }
}
class ValidationAllocator : public PxAllocatorCallback
{
	PxDefaultAllocator fallback;
	std::mutex mutex;
	const char* selectedType;
	PxU32 skip, failuresInjected;
public:
	ValidationAllocator() : selectedType(NULL), skip(0), failuresInjected(0) {}
	void arm(const char* type, PxU32 occurrence)
	{
		std::lock_guard<std::mutex> lock(mutex);
		selectedType = type;
		skip = occurrence - 1;
		failuresInjected = 0;
	}
	PxU32 disarm()
	{
		std::lock_guard<std::mutex> lock(mutex);
		selectedType = NULL;
		return failuresInjected;
	}
	virtual void* allocate(size_t size, const char* type, const char* file, int line) PX_OVERRIDE
	{
		{
			std::lock_guard<std::mutex> lock(mutex);
			if(selectedType && type && std::strstr(type, selectedType))
			{
				if(skip)
					--skip;
				else
				{
					selectedType = NULL;
					++failuresInjected;
					return NULL;
				}
			}
		}
		return fallback.allocate(size, type, file, line);
	}
	virtual void deallocate(void* memory) PX_OVERRIDE
	{
		fallback.deallocate(memory);
	}
};

class ValidationErrors : public PxErrorCallback
{
	std::mutex mutex;
	const char* expectedText;
	PxErrorCode::Enum expectedCode;
	int observed;
	bool allocationFailure;
	PxErrorCode::Enum failureCode;
	const char* failureText;
	int expectedArrayFailures, arrayFailures, finalFailures;
public:
	int expectedTotal, unexpected;
	ValidationErrors() : expectedText(NULL), expectedCode(PxErrorCode::eNO_ERROR), observed(0), allocationFailure(false), failureCode(PxErrorCode::eNO_ERROR), failureText(NULL),
		expectedArrayFailures(0), arrayFailures(0), finalFailures(0), expectedTotal(0), unexpected(0) {}
	void begin(PxErrorCode::Enum code, const char* text)
	{
		std::lock_guard<std::mutex> lock(mutex);
		check(expectedText == NULL, "diagnostic expectations do not overlap");
		expectedCode = code; expectedText = text; observed = 0; allocationFailure = false;
	}
	void beginAllocation(PxErrorCode::Enum code, const char* text, bool arrayFailure)
	{
		begin(PxErrorCode::eABORT, "User allocator returned NULL.");
		std::lock_guard<std::mutex> lock(mutex);
		allocationFailure = true;
		failureCode = code;
		failureText = text;
		expectedArrayFailures = arrayFailure ? 1 : 0;
		arrayFailures = finalFailures = 0;
	}
	void end()
	{
		std::lock_guard<std::mutex> lock(mutex);
		check(expectedText && observed == 1, "exactly one primary diagnostic matches the rejected operation");
		if(allocationFailure)
			check(finalFailures == 1 && arrayFailures == expectedArrayFailures, "allocation failure propagation reports the expected diagnostics");
		expectedText = NULL;
		allocationFailure = false;
	}
	virtual void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) PX_OVERRIDE
	{
		std::lock_guard<std::mutex> lock(mutex);
		if(expectedText && code == expectedCode && std::strstr(message, expectedText))
		{
			++observed; ++expectedTotal; return;
		}
		if(allocationFailure && code == PxErrorCode::eOUT_OF_MEMORY &&
			std::strstr(message, "PxArray::allocate: allocator returned null pointer."))
		{
			++arrayFailures;
			++expectedTotal;
			return;
		}
		if(allocationFailure && code == failureCode && std::strstr(message, failureText))
		{
			++finalFailures;
			++expectedTotal;
			return;
		}
		if(code == PxErrorCode::eDEBUG_INFO || code == PxErrorCode::ePERF_WARNING) return;
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
	desc.solverType = PxSolverType::eNEWTON;
	desc.flags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
	return desc;
}
PxRigidDynamic* body(PxPhysics& physics, PxMaterial& material, PxReal x)
{
	PxRigidDynamic* result = PxCreateDynamic(physics, PxTransform(PxVec3(x, 10.0f, 0.0f)), PxBoxGeometry(PxVec3(0.25f)), material, 1.0f);
	check(result != NULL, "rigid fixture created");
	if(result) { result->setLinearDamping(0.0f); result->setAngularDamping(0.0f); result->setSleepThreshold(0.0f); }
	return result;
}
void usable(PxScene& scene, PxRigidDynamic& probe, PxU32 count)
{
	check(scene.getSolverType() == PxSolverType::eNEWTON, "valid scene retains Newton selection");
	check(scene.getNbArticulations() == 0, "no rejected articulation entered scene");
	check(scene.getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC) == count, "rigid actor count is preserved");
	check(probe.getScene() == &scene, "original actor retains its scene");
	const PxReal oldX = probe.getGlobalPose().p.x;
	probe.setLinearVelocity(PxVec3(1.0f, 0.0f, 0.0f));
	scene.simulate(0.01f);
	check(scene.fetchResults(true), "original scene can still complete simulation");
	check(probe.getGlobalPose().isFinite() && probe.getLinearVelocity().isFinite(), "scene remains finite");
	check(std::abs(double(probe.getGlobalPose().p.x - oldX) - .01) < 1e-5, "scene still integrates original actor");
}
void reject(PxPhysics& physics, ValidationErrors& errors, const PxSceneDesc& desc, const char* label)
{
	const PxU32 count = physics.getNbScenes();
	check(!desc.isValid(), label);
	errors.begin(PxErrorCode::eINVALID_PARAMETER, "Physics::createScene:");
	PxScene* scene = physics.createScene(desc);
	errors.end();
	check(scene == NULL, label);
	check(physics.getNbScenes() == count, "rejected scene leaves registry unchanged");
	if(scene) scene->release();
}
void settings(PxPhysics& physics, PxDefaultCpuDispatcher& dispatcher, ValidationErrors& errors)
{
	PxSceneDesc defaults(physics.getTolerancesScale());
	check(PxSolverType::ePGS == 0 && PxSolverType::eTGS == 1 && PxSolverType::eNEWTON == 2, "solver enum compatibility");
	check(defaults.solverType == PxSolverType::ePGS, "default solver remains PGS");
	check(defaults.newtonMaxIterations == 100 && defaults.newtonTolerance == 1e-8f && defaults.newtonRegularization == 1e-4f, "Newton defaults");
	defaults.cpuDispatcher = &dispatcher; defaults.filterShader = PxDefaultSimulationFilterShader;
	check(defaults.isValid(), "default PGS descriptor valid with required dispatcher and filter");
	PxScene* pgs = physics.createScene(defaults);
	check(pgs && pgs->getSolverType() == PxSolverType::ePGS, "default scene creates PGS");
	if(pgs) pgs->release();
	defaults.newtonMaxIterations = 0; defaults.newtonTolerance = -1.0f; defaults.newtonRegularization = std::numeric_limits<PxReal>::quiet_NaN();
	check(defaults.isValid(), "unused Newton fields do not invalidate PGS");
	defaults.solverType = PxSolverType::eTGS;
	check(defaults.isValid(), "unused Newton fields do not invalidate TGS");
	const PxSceneDesc base = descriptor(physics, dispatcher);
	check(base.isValid(), "default Newton descriptor valid");
	const PxU32 iterations[] = { 0, 0x80000000u, 0xffffffffu };
	for(PxU32 i = 0; i < 3; ++i)
	{
		PxSceneDesc desc = base; desc.newtonMaxIterations = iterations[i];
		reject(physics, errors, desc, "invalid Newton iteration count rejected in Release");
	}
	const PxReal invalid[] = { 0.0f, -1.0f, std::numeric_limits<PxReal>::infinity(), -std::numeric_limits<PxReal>::infinity(), std::numeric_limits<PxReal>::quiet_NaN() };
	for(PxU32 i = 0; i < 5; ++i)
	{
		PxSceneDesc desc = base; desc.newtonTolerance = invalid[i];
		reject(physics, errors, desc, "invalid Newton tolerance rejected in Release");
		desc = base; desc.newtonRegularization = invalid[i];
		reject(physics, errors, desc, "invalid Newton regularization rejected in Release");
	}
	PxSceneDesc desc = base; desc.solverType = static_cast<PxSolverType::Enum>(12345);
	reject(physics, errors, desc, "invalid solver enum rejected in Release");
	desc = base; desc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
	reject(physics, errors, desc, "Newton GPU dynamics rejected before GPU initialization");
	desc = base; desc.flags |= PxSceneFlag::eENABLE_DIRECT_GPU_API | PxSceneFlag::eDISABLE_SLEEPING;
	reject(physics, errors, desc, "Newton direct GPU API rejected before GPU initialization");
	desc = base; desc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS | PxSceneFlag::eENABLE_DIRECT_GPU_API | PxSceneFlag::eDISABLE_SLEEPING;
	desc.broadPhaseType = PxBroadPhaseType::eGPU;
	reject(physics, errors, desc, "combined GPU dynamics and direct API rejected");
	desc = base; desc.flags |= PxSceneFlag::eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS;
	reject(physics, errors, desc, "TGS-only external forces flag rejected");
}
void allocationFailures(PxPhysics& physics, PxDefaultCpuDispatcher& dispatcher, PxMaterial& material,
	ValidationAllocator& allocator, ValidationErrors& errors, PxScene& original, PxRigidDynamic& probe)
{
	// The injector targets existing allocator type names; there is no production test hook.
	const PxU32 sceneCount = physics.getNbScenes();
	allocator.arm("NewtonSolver", 1);
	errors.beginAllocation(PxErrorCode::eOUT_OF_MEMORY, "Unable to create Newton scene.", false);
	PxScene* rejected = physics.createScene(descriptor(physics, dispatcher));
	const PxU32 stateFailures = allocator.disarm();
	errors.end();
	check(stateFailures == 1, "Newton solver-state allocation failure was injected");
	check(rejected == NULL && physics.getNbScenes() == sceneCount, "failed Newton state rejects scene creation without registry mutation or PGS fallback");
	if(rejected)
		rejected->release();
	usable(original, probe, 1);

	for(PxU32 failure = 0; failure < 4; ++failure)
	{
		PxScene* scene = physics.createScene(descriptor(physics, dispatcher));
		PxRigidDynamic* actor = body(physics, material, 3.0f);
		check(scene != NULL, "allocation recovery scene created");
		if(scene && actor)
		{
			check(scene->addActor(*actor), "allocation recovery actor inserted");
			actor->setLinearVelocity(PxVec3(1.0f, 0.0f, 0.0f));
			const PxTransform before = actor->getGlobalPose();
			// A new island first allocates the ownership registry, then the available
			// registry, and finally its workspace. Each case uses a fresh scene.
			allocator.arm(failure == 0 ? "NewtonBodySeed" : "NewtonIslandWorkspace", failure == 0 ? 1 : failure);
			errors.beginAllocation(PxErrorCode::eINVALID_OPERATION,
				failure == 0 ? "Newton warm-start storage allocation failed." : "Newton workspace allocation failed.",
				failure != 3);
			scene->simulate(0.01f);
			check(scene->fetchResults(true), "allocation failure still completes the task graph");
			const PxU32 injected = allocator.disarm();
			errors.end();
			check(injected == 1, "selected Newton seed/registry/workspace allocation failed exactly once");
			check(actor->getGlobalPose().p == before.p && actor->getGlobalPose().q == before.q,
				"failed preparation skips pose integration safely");
			check(scene->getSolverType() == PxSolverType::eNEWTON, "allocation failure does not select PGS");
			usable(*scene, *actor, 1);
		}
		if(actor)
			actor->release();
		if(scene)
			scene->release();
	}
	usable(original, probe, 1);
}

class ThresholdEvents : public PxSimulationEventCallback
{
public:
	std::atomic<PxU32> count;
	ThresholdEvents() : count(0) {}
	virtual void onConstraintBreak(PxConstraintInfo*, PxU32) PX_OVERRIDE {}
	virtual void onWake(PxActor**, PxU32) PX_OVERRIDE {}
	virtual void onSleep(PxActor**, PxU32) PX_OVERRIDE {}
	virtual void onTrigger(PxTriggerPair*, PxU32) PX_OVERRIDE {}
	virtual void onAdvance(const PxRigidBody* const*, const PxTransform*, const PxU32) PX_OVERRIDE {}
	virtual void onContact(const PxContactPairHeader&, const PxContactPair* pairs, PxU32 size) PX_OVERRIDE
	{
		for(PxU32 i = 0; i < size; ++i)
			if(pairs[i].events & (PxPairFlag::eNOTIFY_THRESHOLD_FORCE_FOUND |
				PxPairFlag::eNOTIFY_THRESHOLD_FORCE_PERSISTS | PxPairFlag::eNOTIFY_THRESHOLD_FORCE_LOST))
				++count;
	}
};

PxFilterFlags thresholdFilter(PxFilterObjectAttributes, PxFilterData, PxFilterObjectAttributes, PxFilterData,
	PxPairFlags& flags, const void*, PxU32)
{
	flags = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eNOTIFY_TOUCH_FOUND | PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
		PxPairFlag::eNOTIFY_THRESHOLD_FORCE_FOUND | PxPairFlag::eNOTIFY_THRESHOLD_FORCE_PERSISTS |
		PxPairFlag::eNOTIFY_THRESHOLD_FORCE_LOST;
	return PxFilterFlag::eDEFAULT;
}

void thresholdAllocationFailure(PxPhysics& physics, PxDefaultCpuDispatcher& dispatcher, PxMaterial& material,
	ValidationAllocator& allocator, ValidationErrors& errors)
{
	ThresholdEvents events;
	PxSceneDesc desc = descriptor(physics, dispatcher);
	desc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	desc.filterShader = thresholdFilter;
	desc.simulationEventCallback = &events;
	PxScene* scene = physics.createScene(desc);
	check(scene != NULL, "threshold allocation fixture creates its scene");
	if(!scene)
		return;
	PxRigidStatic* floor = PxCreatePlane(physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), material);
	PxRigidDynamic* supported = body(physics, material, 0.0f);
	PxRigidDynamic* added = NULL;
	if(floor && supported)
	{
		supported->setGlobalPose(PxTransform(PxVec3(0.0f, 0.25f, 0.0f)));
		PxRigidBodyExt::setMassAndUpdateInertia(*supported, 1.0f);
		supported->setContactReportThreshold(0.1f);
		check(scene->addActor(*floor) && scene->addActor(*supported), "threshold fixture actors inserted");
		for(PxU32 frame = 0; frame < 3; ++frame)
		{
			events.count = 0;
			scene->simulate(0.01f);
			check(scene->fetchResults(true), "threshold fixture initial solve completes");
		}
		check(events.count.load() > 0, "supported contact produces a force-threshold event before failure");
		added = body(physics, material, 5.0f);
		if(added)
		{
			check(scene->addActor(*added), "new body requires growing Newton node storage");
			events.count = 0;
			allocator.arm("NewtonBodySeed", 1);
			errors.beginAllocation(PxErrorCode::eINVALID_OPERATION, "Newton warm-start storage allocation failed.", true);
			scene->simulate(0.01f);
			check(scene->fetchResults(true), "failed update with existing contacts completes task graph");
			const PxU32 injected = allocator.disarm();
			errors.end();
			check(injected == 1, "seed growth failure injected after a reporting frame");
			check(events.count.load() == 0, "failed Newton update does not replay stale force-threshold events");
			events.count = 0;
			scene->simulate(0.01f);
			check(scene->fetchResults(true), "reporting scene recovers after failed update");
			check(events.count.load() > 0 && supported->getGlobalPose().isFinite(),
				"force reporting and supported simulation resume after recovery");
		}
	}
	else
		check(false, "threshold allocation fixture creates valid actors");
	if(added)
		added->release();
	if(supported)
		supported->release();
	if(floor)
		floor->release();
	scene->release();
}

void outside(PxArticulationReducedCoordinate& articulation, PxArticulationLink& link, PxAggregate* aggregate)
{
	check(articulation.getScene() == NULL && link.getScene() == NULL, "articulation and link remain outside scene");
	check(articulation.getAggregate() == aggregate && link.getAggregate() == aggregate, "articulation aggregate ownership preserved");
}
void articulations(PxPhysics& physics, PxMaterial& material, ValidationErrors& errors, PxScene& scene, PxRigidDynamic& probe)
{
	PxArticulationReducedCoordinate* articulation = physics.createArticulationReducedCoordinate();
	check(articulation != NULL, "articulation fixture created");
	if(!articulation) return;
	PxArticulationLink* link = articulation->createLink(NULL, PxTransform(PxVec3(20.0f, 10.0f, 0.0f)));
	check(link != NULL, "nonempty articulation created");
	if(!link) { articulation->release(); return; }
	check(PxRigidActorExt::createExclusiveShape(*link, PxBoxGeometry(PxVec3(.1f)), material) != NULL, "articulation has valid geometry");
	link->setMass(1.0f); link->setMassSpaceInertiaTensor(PxVec3(1.0f));
	errors.begin(PxErrorCode::eINVALID_OPERATION, "Newton does not support articulations");
	const bool added = scene.addArticulation(*articulation);
	errors.end(); check(!added, "direct articulation insertion rejected"); outside(*articulation, *link, NULL); usable(scene, probe, 1);
	errors.begin(PxErrorCode::eINVALID_PARAMETER, "Individual articulation links");
	const bool addedLink = scene.addActor(*link);
	errors.end(); check(!addedLink, "individual link insertion rejected"); outside(*articulation, *link, NULL); usable(scene, probe, 1);
	PxRigidDynamic* member = body(physics, material, 5.0f);
	if(member)
	{
		for(PxU32 order = 0; order < 2; ++order)
		{
			PxActor* actors[2] = { member, link };
			if(order) { actors[0] = link; actors[1] = member; }
			// This existing generic batch rejection is a warning, not an invalid-operation error.
			errors.begin(PxErrorCode::eDEBUG_WARNING, "Batch addition is not permitted for this actor type");
			const bool batch = scene.addActors(actors, 2);
			errors.end(); check(!batch, "mixed articulation-link batch rejected");
			check(member->getScene() == NULL, "failed batch rolls back rigid prefix and leaves suffix outside");
			outside(*articulation, *link, NULL); usable(scene, probe, 1);
		}
		PxActor* valid[] = { member };
		check(scene.addActors(valid, 1), "valid batch succeeds after rejected batch");
		usable(scene, probe, 2); scene.removeActors(valid, 1);
		PxAggregate* aggregate = physics.createAggregate(4, 4, PxGetAggregateFilterHint(PxAggregateType::eGENERIC, false));
		check(aggregate != NULL, "aggregate created");
		if(aggregate)
		{
			check(aggregate->addActor(*member) && aggregate->addArticulation(*articulation), "outside-scene aggregate accepts valid membership");
			errors.begin(PxErrorCode::eINVALID_OPERATION, "aggregates containing articulations");
			const bool populated = scene.addAggregate(*aggregate);
			errors.end(); check(!populated, "populated articulation aggregate rejected");
			check(aggregate->getScene() == NULL && scene.getNbAggregates() == 0, "rejected aggregate leaves scene registry intact");
			check(aggregate->getNbActors() == 2 && member->getScene() == NULL && member->getAggregate() == aggregate, "rejected aggregate preserves membership without partial insertion");
			outside(*articulation, *link, aggregate); usable(scene, probe, 1);
			check(aggregate->removeArticulation(*articulation), "rejected aggregate remains editable");
			outside(*articulation, *link, NULL);
			check(scene.addAggregate(*aggregate), "same aggregate succeeds after removing articulation");
			check(aggregate->getScene() == &scene && member->getScene() == &scene, "valid aggregate enters scene");
			usable(scene, probe, 2);
			errors.begin(PxErrorCode::eINVALID_OPERATION, "Newton does not support articulations");
			const bool live = aggregate->addArticulation(*articulation);
			errors.end(); check(!live, "in-scene aggregate articulation append rejected");
			check(aggregate->getNbActors() == 1 && aggregate->getScene() == &scene && member->getScene() == &scene, "failed append preserves existing aggregate state");
			outside(*articulation, *link, NULL); usable(scene, probe, 2);
			scene.removeAggregate(*aggregate);
			check(aggregate->getScene() == NULL && member->getScene() == NULL && scene.getNbAggregates() == 0, "aggregate remains safely removable");
			aggregate->release();
		}
		member->release();
	}
	articulation->release(); usable(scene, probe, 1);
}
}
int main()
{
	ValidationAllocator allocator;
	ValidationErrors errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	if(!foundation) return 1;
	foundation->setReportAllocationNames(true);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	if(!physics) { foundation->release(); return 1; }
	const bool extensions = PxInitExtensions(*physics, NULL);
	check(extensions, "extensions initialized");
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(8);
	PxMaterial* material = physics->createMaterial(.5f, .5f, 0.0f);
	check(dispatcher && material, "dispatcher and material created");
	if(dispatcher && material)
	{
		PxScene* scene = physics->createScene(descriptor(*physics, *dispatcher));
		check(scene && scene->getSolverType() == PxSolverType::eNEWTON, "valid Newton scene created");
		PxRigidDynamic* probe = body(*physics, *material, 0.0f);
		if(scene && probe)
		{
			check(scene->addActor(*probe), "valid rigid actor inserted");
			settings(*physics, *dispatcher, errors); usable(*scene, *probe, 1);
			articulations(*physics, *material, errors, *scene, *probe);
			allocationFailures(*physics, *dispatcher, *material, allocator, errors, *scene, *probe);
			thresholdAllocationFailure(*physics, *dispatcher, *material, allocator, errors);
		}
		if(probe) probe->release();
		if(scene) scene->release();
	}
	if(material) material->release();
	if(dispatcher) dispatcher->release();
	if(extensions) PxCloseExtensions();
	physics->release(); foundation->release();
	std::printf("Native SDK validation: failures=%d expected_diagnostics=%d unexpected_diagnostics=%d\n", failures, errors.expectedTotal, errors.unexpected);
	return failures || errors.unexpected ? 1 : 0;
}
