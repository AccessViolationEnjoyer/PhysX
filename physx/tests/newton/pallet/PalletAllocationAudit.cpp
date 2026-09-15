// Exercise the exact adapter and task dispatch used by the simulation runner.
// Renaming its entry point keeps instrumentation out of the normal executable.
#define main palletRunnerMain
#include "MujocoPalletConveyor.cpp"
#undef main
#include "../AllocationCounter.h"

int main(int argc, char** argv)
{
	if(argc != 2 || !allocationAudit::install())
		return 1;
	char error[1024];
	mjModel* model = mj_loadXML(argv[1], NULL, error, sizeof(error));
	if(!model)
		return 1;
	mjData* data = mj_makeData(model);
	mjThreadPool* pool = mju_threadPoolCreate(8);
	mju_bindThreadPool(data, pool);
	size_t allocations = 0, frees = 0;
	for(int step = 0; step < 600; ++step)
	{
		mj_step1(model, data);
		applyRestDistance(data);
		applyConveyorVelocity(data);
		mj_fwdActuation(model, data);
		mj_fwdAcceleration(model, data);
		newton::MujocoSolverProfile profile;
		if(step >= 200)
			allocationAudit::begin();
		newton::solveMujocoConstraints(model, data, pool, profile);
		allocationAudit::enabled = false;
		if(step >= 200)
		{
			allocations += allocationAudit::allocations.load();
			frees += allocationAudit::frees.load();
		}
		mj_Euler(model, data);
	}
	mju_threadPoolDestroy(pool);
	mj_deleteData(data);
	mj_deleteModel(model);
	std::printf("PALLET_ALLOCATIONS,steps=400,workers=8,allocations=%zu,frees=%zu\n", allocations, frees);
	return allocations || frees ? 1 : 0;
}
