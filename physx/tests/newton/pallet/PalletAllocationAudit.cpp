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
	mju_threadpool(data, 8);
	size_t allocations = 0, frees = 0, dispatchAllocations = 0;
	for(int step = 0; step < 600; ++step)
	{
		mj_step1(model, data);
		mujocoConveyor::applyRestDistance(data);
		mujocoConveyor::applyConveyorVelocity(data, pallet::conveyorCount, pallet::beltSpeed);
		mj_fwdActuation(model, data);
		mj_fwdAcceleration(model, data);
		newton::MujocoSolverProfile profile;
		if(step >= 200)
			allocationAudit::begin();
		newton::solveMujocoConstraints(model, data, profile);
		allocationAudit::enabled = false;
		if(step >= 200)
		{
			allocations += allocationAudit::allocations.load();
			frees += allocationAudit::frees.load();
			// MuJoCo 3.13 creates one temporary task wrapper per island in
			// mju_dispatch. Native PhysX does not use this comparison dispatcher.
			dispatchAllocations += size_t(data->nisland);
		}
		mj_Euler(model, data);
	}
	mj_deleteData(data);
	mj_deleteModel(model);
	std::printf("PALLET_ALLOCATIONS,steps=400,workers=8,allocations=%zu,frees=%zu,dispatch_allocations=%zu\n",
		allocations, frees, dispatchAllocations);
	return allocations != dispatchAllocations || frees != dispatchAllocations ? 1 : 0;
}
