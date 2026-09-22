#ifndef NEWTON_MUJOCO_ADAPTER_H
#define NEWTON_MUJOCO_ADAPTER_H

#include "NewtonSolver.h"
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>

// MuJoCo 3.13 exposes thread-pool creation publicly, while its synchronous
// task dispatcher remains an internal exported entry point.
typedef void (*NewtonMujocoTaskFunction)(const mjModel*, mjData*, void*, int, int);
extern "C" MJAPI void mju_dispatch(const mjModel*, mjData*, NewtonMujocoTaskFunction, void*, int);

// Shared benchmark adapter for independent centered free bodies with diagonal
// inertia, pyramidal contact rows, and Euler integration.
namespace newton
{
static double mujocoMassDiagonal(const mjModel* model, const mjData* data, int dof)
{
	return data->M[model->M_rowadr[dof] + model->M_rownnz[dof] - 1];
}

struct MujocoSolverProfile
{
	// Preparation and solve overlap across islands, so isolated wall times do not exist.
	double preparationWallMs = -1.0;
	double solverOnlyMs = -1.0;
	double globalPreparationWallMs = 0.0;
	double islandTasksWallMs = -1.0;
	double scatterWallMs = 0.0;
	bool enabled = false;
	double preparationMs = 0.0;
	double solveMs = 0.0;
	double matrixMs = 0.0;
	double factorMs = 0.0;
	double symbolicMs = 0.0;
	double updateMs = 0.0;
	double evaluationMs = 0.0;
	double lineSearchMs = 0.0;
	double backsolveMs = 0.0;
	int updates = 0;
	int symbolicAnalyses = 0;
};

struct IslandTask
{
	newton::Problem problem;
	newton::VectorStorage rootMass;
	newton::Result previous;
	const mjModel* model;
	mjData* data;
	int island;
	newton::Settings settings;
	newton::Result result;
	double preparationMs = 0.0;
	std::exception_ptr error;
};

static void* prepareIsland(void* argument)
{
	IslandTask& task = *static_cast<IslandTask*>(argument);
	try
	{
		const newton::Clock::time_point prepareStart = newton::Clock::now();
		const mjModel* model = task.model;
		mjData* data = task.data;
		const int island = task.island;
		const int dofStart = data->island_idofadr[island];
		const int dofCount = data->island_nv[island];
		const int rowStart = data->island_iefcadr[island];
		const int rowCount = data->island_nefc[island];
		// Each island retains its preparation buffers while its tasks move between
		// workers. Every coefficient and warm start is rebuilt for this step.
		newton::Problem& problem = task.problem;
		problem.timestep = model->opt.timestep;
		problem.inverseMass.resize(dofCount / 6);
		problem.massDiagonal.resize(dofCount);
		problem.freeBodyVelocity = newton::Vector::Zero(dofCount);
		newton::VectorStorage& rootMass = task.rootMass;
		rootMass.resize(dofCount);
		newton::Result& previous = task.previous;
		previous.primal.resize(dofCount);
		for(int i = 0; i < dofCount; ++i)
		{
			const int dof = data->map_idof2dof[dofStart + i];
			const double mass = mujocoMassDiagonal(model, data, dof);
			problem.massDiagonal[i] = mass;
			rootMass[i] = std::sqrt(mass);
			previous.primal[i] = problem.timestep * rootMass[i] * (data->qacc_warmstart[dof] - data->qacc_smooth[dof]);
			if(i % 6 == 0)
				problem.inverseMass[i / 6] = 1.0 / mass;
		}
		problem.columnCursors.assign(dofCount, 0);
		newton::reserveStorage(problem.contacts, size_t(rowCount));
		problem.contacts.resize(rowCount);
		for(int i = 0; i < rowCount; ++i)
		{
			const int row = data->map_iefc2efc[rowStart + i];
			newton::CompactContact& contact = problem.contacts[i];
			contact.body[0] = contact.body[1] = -1;
			contact.block = 0;
			contact.freeVelocity = problem.timestep * data->efc_b[row];
			contact.regularization = data->efc_R[row];
			const int start = data->efc_J_rowadr[row];
			const int end = start + data->efc_J_rownnz[row];
			// MuJoCo includes all six columns of each free body in a contact row.
			// Map each endpoint once rather than looking it up for every column.
			for(int j = start, side = 0; j < end; j += 6, ++side)
			{
				const int local = data->map_dof2idof[data->efc_J_colind[j]] - dofStart;
				contact.body[side] = local / 6;
				for(int axis = 0; axis < 6; ++axis)
				{
					const double value = data->efc_J[j + axis] / rootMass[local + axis];
					contact.jacobian[side][axis] = value;
					problem.columnCursors[local + axis] += value != 0.0;
				}
			}
			if(contact.body[1] < 0)
				contact.jacobian[1].setZero();
		}
		newton::prepareProblemFromColumnCounts(problem);
		task.preparationMs = newton::elapsed(prepareStart);

	}
	catch(...)
	{
		task.error = std::current_exception();
	}
	return NULL;
}

static void* solvePreparedIsland(void* argument)
{
	IslandTask& task = *static_cast<IslandTask*>(argument);
	try
	{
		newton::solveNewton(task.problem, task.settings, task.result, &task.previous);
	}
	catch(...)
	{
		task.error = std::current_exception();
	}
	return NULL;
}

static void* prepareAndSolveIsland(void* argument)
{
	prepareIsland(argument);
	IslandTask& task = *static_cast<IslandTask*>(argument);
	if(!task.error)
		solvePreparedIsland(argument);
	return NULL;
}

static void scatterIsland(IslandTask& task)
{
	mjData* data = task.data;
	const int dofStart = data->island_idofadr[task.island];
	const int rowStart = data->island_iefcadr[task.island];
	for(int i = 0; i < data->island_nv[task.island]; ++i)
	{
		const int dof = data->map_idof2dof[dofStart + i];
		data->qacc[dof] = data->qacc_smooth[dof] + task.result.primal[i] / (task.problem.timestep * task.rootMass[i]);
	}
	for(int i = 0; i < data->island_nefc[task.island]; ++i)
		data->efc_force[data->map_iefc2efc[rowStart + i]] = task.result.impulse[i] / task.problem.timestep;
}

static void dispatchIsland(const mjModel*, mjData*, void* argument, int, int task)
{
	std::vector<IslandTask>& islands = *static_cast<std::vector<IslandTask>*>(argument);
	prepareAndSolveIsland(&islands[task]);
}

static int solveMujocoConstraints(const mjModel* model, mjData* data, MujocoSolverProfile& profile)
{
	const newton::Clock::time_point globalPreparationStart = newton::Clock::now();
	mj_mulJacVec(model, data, data->efc_b, data->qacc_smooth);
	mju_subFrom(data->efc_b, data->efc_aref, data->nefc);
	mju_copy(data->qacc, data->qacc_smooth, int(model->nv));
	static thread_local std::vector<IslandTask> islands;
	islands.resize(std::max(islands.size(), size_t(data->nisland)));
	for(int i = 0; i < data->nisland; ++i)
	{
		islands[i].error = NULL;
		islands[i].model = model;
		islands[i].data = data;
		islands[i].island = i;
		islands[i].settings.iterations = model->opt.iterations;
		islands[i].settings.tolerance = model->opt.tolerance;
		islands[i].settings.profile = profile.enabled;
	}
	profile.globalPreparationWallMs = newton::elapsed(globalPreparationStart);
	const newton::Clock::time_point islandTasksStart = newton::Clock::now();
	mju_dispatch(model, data, dispatchIsland, &islands, data->nisland);
	// Each island is fully prepared before its solve. Only the final join is global.
	profile.islandTasksWallMs = newton::elapsed(islandTasksStart);
	std::exception_ptr error;
	int iterations = 0;
	for(int i = 0; i < data->nisland; ++i)
	{
		if(islands[i].error)
			error = islands[i].error;
		const newton::Result& result = islands[i].result;
		iterations = std::max(iterations, result.iterations);
		profile.preparationMs += islands[i].preparationMs;
		profile.solveMs += result.elapsedMs;
		profile.matrixMs += result.matrixMs;
		profile.factorMs += result.factorMs;
		profile.symbolicMs += result.symbolicMs;
		profile.updateMs += result.updateMs;
		profile.evaluationMs += result.evaluationMs;
		profile.lineSearchMs += result.lineSearchMs;
		profile.backsolveMs += result.backsolveMs;
		profile.updates += result.rankUpdates;
		profile.symbolicAnalyses += result.symbolicAnalyses;
	}
	if(error)
		std::rethrow_exception(error);
	const newton::Clock::time_point scatterStart = newton::Clock::now();
	for(int i = 0; i < data->nisland; ++i)
		scatterIsland(islands[i]);
	mj_mulJacTVec(model, data, data->qfrc_constraint, data->efc_force);
	profile.scatterWallMs = newton::elapsed(scatterStart);
	return iterations;
}

// This adapter supports separate, centered free boxes. Their angular DOFs are
// expressed in each body's frame, so rotation does not make M non-diagonal.
static void validateMujocoModel(const mjModel* model, mjData* data, size_t bodyCount)
{
	if(model->nv != int(bodyCount * 6) || model->njnt != int(bodyCount) || data->ne || data->nf ||
		model->opt.cone != mjCONE_PYRAMIDAL || model->opt.integrator != mjINT_EULER)
		throw std::runtime_error("The MuJoCo adapter requires free boxes, pyramidal contacts and Euler integration");
	for(int i = 0; i < model->nv; ++i)
	{
		const int diagonal = model->M_rowadr[i] + model->M_rownnz[i] - 1;
		for(int address = model->M_rowadr[i]; address < diagonal; ++address)
		{
			if(std::abs(data->M[address]) > 1.0e-12 * data->M[diagonal])
				throw std::runtime_error("The MuJoCo adapter requires diagonal free-body inertia");
		}
	}
}

}
#endif
