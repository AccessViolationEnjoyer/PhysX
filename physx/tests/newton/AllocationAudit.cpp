#include "NewtonSolver.h"
#include "AllocationCounter.h"
#include <cstdio>
#include <cmath>

static void prepare(newton::Problem& problem, int bodies, int contacts, int step)
{
	problem.timestep = 0.01;
	problem.inverseMass.assign(bodies, 1.0);
	problem.massDiagonal.resize(6 * bodies);
	problem.massDiagonal.setOnes();
	problem.clearContacts();
	for(int i = 0; i < contacts; ++i)
	{
		newton::Contact contact;
		contact.body[0] = i % bodies;
		contact.body[1] = i < bodies ? -1 : (i + 1 + step % 3) % bodies;
		contact.bilateral = i % 3 == 0;
		contact.friction = i % 3 == 1 ? 0.0 : 0.5;
		contact.regularization.setConstant(0.05);
		contact.freeVelocity = newton::Vec3(std::sin(double(i)), std::cos(double(i)), -0.2 - 0.001 * step);
		for(int end = 0; end < 2; ++end)
			for(int row = 0; row < 3; ++row)
				for(int column = 0; column < 6; ++column)
					contact.jacobian[end](row, column) = contact.body[end] < 0 ? 0.0 :
						std::sin(0.37 * (1 + i * 13 + end * 19 + row * 7 + column));
		problem.addContact(contact);
	}
	newton::prepareProblem(problem);
}

static void prepareExtended(newton::Problem& problem, int step)
{
	const int bodies = step % 3 == 0 ? 12 : (step % 3 == 1 ? 8 : 4);
	const int groups = 5 + step % 4;
	const int normals = 1 + step % 4;
	const int tangents = step % 2 ? 2 : 4;
	const bool grouped = step % 2 != 0;
	const double infinity = std::numeric_limits<double>::infinity();
	problem.timestep = 0.01;
	problem.inverseMass.assign(bodies, 1.0);
	problem.massDiagonal.setZero(6 * bodies);
	problem.massDiagonal.setOnes();
	problem.clearContacts();
	for(int group = 0; group < groups; ++group)
	{
		const int first = int(problem.contacts.size());
		for(int row = 0; row < normals + tangents; ++row)
		{
			newton::CompactContact contact;
			contact.body[0] = group % bodies;
			contact.body[1] = (contact.body[0] + 1 + step % 3) % bodies;
			contact.regularization = 0.04 + 0.01 * (row % 3);
			contact.freeVelocity = row < normals ? -0.2 + 0.3 * std::sin(0.2 * (step + row)) : std::cos(0.37 * (step - row));
			for(int end = 0; end < 2; ++end)
				for(int axis = 0; axis < 6; ++axis)
					contact.jacobian[end][axis] = 0.3 * std::sin(0.41 * (1 + group * 11 + row * 3 + end * 17 + axis));
			problem.addScalarContact(contact, row < normals ? 0.0 : -infinity,
				row < normals ? 0.2 + 0.03 * row : infinity);
		}
		if(grouped)
			problem.addPatch(first, normals, tangents, 0.6);
	}
	newton::prepareProblem(problem);
}

static bool auditExtended()
{
	newton::Problem problem;
	newton::Workspace workspace;
	newton::Result result;
	newton::Settings settings;
	settings.tolerance = 1.0e-12;
	size_t allocations = 0, frees = 0;
	int updates = 0;
	for(int step = 0; step < 1060; ++step)
	{
		// Repeating the same 60-state cycle first exercises every capacity,
		// changing body graphs, group counts and scalar/patch transitions.
		allocationAudit::allocations = 0;
		allocationAudit::frees = 0;
		allocationAudit::enabled = step >= 60;
		prepareExtended(problem, step % 60);
		newton::SolveStatus::Enum status = newton::solveNewton(problem, settings, result, workspace);
		updates += result.rankUpdates;
		if(step % 2 == 0)
		{
			const int normals = 1 + step % 4, tangents = 4;
			for(int group = 0; group < 5 + step % 4; ++group)
				for(int row = 0; row < tangents; ++row)
					problem.setScalarBounds(group * (normals + tangents) + normals + row, -0.015, 0.015);
		}
		else
			for(size_t patch = 0; patch < problem.patches.size(); ++patch)
				problem.patches[patch].friction = 0.4;
		status = status == newton::SolveStatus::eSUCCESS ? newton::continueNewton(problem, settings, result, workspace, &result) : status;
		updates += result.rankUpdates;
		allocationAudit::enabled = false;
		allocations += allocationAudit::allocations;
		frees += allocationAudit::frees;
		if(status != newton::SolveStatus::eSUCCESS || result.gradientResidual > 1.0e-7)
		{
			std::printf("EXTENDED_FAILURE,step=%d,status=%d,residual=%.12g\n", step, int(status), result.gradientResidual);
			return false;
		}
	}
	std::printf("EXTENDED_CONTINUATION_ALLOCATIONS,steps=1000,allocations=%zu,frees=%zu,rank_updates=%d\n", allocations, frees, updates);
	return allocations == 0 && frees == 0;
}

int main()
{
	if(!allocationAudit::install())
		return 1;
	newton::Problem problem;
	newton::Result result;
	newton::Settings settings;
	settings.tolerance = 1.0e-12;
	// Exercise every topology and size before measuring retained capacity.
	for(int step = 0; step < 36; ++step)
	{
		prepare(problem, step % 2 ? 12 : 6, step % 4 ? 72 : 60, step);
		newton::solveNewton(problem, settings, result);
		newton::solveNewton(problem, settings, result, &result);
	}
	size_t preparation = 0, solves = 0, releases = 0;
	int analyses = 0, updates = 0;
	for(int step = 0; step < 1000; ++step)
	{
		const int bodies = step % 4 < 2 ? 12 : 6;
		const int contacts = step % 2 ? 72 : 60;
		allocationAudit::allocations = 0;
		allocationAudit::frees = 0;
		allocationAudit::enabled = true;
		prepare(problem, bodies, contacts, step);
		allocationAudit::enabled = false;
		preparation += allocationAudit::allocations;
		releases += allocationAudit::frees;
		allocationAudit::allocations = 0;
		allocationAudit::frees = 0;
		allocationAudit::enabled = true;
		const newton::Result* previous = step % 2 && result.primal.size() == 6 * bodies ? &result : NULL;
		newton::solveNewton(problem, settings, result, previous);
		allocationAudit::enabled = false;
		solves += allocationAudit::allocations;
		releases += allocationAudit::frees;
		analyses += result.symbolicAnalyses;
		updates += result.rankUpdates;
		if(!result.primal.allFinite() || result.gradientResidual > 1.0e-6)
			return 1;
	}
	std::printf("ALLOCATIONS,steps=1000,prepare=%zu,solve=%zu,frees=%zu,analyses=%d,rank_updates=%d\n",
		preparation, solves, releases, analyses, updates);
	return preparation || solves || releases || !auditExtended() ? 1 : 0;
}
