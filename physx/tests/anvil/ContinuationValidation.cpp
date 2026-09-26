#include "AnvilSolver.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

static bool check(bool condition, const char* message)
{
	if(!condition)
	{
		std::printf("FAILED,%s\n", message);
	}
	return condition;
}

static anvil::CompactContact scalar(int body0, int body1, int index)
{
	anvil::CompactContact result;
	result.body[0] = body0;
	result.body[1] = body1;
	result.freeVelocity = std::sin(0.67 * (index + 1)) - 0.8;
	result.regularization = 0.03 + 0.01 * (index % 4);
	for(int end = 0; end < 2; ++end)
	{
		for(int axis = 0; axis < 6; ++axis)
		{
			result.jacobian[end][axis] = result.body[end] < 0 ? 0.0 : 0.4 * std::sin(0.37 * (index * 13 + end * 7 + axis + 1));
		}
	}
	return result;
}

static void makeProblem(anvil::Problem& problem, int fixture)
{
	const double maximumImpulse = anvil::MAX_IMPULSE;
	problem.timestep = 0.01;
	problem.inverseMass.assign(3, 1.0);
	problem.massDiagonal.resize(18);
	problem.massDiagonal.setOnes();
	problem.clearContacts();
	problem.addScalarContact(scalar(0, -1, fixture), -0.2, 0.3);
	problem.addScalarContact(scalar(1, 2, fixture + 1), -maximumImpulse, maximumImpulse);
	for(int group = 0; group < 3; ++group)
	{
		const int first = int(problem.contacts.size());
		const int normals = 1 + (fixture + group) % 4;
		const int tangents = group % 2 ? 4 : 2;
		for(int row = 0; row < normals + tangents; ++row)
		{
			problem.addScalarContact(scalar(group, group == 2 ? -1 : group + 1, fixture + row + group * 13),
				row < normals ? 0.0 : -maximumImpulse, row < normals ? 0.25 + 0.1 * row : maximumImpulse);
		}
		problem.addPatch(first, normals, tangents, 0.7);
	}
	anvil::Contact coupled;
	coupled.body[0] = 0;
	coupled.body[1] = 2;
	coupled.bilateral = fixture % 2 == 0;
	coupled.friction = 0.4;
	coupled.regularization.setConstant(0.06);
	coupled.freeVelocity = anvil::Vec3(0.3, -0.2, -0.5);
	for(int end = 0; end < 2; ++end)
	{
		for(int row = 0; row < 3; ++row)
		{
			coupled.jacobian[end].row(row) = scalar(0, 2, row + fixture + 9).jacobian[end];
		}
	}
	problem.addContact(coupled);
	anvil::prepareProblem(problem);
}

static bool compare(const anvil::Problem& problem, const anvil::Settings& settings, const anvil::Result* seed, anvil::Workspace& continuedWorkspace, anvil::Workspace& referenceWorkspace, anvil::Result& continued, anvil::Result& reference, int& factors, int& updates, int& reuse, double& maximum)
{
	const anvil::SolveStatus::Enum actual = anvil::continueAnvil(problem, settings, continued, continuedWorkspace, seed);
	const anvil::SolveStatus::Enum expected = anvil::solveAnvil(problem, settings, reference, referenceWorkspace, seed);
	if(!check(actual == anvil::SolveStatus::eSUCCESS && expected == actual, "continued/reference success"))
	{
		return false;
	}
	double difference = 0.0;
	const int primalCount = int(continued.primal.size());
	for(int i = 0; i < primalCount; ++i)
	{
		difference = std::max(difference, std::abs(continued.primal[i] - reference.primal[i]));
	}
	const int impulseCount = int(continued.impulse.size());
	for(int i = 0; i < impulseCount; ++i)
	{
		difference = std::max(difference, std::abs(continued.impulse[i] - reference.impulse[i]));
	}
	maximum = std::max(maximum, difference);
	factors += continued.factorizations;
	updates += continued.rankUpdates;
	reuse += continued.reusedFactors;
	if(!check(difference <= 1.0e-8, "continued solution matches fresh factor"))
	{
		std::printf("DIFFERENCE,maximum=%.12g,gradient=%.12g,reference_gradient=%.12g\n", difference, continued.scaledGradient, reference.scaledGradient);
		return false;
	}
	return check(continued.factorError <= 1.0e-6, "independent factor direction audit");
}

static bool targetSequences()
{
	anvil::Workspace continuedWorkspace, referenceWorkspace;
	anvil::Settings settings;
	settings.tolerance = 1.0e-13;
	settings.checkFactor = true;
	anvil::Result continued, reference, earlier;
	int factors = 0, updates = 0, reuse = 0, solves = 0;
	double maximum = 0.0;
	for(int fixture = 0; fixture < 12; ++fixture)
	{
		anvil::Problem problem;
		makeProblem(problem, fixture);
		const anvil::Vector originalFree = problem.freeVelocity;
		if(!check(anvil::solveAnvil(problem, settings, continued, continuedWorkspace) == anvil::SolveStatus::eSUCCESS, "initial ordinary solve"))
		{
			return false;
		}
		earlier = continued;
		for(int step = 0; step < 30; ++step)
		{
			const int rowCount = problem.rowCount();
			for(int row = 0; row < rowCount; ++row)
			{
				problem.freeVelocity[row] = originalFree[row] + 0.35 * std::sin(0.29 * step + 0.23 * row);
			}
			const size_t patchCount = problem.patches.size();
			for(size_t patch = 0; patch < patchCount; ++patch)
			{
				problem.patches[patch].friction = 0.2 + 0.1 * ((step / 4 + int(patch)) % 6);
			}
			if(step == 7)
			{
				const std::uint64_t generation = problem.preparationGeneration;
				problem.setScalarBounds(0, -0.03, 0.03);
				if(!check(generation != problem.preparationGeneration, "bounds invalidate generation"))
				{
					return false;
				}
			}
			if(step == 12)
			{
				anvil::CompactContact& contact = problem.contacts[problem.patches[0].firstContact];
				problem.scalarBounds[size_t(anvil::CompactContact::SCALAR_BOUNDS_TAG - contact.block)].upper = 0.08;
				const std::uint64_t generation = problem.preparationGeneration;
				anvil::prepareProblem(problem);
				if(!check(generation != problem.preparationGeneration, "normal caps and preparation invalidate"))
				{
					return false;
				}
			}
			if(step == 18)
			{
				problem.contacts[0].jacobian[0] *= 1.3;
				problem.contacts[0].regularization *= 0.8;
				anvil::prepareProblem(problem);
			}
			const anvil::Result* seed = step % 3 ? &earlier : NULL;
			if(!compare(problem, settings, seed, continuedWorkspace, referenceWorkspace, continued, reference, factors, updates, reuse, maximum))
			{
				return false;
			}
			++solves;
			if(step % 5 == 0)
			{
				earlier = reference;
			}
		}
		// A copied problem can share a generation, but never the identity binding.
		anvil::Problem other = problem;
		if(!compare(other, settings, NULL, continuedWorkspace, referenceWorkspace, continued, reference, factors, updates, reuse, maximum))
		{
			return false;
		}
		++solves;
	}
	std::printf("CONTINUATION,solves=%d,factors=%d,updates=%d,reuse=%d,max_difference=%.12g\n", solves, factors, updates, reuse, maximum);
	return check(updates > 0 && reuse > 0, "tests exercised rank updates and exact factor reuse");
}

static bool invalidation()
{
	const double maximumImpulse = anvil::MAX_IMPULSE;
	anvil::Problem problem;
	problem.timestep = 0.01;
	problem.inverseMass.assign(1, 1.0);
	problem.massDiagonal.resize(6);
	problem.massDiagonal.setOnes();
	anvil::CompactContact contact = scalar(0, -1, 0);
	contact.jacobian[0].setZero();
	contact.jacobian[0][0] = 1.0;
	contact.freeVelocity = -1.0;
	contact.regularization = 0.05;
	problem.addScalarContact(contact, 0.1, 0.1);
	anvil::prepareProblem(problem);
	anvil::Workspace workspace;
	anvil::Settings settings;
	settings.checkFactor = true;
	settings.tolerance = 1.0e-13;
	anvil::Result result;
	if(!check(anvil::solveAnvil(problem, settings, result, workspace) == anvil::SolveStatus::eSUCCESS, "fixed impulse initial"))
	{
		return false;
	}
	problem.setScalarBounds(0, -maximumImpulse, maximumImpulse);
	if(!check(anvil::continueAnvil(problem, settings, result, workspace, &result) == anvil::SolveStatus::eSUCCESS && result.factorizations > 0 && std::abs(result.primal[0] - 1.0 / 1.05) < 1.0e-12, "equality shortcut cannot reuse old fixed-row Hessian"))
	{
		return false;
	}
	problem.freeVelocity[0] = -1.2;
	if(!check(anvil::continueAnvil(problem, settings, result, workspace, &result) == anvil::SolveStatus::eSUCCESS && result.factorizations == 0 && result.reusedFactors > 0, "unchanged equality factor reused for target change"))
	{
		return false;
	}
	problem.freeVelocity[0] = -1.3;
	if(!check(anvil::continueAnvil(problem, settings, result, workspace) == anvil::SolveStatus::eSUCCESS, "changed target succeeds"))
	{
		return false;
	}
	anvil::Workspace moved(std::move(workspace));
	problem.freeVelocity[0] = -1.4;
	if(!check(anvil::continueAnvil(problem, settings, result, moved, &result) == anvil::SolveStatus::eSUCCESS && result.factorizations == 0, "workspace move preserves valid continuation"))
	{
		return false;
	}
	if(!check(anvil::solveAnvil(problem, settings, result, moved) == anvil::SolveStatus::eSUCCESS && result.factorizations > 0, "ordinary solve always resets numerics"))
	{
		return false;
	}
	const std::uint64_t generation = problem.preparationGeneration;
	problem.contacts[0].regularization = 0.08;
	anvil::prepareProblem(problem);
	if(!check(generation != problem.preparationGeneration, "reprepared problem has new generation"))
	{
		return false;
	}
	if(!check(anvil::continueAnvil(problem, settings, result, moved) == anvil::SolveStatus::eSUCCESS && result.factorizations > 0, "repaired preparation refactors"))
	{
		return false;
	}
	std::printf("INVALIDATION,scalar_bounds,equality,ordinary_reset,workspace_move,reprepare=PASS\n");
	return true;
}

int main()
{
	return targetSequences() && invalidation() ? 0 : 1;
}
