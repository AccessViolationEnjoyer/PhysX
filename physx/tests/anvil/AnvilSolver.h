#ifndef ANVIL_TEST_SOLVER_COMPATIBILITY_H
#define ANVIL_TEST_SOLVER_COMPATIBILITY_H
#include "../../source/lowleveldynamics/src/anvil/core/AnvilSolver.h"
#include <cstdio>
#include <cstdlib>

namespace anvil
{
// Compatibility for existing standalone fixtures. Native jobs pass an owned Workspace.
inline void solveAnvil(const Problem& problem, const Settings& settings, Result& result, const Result* previous = NULL)
{
	static thread_local Workspace workspace;
	const SolveStatus::Enum status = solveAnvil(problem, settings, result, workspace, previous);
	if(status != SolveStatus::eSUCCESS && status != SolveStatus::eITERATION_LIMIT)
	{
		std::fprintf(stderr, "Anvil solve failed with status %d\n", int(status));
		std::abort();
	}
}
}
#endif
