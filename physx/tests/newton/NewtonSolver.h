#ifndef NEWTON_TEST_SOLVER_COMPATIBILITY_H
#define NEWTON_TEST_SOLVER_COMPATIBILITY_H
#include "../../source/lowleveldynamics/src/newton/core/NewtonSolver.h"
#include <stdexcept>

namespace newton
{
// Compatibility for existing standalone fixtures. Native jobs pass an owned Workspace.
inline void solveNewton(const Problem& problem, const Settings& settings, Result& result, const Result* previous = NULL)
{
	static thread_local Workspace workspace;
	const SolveStatus::Enum status = solveNewton(problem, settings, result, workspace, previous);
	if(status != SolveStatus::eSUCCESS && status != SolveStatus::eITERATION_LIMIT)
		throw std::runtime_error("Newton solve failed with status " + std::to_string(int(status)));
}
}
#endif
