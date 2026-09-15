#ifndef NEWTON_NATIVE_PROFILER_H
#define NEWTON_NATIVE_PROFILER_H

#include "foundation/PxProfiler.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>

// Test-only observer for the SDK's existing profile zones. No event storage or
// allocation is needed; simultaneous islands contribute to one wall-time span.
class NativeNewtonProfiler : public physx::PxProfilerCallback
{
public:
	std::atomic<uint64_t> firstStart;
	std::atomic<uint64_t> lastEnd;
	std::atomic<uint64_t> firstSolveStart;
	std::atomic<uint64_t> lastSolveEnd;
	std::atomic<uint64_t> islandTime;
	std::atomic<uint64_t> prepareTime;
	std::atomic<uint64_t> solveTime;
	std::atomic<int> iterations;
	std::atomic<int> minimumIterations;
	std::atomic<int> maximumIterations;
	std::atomic<int> iterationLimits;
	std::atomic<int> islandCount;
	std::atomic<int> rows;
	std::atomic<int> factors, updates, lineEvaluations;
	std::atomic<float> scaledGradient;

	NativeNewtonProfiler() { reset(); }

	void reset()
	{
		firstStart = std::numeric_limits<uint64_t>::max();
		firstSolveStart = std::numeric_limits<uint64_t>::max();
		lastEnd = islandTime = prepareTime = solveTime = 0;
		lastSolveEnd = 0;
		iterations = iterationLimits = islandCount = rows = factors = updates = lineEvaluations = 0;
		minimumIterations = std::numeric_limits<int>::max();
		maximumIterations = 0;
		scaledGradient = 0.0f;
	}

	static uint64_t now()
	{
		return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count());
	}

	virtual void* zoneStart(const char* name, bool, uint64_t) PX_OVERRIDE
	{
		if(std::strncmp(name, "Dynamics.newton", 15) != 0)
			return NULL;
		return reinterpret_cast<void*>(uintptr_t(now()));
	}

	virtual void zoneEnd(void* data, const char* name, bool, uint64_t) PX_OVERRIDE
	{
		if(!data)
			return;
		const uint64_t start = uint64_t(reinterpret_cast<uintptr_t>(data));
		const uint64_t end = now();
		if(std::strcmp(name, "Dynamics.newtonIsland") == 0)
		{
			uint64_t earliest = firstStart.load();
			while(start < earliest && !firstStart.compare_exchange_weak(earliest, start)) {}
			uint64_t latest = lastEnd.load();
			while(end > latest && !lastEnd.compare_exchange_weak(latest, end)) {}
			islandTime += end - start;
			++islandCount;
		}
		else if(std::strcmp(name, "Dynamics.newtonPrepare") == 0)
			prepareTime += end - start;
		else if(std::strcmp(name, "Dynamics.newtonSolve") == 0)
		{
			uint64_t earliest = firstSolveStart.load();
			while(start < earliest && !firstSolveStart.compare_exchange_weak(earliest, start)) {}
			uint64_t latest = lastSolveEnd.load();
			while(end > latest && !lastSolveEnd.compare_exchange_weak(latest, end)) {}
			solveTime += end - start;
		}
	}

	virtual void recordData(int32_t value, const char* name, uint64_t) PX_OVERRIDE
	{
		if(std::strcmp(name, "Dynamics.newtonIterations") == 0)
		{
			iterations += value;
			int minimum = minimumIterations.load();
			while(value < minimum && !minimumIterations.compare_exchange_weak(minimum, value)) {}
			int maximum = maximumIterations.load();
			while(value > maximum && !maximumIterations.compare_exchange_weak(maximum, value)) {}
		}
		else if(std::strcmp(name, "Dynamics.newtonRows") == 0)
			rows += value;
		else if(std::strcmp(name, "Dynamics.newtonFactorizations") == 0)
			factors += value;
		else if(std::strcmp(name, "Dynamics.newtonRankUpdates") == 0)
			updates += value;
		else if(std::strcmp(name, "Dynamics.newtonLineEvaluations") == 0)
			lineEvaluations += value;
		else if(std::strcmp(name, "Dynamics.newtonStatus") == 0 && value == 1)
			++iterationLimits;
	}

	virtual void recordData(float value, const char* name, uint64_t) PX_OVERRIDE
	{
		std::atomic<float>* maximum = NULL;
		if(std::strcmp(name, "Dynamics.newtonScaledGradient") == 0)
			maximum = &scaledGradient;
		if(maximum)
		{
			float current = maximum->load();
			while(value > current && !maximum->compare_exchange_weak(current, value)) {}
		}
	}

	double wallMilliseconds() const
	{
		return islandCount.load() ? double(lastEnd.load() - firstStart.load()) * 1e-6 : -1.0;
	}

	double solveWallMilliseconds() const
	{
		return lastSolveEnd.load() ? double(lastSolveEnd.load() - firstSolveStart.load()) * 1e-6 : -1.0;
	}
};
#endif


