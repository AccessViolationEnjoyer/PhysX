#ifndef ANVIL_NATIVE_PROFILER_H
#define ANVIL_NATIVE_PROFILER_H

#include "foundation/PxProfiler.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

// CPU time and wall time (the union of the intervals) of one kind of zone that runs as parallel tasks.
class ZoneIntervals
{
public:
	void reset()
	{
		std::lock_guard<std::mutex> lock(mMutex);
		mIntervals.clear();
		mCpu = 0;
	}

	void add(uint64_t start, uint64_t end)
	{
		std::lock_guard<std::mutex> lock(mMutex);
		mIntervals.push_back(std::make_pair(start, end));
		mCpu += end - start;
	}

	double cpuMilliseconds() const
	{
		return double(mCpu) * 1e-6;
	}

	double wallMilliseconds()
	{
		std::lock_guard<std::mutex> lock(mMutex);
		std::sort(mIntervals.begin(), mIntervals.end());
		uint64_t total = 0, coveredTo = 0;
		for(size_t i = 0; i < mIntervals.size(); ++i)
		{
			const uint64_t start = std::max(mIntervals[i].first, coveredTo);
			if(mIntervals[i].second > start)
			{
				total += mIntervals[i].second - start;
				coveredTo = mIntervals[i].second;
			}
		}
		return double(total) * 1e-6;
	}

private:
	std::mutex mMutex;
	std::vector<std::pair<uint64_t, uint64_t> > mIntervals;
	uint64_t mCpu = 0;
};

// Test-only observer for the SDK's existing profile zones. No event storage or
// allocation is needed; simultaneous islands contribute to one wall-time span.
class NativeSolverProfiler : public physx::PxProfilerCallback
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
	// Collision detection: narrow-phase tasks and broad-phase passes.
	ZoneIntervals narrowPhase, broadPhase;

	NativeSolverProfiler() { reset(); }

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
		narrowPhase.reset();
		broadPhase.reset();
	}

	static bool isNarrowPhase(const char* name)
	{
		return std::strcmp(name, "Sim.narrowPhase") == 0;
	}

	static bool isBroadPhase(const char* name)
	{
		return std::strcmp(name, "Basic.updateBroadPhase") == 0 || std::strcmp(name, "Basic.broadPhaseFirstPass") == 0 ||
			std::strcmp(name, "Basic.broadPhaseSecondPass") == 0;
	}

	static uint64_t now()
	{
		return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count());
	}

	virtual void* zoneStart(const char* name, bool, uint64_t) PX_OVERRIDE
	{
		static const char prefix[] = "Dynamics.anvil";
		if(std::strncmp(name, prefix, sizeof(prefix) - 1) != 0 && !isNarrowPhase(name) && !isBroadPhase(name))
		{
			return NULL;
		}
		return reinterpret_cast<void*>(uintptr_t(now()));
	}

	virtual void zoneEnd(void* data, const char* name, bool, uint64_t) PX_OVERRIDE
	{
		if(!data)
		{
			return;
		}
		const uint64_t end = now();
		// 32-bit targets keep only the start time's low bits; zones are far shorter than 4 s.
		const uint64_t start = end - uint64_t(uintptr_t(uintptr_t(end) - reinterpret_cast<uintptr_t>(data)));
		if(isNarrowPhase(name))
		{
			narrowPhase.add(start, end);
		}
		else if(isBroadPhase(name))
		{
			broadPhase.add(start, end);
		}
		else if(std::strcmp(name, "Dynamics.anvilIsland") == 0)
		{
			uint64_t earliest = firstStart.load();
			while(start < earliest && !firstStart.compare_exchange_weak(earliest, start))
			{
			}
			uint64_t latest = lastEnd.load();
			while(end > latest && !lastEnd.compare_exchange_weak(latest, end))
			{
			}
			islandTime += end - start;
			++islandCount;
		}
		else if(std::strcmp(name, "Dynamics.anvilPrepare") == 0)
		{
			prepareTime += end - start;
		}
		else if(std::strcmp(name, "Dynamics.anvilSolve") == 0)
		{
			uint64_t earliest = firstSolveStart.load();
			while(start < earliest && !firstSolveStart.compare_exchange_weak(earliest, start))
			{
			}
			uint64_t latest = lastSolveEnd.load();
			while(end > latest && !lastSolveEnd.compare_exchange_weak(latest, end))
			{
			}
			solveTime += end - start;
		}
	}

	virtual void recordData(int32_t value, const char* name, uint64_t) PX_OVERRIDE
	{
		if(std::strcmp(name, "Dynamics.anvilIterations") == 0)
		{
			iterations += value;
			int minimum = minimumIterations.load();
			while(value < minimum && !minimumIterations.compare_exchange_weak(minimum, value))
			{
			}
			int maximum = maximumIterations.load();
			while(value > maximum && !maximumIterations.compare_exchange_weak(maximum, value))
			{
			}
		}
		else if(std::strcmp(name, "Dynamics.anvilRows") == 0)
		{
			rows += value;
		}
		else if(std::strcmp(name, "Dynamics.anvilFactorizations") == 0)
		{
			factors += value;
		}
		else if(std::strcmp(name, "Dynamics.anvilRankUpdates") == 0)
		{
			updates += value;
		}
		else if(std::strcmp(name, "Dynamics.anvilLineEvaluations") == 0)
		{
			lineEvaluations += value;
		}
		else if(std::strcmp(name, "Dynamics.anvilStatus") == 0 && value == 1)
		{
			++iterationLimits;
		}
	}

	virtual void recordData(float value, const char* name, uint64_t) PX_OVERRIDE
	{
		std::atomic<float>* maximum = NULL;
		if(std::strcmp(name, "Dynamics.anvilScaledGradient") == 0)
		{
			maximum = &scaledGradient;
		}
		if(maximum)
		{
			float current = maximum->load();
			while(value > current && !maximum->compare_exchange_weak(current, value))
			{
			}
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

