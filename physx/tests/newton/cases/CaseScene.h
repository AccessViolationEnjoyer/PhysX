#ifndef CASE_SCENE_H
#define CASE_SCENE_H

// Individual cases riding ten long conveyors, shared by the PhysX and MuJoCo runners.
// Every case rests on its own belt, separated from its neighbours, so each case is an
// independent island touching only its static belt.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace cases
{
const int conveyorCount = 10;
const int casesPerConveyor = 200;
const int caseCount = conveyorCount * casesPerConveyor;
const double caseSize = 0.4;
const double caseGap = 0.2;
const double caseMass = 10.0;
const double friction = 0.5;
const double beltSpeed = 0.5;
const double conveyorWidth = 1.0;
const double conveyorPitch = 1.5;
const double conveyorLength = 140.0;
const double conveyorTop = 0.5;
const double beltThickness = 0.2;
// Lanes start 1 m from the belts' upstream ends.
const double firstCaseX = -0.5 * conveyorLength + 1.0 + 0.5 * caseSize;
const double restHeight = conveyorTop + 0.5 * caseSize;

inline double laneZ(int lane)
{
	return (lane - 0.5 * (conveyorCount - 1)) * conveyorPitch;
}

// Cases start at rest; the belts carry the leading case this far before it reaches the end.
inline double maximumDuration()
{
	const double leadingEdge = firstCaseX + (casesPerConveyor - 1) * (caseSize + caseGap) + 0.5 * caseSize;
	return (0.5 * conveyorLength - leadingEdge) / beltSpeed;
}

// Case i is number i % casesPerConveyor on lane i / casesPerConveyor, ordered along x.
inline void casePosition(int index, double position[3])
{
	position[0] = firstCaseX + (index % casesPerConveyor) * (caseSize + caseGap);
	position[1] = restHeight;
	position[2] = laneZ(index / casesPerConveyor);
}

struct State
{
	double position[3];
	double velocity[3];
};

struct Metrics
{
	double meanSpeed = 0.0;
	double minimumSpeed = 1.0e30;
	double maximumSpeed = -1.0e30;
	double maximumSink = -1.0e30;   // Below the resting height: penetration into the belt.
	double maximumLift = -1.0e30;   // Above the resting height.
	double minimumGap = 1.0e30;     // Between neighbours on a lane; starts at caseGap.
	double maximumSideways = 0.0;   // From the lane centre.
	int offBelt = 0;
};

inline Metrics measure(const std::vector<State>& states)
{
	Metrics metrics;
	for(int i = 0; i < caseCount; ++i)
	{
		const State& state = states[size_t(i)];
		const double speed = state.velocity[0];
		metrics.meanSpeed += speed / caseCount;
		metrics.minimumSpeed = std::min(metrics.minimumSpeed, speed);
		metrics.maximumSpeed = std::max(metrics.maximumSpeed, speed);
		const double height = state.position[1] - restHeight;
		metrics.maximumSink = std::max(metrics.maximumSink, -height);
		metrics.maximumLift = std::max(metrics.maximumLift, height);
		metrics.maximumSideways = std::max(metrics.maximumSideways, std::abs(state.position[2] - laneZ(i / casesPerConveyor)));
		if(state.position[1] < conveyorTop || std::abs(state.position[0]) + 0.5 * caseSize > 0.5 * conveyorLength)
			++metrics.offBelt;
		if(i % casesPerConveyor + 1 < casesPerConveyor)
			metrics.minimumGap = std::min(metrics.minimumGap, states[size_t(i) + 1].position[0] - state.position[0] - caseSize);
	}
	return metrics;
}

inline FILE* openOutput(const std::string& path)
{
	FILE* file = fopen(path.c_str(), "w");
	if(!file)
		std::fprintf(stderr, "Cannot write %s\n", path.c_str());
	return file;
}

// One row per step, and every case's final state. contacts counts PhysX contact pairs or
// MuJoCo contact points; rows is -1 for PGS, and iterations is the PGS iteration total.
class Recorder
{
	FILE* m_steps;
	std::string m_prefix;

public:
	explicit Recorder(const std::string& prefix) : m_steps(openOutput(prefix + "-steps.csv")), m_prefix(prefix)
	{
		if(m_steps)
			std::fprintf(m_steps, "step,time,step_ms,contacts,rows,iterations,mean_speed,min_speed,max_speed,"
				"max_sink_m,max_lift_m,min_gap_m,max_sideways_m,off_belt\n");
	}

	~Recorder()
	{
		if(m_steps)
			fclose(m_steps);
	}

	bool isValid() const
	{
		return m_steps != NULL;
	}

	void record(int step, double time, double stepMs, int contacts, int rows, int iterations, const std::vector<State>& states)
	{
		const Metrics metrics = measure(states);
		std::fprintf(m_steps, "%d,%.9g,%.9g,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n", step, time, stepMs, contacts, rows,
			iterations, metrics.meanSpeed, metrics.minimumSpeed, metrics.maximumSpeed, metrics.maximumSink, metrics.maximumLift,
			metrics.minimumGap, metrics.maximumSideways, metrics.offBelt);
	}

	void writeFinal(const std::vector<State>& states) const
	{
		FILE* file = openOutput(m_prefix + "-final.csv");
		if(!file)
			return;
		std::fprintf(file, "case,lane,x,y,z,vx,vy,vz\n");
		for(int i = 0; i < caseCount; ++i)
		{
			const State& state = states[size_t(i)];
			std::fprintf(file, "%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n", i, i / casesPerConveyor, state.position[0],
				state.position[1], state.position[2], state.velocity[0], state.velocity[1], state.velocity[2]);
		}
		fclose(file);
	}
};

typedef std::chrono::steady_clock Clock;
inline double elapsed(Clock::time_point start)
{
	return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}
#endif
