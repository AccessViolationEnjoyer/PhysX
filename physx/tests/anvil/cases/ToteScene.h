#ifndef TOTE_SCENE_H
#define TOTE_SCENE_H

// Open totes, each holding four heavy boxes, riding ten long conveyors; shared by the PhysX
// and MuJoCo runners. A tote is one rigid body of five boxes (base and four walls). The 0.2 kg
// tote carries 40 kg, so every box-on-tote contact has a 50:1 mass ratio. Totes are separated
// from their neighbours, so each tote and its boxes form an independent island touching only
// its static belt. The belts, their spacing and friction match the case scene.
#include "CaseScene.h"

namespace totes
{
using cases::conveyorCount;
using cases::conveyorWidth;
using cases::conveyorTop;
using cases::beltThickness;
using cases::beltSpeed;
using cases::beltSpeedAt;
using cases::friction;
using cases::laneZ;

const int totesPerConveyor = 200;
const int toteCount = conveyorCount * totesPerConveyor;
const int boxesPerTote = 4;
const int bodiesPerTote = 1 + boxesPerTote;
const int bodyCount = toteCount * bodiesPerTote;
// A 600 x 400 x 220 mm Euro container with 4 mm walls and base. Its length runs along the belt.
const double toteLength = 0.6;
const double toteWidth = 0.4;
const double toteHeight = 0.22;
const double wallThickness = 0.004;
const double baseThickness = 0.004;
const double toteMass = 0.2;
// Four boxes in a 2 x 2 grid, each centred in its quarter of the 592 x 392 mm floor with 8 mm
// all round, cover 87% of it.
const double boxLength = 0.28;
const double boxWidth = 0.18;
const double boxHeight = 0.15;
const double boxMass = 10.0;
const double toteGap = 0.2;
const double totePitch = toteLength + toteGap;
const double conveyorLength = 180.0;
// PhysX shape contact offset. Pairs make contacts within the sum of their offsets, 1 mm like
// MuJoCo's margin, instead of PhysX's default 4 cm, which would add speculative contacts
// between every box and the nearby walls, and between the walls and the belt.
const double contactOffset = 0.0005;
// Lanes start 1 m from the belts' upstream ends. A tote's frame is at the centre of its base's
// underside, so it rests at the belt surface.
const double firstToteX = -0.5 * conveyorLength + 1.0 + 0.5 * toteLength;
const double boxRestHeight = baseThickness + 0.5 * boxHeight;

// The tote's five boxes in its frame: half extents and centres.
struct Part
{
	double halfExtents[3];
	double centre[3];
};

inline void toteParts(Part parts[5])
{
	const double wallHalfHeight = 0.5 * (toteHeight - baseThickness);
	const double wallY = baseThickness + wallHalfHeight;
	const Part base = { { 0.5 * toteLength, 0.5 * baseThickness, 0.5 * toteWidth }, { 0.0, 0.5 * baseThickness, 0.0 } };
	parts[0] = base;
	// Side walls run the full length; end walls fit between them.
	for(int side = 0; side < 2; ++side)
	{
		const double sign = side ? 1.0 : -1.0;
		const Part wall = { { 0.5 * toteLength, wallHalfHeight, 0.5 * wallThickness }, { 0.0, wallY, sign * 0.5 * (toteWidth - wallThickness) } };
		const Part end = { { 0.5 * wallThickness, wallHalfHeight, 0.5 * toteWidth - wallThickness }, { sign * 0.5 * (toteLength - wallThickness), wallY, 0.0 } };
		parts[1 + side] = wall;
		parts[3 + side] = end;
	}
}

inline double toteVolume()
{
	Part parts[5];
	toteParts(parts);
	double volume = 0.0;
	for(int i = 0; i < 5; ++i)
		volume += 8.0 * parts[i].halfExtents[0] * parts[i].halfExtents[1] * parts[i].halfExtents[2];
	return volume;
}

// Box b's centre in its tote's frame.
inline void boxOffset(int box, double offset[3])
{
	offset[0] = (box & 1 ? 0.25 : -0.25) * (toteLength - 2.0 * wallThickness);
	offset[1] = boxRestHeight;
	offset[2] = (box & 2 ? 0.25 : -0.25) * (toteWidth - 2.0 * wallThickness);
}

inline double maximumDuration()
{
	const double leadingEdge = firstToteX + (totesPerConveyor - 1) * totePitch + 0.5 * toteLength;
	return (0.5 * conveyorLength - leadingEdge) / beltSpeed;
}

// Tote i is number i % totesPerConveyor on lane i / totesPerConveyor, ordered along x.
inline void totePosition(int index, double position[3])
{
	position[0] = firstToteX + (index % totesPerConveyor) * totePitch;
	position[1] = conveyorTop;
	position[2] = laneZ(index / totesPerConveyor);
}

// Bodies are ordered tote, then its four boxes. rotation is (w, x, y, z).
struct State
{
	double position[3];
	double velocity[3];
	double rotation[4];
};

// v in the frame of rotation q, that is, q^-1 v q.
inline void rotateInverse(const double q[4], const double v[3], double result[3])
{
	const double x = -q[1], y = -q[2], z = -q[3], w = q[0];
	const double t[3] = { 2.0 * (y * v[2] - z * v[1]), 2.0 * (z * v[0] - x * v[2]), 2.0 * (x * v[1] - y * v[0]) };
	result[0] = v[0] + w * t[0] + y * t[2] - z * t[1];
	result[1] = v[1] + w * t[1] + z * t[0] - x * t[2];
	result[2] = v[2] + w * t[2] + x * t[1] - y * t[0];
}

struct Metrics
{
	double meanSpeed = 0.0;         // Totes.
	double minimumSpeed = 1.0e30;
	double maximumSpeed = -1.0e30;
	double maximumSink = -1.0e30;   // Totes below the belt surface.
	double maximumLift = -1.0e30;
	double minimumGap = 1.0e30;     // Between neighbouring totes on a lane; starts at toteGap.
	double maximumSideways = 0.0;   // Tote centres from the lane centre.
	double maximumTilt = 0.0;       // Tote up axis from vertical, degrees.
	double boxMinimumSpeed = 1.0e30;
	double boxMaximumSpeed = -1.0e30;
	double boxMaximumSink = -1.0e30; // Boxes into the tote floor, in the tote's frame.
	double boxMaximumLift = -1.0e30;
	double boxMaximumSlip = 0.0;    // Boxes' horizontal travel in the tote's frame.
	int escaped = 0;                // Boxes outside their tote's walls or above its rim.
	int offBelt = 0;                // Totes.
};

inline Metrics measure(const std::vector<State>& states)
{
	Metrics metrics;
	const double interiorX = 0.5 * toteLength - wallThickness, interiorZ = 0.5 * toteWidth - wallThickness;
	for(int i = 0; i < toteCount; ++i)
	{
		const State& tote = states[size_t(i) * bodiesPerTote];
		const double speed = tote.velocity[0];
		metrics.meanSpeed += speed / toteCount;
		metrics.minimumSpeed = std::min(metrics.minimumSpeed, speed);
		metrics.maximumSpeed = std::max(metrics.maximumSpeed, speed);
		const double height = tote.position[1] - conveyorTop;
		metrics.maximumSink = std::max(metrics.maximumSink, -height);
		metrics.maximumLift = std::max(metrics.maximumLift, height);
		metrics.maximumSideways = std::max(metrics.maximumSideways, std::abs(tote.position[2] - laneZ(i / totesPerConveyor)));
		const double* q = tote.rotation;
		const double up = 1.0 - 2.0 * (q[1] * q[1] + q[3] * q[3]);
		metrics.maximumTilt = std::max(metrics.maximumTilt, std::acos(std::max(-1.0, std::min(1.0, up))) * 180.0 / 3.14159265358979323846);
		if(tote.position[1] < conveyorTop - 0.1 || std::abs(tote.position[0]) + 0.5 * toteLength > 0.5 * conveyorLength)
			++metrics.offBelt;
		if(i % totesPerConveyor + 1 < totesPerConveyor)
			metrics.minimumGap = std::min(metrics.minimumGap, states[size_t(i + 1) * bodiesPerTote].position[0] - tote.position[0] - toteLength);
		for(int b = 0; b < boxesPerTote; ++b)
		{
			const State& box = states[size_t(i) * bodiesPerTote + 1 + size_t(b)];
			metrics.boxMinimumSpeed = std::min(metrics.boxMinimumSpeed, box.velocity[0]);
			metrics.boxMaximumSpeed = std::max(metrics.boxMaximumSpeed, box.velocity[0]);
			const double relative[3] = { box.position[0] - tote.position[0], box.position[1] - tote.position[1], box.position[2] - tote.position[2] };
			double local[3], rest[3];
			rotateInverse(q, relative, local);
			boxOffset(b, rest);
			metrics.boxMaximumSink = std::max(metrics.boxMaximumSink, rest[1] - local[1]);
			metrics.boxMaximumLift = std::max(metrics.boxMaximumLift, local[1] - rest[1]);
			metrics.boxMaximumSlip = std::max(metrics.boxMaximumSlip, std::sqrt((local[0] - rest[0]) * (local[0] - rest[0]) + (local[2] - rest[2]) * (local[2] - rest[2])));
			if(std::abs(local[0]) > interiorX || std::abs(local[2]) > interiorZ || local[1] < 0.0 || local[1] > toteHeight)
				++metrics.escaped;
		}
	}
	return metrics;
}

// One row per step, and every body's final state. contacts counts PhysX contact pairs or
// MuJoCo contact points; rows is -1 for PGS, and iterations is the PGS iteration total.
class Recorder
{
	FILE* m_steps;
	std::string m_prefix;

public:
	explicit Recorder(const std::string& prefix) : m_steps(cases::openOutput(prefix + "-steps.csv")), m_prefix(prefix)
	{
		if(m_steps)
			std::fprintf(m_steps, "step,time,step_ms,contacts,rows,iterations,mean_speed,min_speed,max_speed,max_sink_m,max_lift_m,"
				"min_gap_m,max_sideways_m,max_tilt_deg,box_min_speed,box_max_speed,box_max_sink_m,box_max_lift_m,box_max_slip_m,"
				"escaped,off_belt\n");
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
		const Metrics m = measure(states);
		std::fprintf(m_steps, "%d,%.9g,%.9g,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d\n", step, time,
			stepMs, contacts, rows, iterations, m.meanSpeed, m.minimumSpeed, m.maximumSpeed, m.maximumSink, m.maximumLift, m.minimumGap,
			m.maximumSideways, m.maximumTilt, m.boxMinimumSpeed, m.boxMaximumSpeed, m.boxMaximumSink, m.boxMaximumLift, m.boxMaximumSlip,
			m.escaped, m.offBelt);
	}

	void writeFinal(const std::vector<State>& states) const
	{
		FILE* file = cases::openOutput(m_prefix + "-final.csv");
		if(!file)
			return;
		std::fprintf(file, "body,tote,lane,x,y,z,vx,vy,vz,qw,qx,qy,qz\n");
		for(int i = 0; i < bodyCount; ++i)
		{
			const State& s = states[size_t(i)];
			const int tote = i / bodiesPerTote;
			std::fprintf(file, "%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n", i, tote, tote / totesPerConveyor,
				s.position[0], s.position[1], s.position[2], s.velocity[0], s.velocity[1], s.velocity[2], s.rotation[0], s.rotation[1],
				s.rotation[2], s.rotation[3]);
		}
		fclose(file);
	}
};
}
#endif
