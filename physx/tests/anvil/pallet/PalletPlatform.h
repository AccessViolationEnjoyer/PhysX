#ifndef PALLET_PLATFORM_H
#define PALLET_PLATFORM_H

// One pallet of boxes and slipsheets on a platform shuttling from side to side, shared by the
// PhysX and MuJoCo runners. The platform accelerates at 0.5 m/s^2 to 0.5 m/s, cruises, and
// decelerates to rest at each end of a 1 m stroke along x, so friction must hold the whole
// stack through every reversal. The per-step measurements show how far the pallet slips on the
// platform and the boxes and sheets drift on the pallet.
#include "PalletScene.h"

namespace platform
{
const double maximumSpeed = 0.5;
const double acceleration = 0.5;
const double stroke = 1.0;
const double halfExtents[3] = { 0.8, 0.05, 0.7 };
const double top = pallet::conveyorTop;
// The stack and its platform start at rest at the stroke's lower end.
const double startX = -0.5 * stroke;
// PhysX shape contact offset: pairs make contacts within 1 mm, like MuJoCo's margin.
const double contactOffset = 0.0005;

// Stroke time: accelerate to full speed, cruise, and decelerate to rest.
inline double strokeDuration()
{
	const double ramp = maximumSpeed / acceleration;
	return 2.0 * ramp + (stroke - maximumSpeed * ramp) / maximumSpeed;
}

// Platform x at a time: strokes alternate towards +x and back to -x.
inline double position(double time)
{
	const double duration = strokeDuration(), ramp = maximumSpeed / acceleration;
	const int index = int(std::floor(time / duration));
	const double t = time - index * duration;
	double travel;
	if(t < ramp)
		travel = 0.5 * acceleration * t * t;
	else if(t < duration - ramp)
		travel = 0.5 * maximumSpeed * ramp + maximumSpeed * (t - ramp);
	else
		travel = stroke - 0.5 * acceleration * (duration - t) * (duration - t);
	return index % 2 == 0 ? startX + travel : startX + stroke - travel;
}

// The middle lane's pallet of the conveyor scene (at z = 0), moved onto the platform. The
// pallet comes first, then its boxes and sheets.
inline std::vector<pallet::Body> createStack()
{
	const int lane = pallet::conveyorCount / 2;
	const std::vector<pallet::Body> bodies = pallet::createBodies();
	const double shift = startX - bodies[size_t(lane * pallet::bodiesPerPallet)].position[0];
	std::vector<pallet::Body> stack;
	for(size_t i = 0; i < bodies.size(); ++i)
	{
		if(bodies[i].lane != lane)
			continue;
		pallet::Body body = bodies[i];
		body.lane = 0;
		body.position[0] += shift;
		stack.push_back(body);
	}
	return stack;
}

struct Metrics
{
	double palletSlip = 0.0;     // Pallet centre from the platform centre, horizontally.
	double boxDrift = 0.0;       // Largest horizontal travel of a box in the pallet's frame.
	double sheetDrift = 0.0;
	double relativeSpeed = 0.0;  // Largest body speed relative to the platform.
	pallet::Metrics stack;       // Slipsheet overlap, crossings and lowest box.
};

// Poses follow createStack's order; the platform moves along x only.
inline Metrics measure(const std::vector<pallet::Body>& bodies, const std::vector<pallet::Pose>& poses, double platformX, double platformSpeed)
{
	Metrics metrics;
	const pallet::Pose& base = poses[0];
	const double slip[2] = { base.position[0] - platformX, base.position[2] };
	metrics.palletSlip = std::sqrt(slip[0] * slip[0] + slip[1] * slip[1]);
	for(size_t i = 0; i < bodies.size(); ++i)
	{
		const pallet::Pose& pose = poses[i];
		const double velocity[3] = { pose.velocity[0] - platformSpeed, pose.velocity[1], pose.velocity[2] };
		metrics.relativeSpeed = std::max(metrics.relativeSpeed, std::sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]));
		if(bodies[i].kind == pallet::PALLET)
			continue;
		// Start and current positions in the pallet's frame; it starts unrotated.
		double drift[3];
		for(int axis = 0; axis < 3; ++axis)
		{
			double local = 0.0;
			for(int k = 0; k < 3; ++k)
				local += base.rotation[k * 3 + axis] * (pose.position[k] - base.position[k]);
			drift[axis] = local - (bodies[i].position[axis] - bodies[0].position[axis]);
		}
		const double horizontal = std::sqrt(drift[0] * drift[0] + drift[2] * drift[2]);
		double& largest = bodies[i].kind == pallet::BOX ? metrics.boxDrift : metrics.sheetDrift;
		largest = std::max(largest, horizontal);
	}
	metrics.stack = pallet::measure(bodies, poses);
	return metrics;
}

// One row per step. contacts counts PhysX contact pairs or MuJoCo contact points;
// iterations is the Anvil iteration total (0 outside profile builds) or PGS iterations.
class Recorder
{
	FILE* m_steps;

public:
	explicit Recorder(const std::string& prefix) : m_steps(pallet::openOutput(prefix + "-steps.csv"))
	{
		if(m_steps)
			std::fprintf(m_steps, "step,time,step_ms,contacts,iterations,platform_x,platform_speed,pallet_slip_m,box_drift_m,sheet_drift_m,"
				"max_relative_speed,sheet_overlap_m,through_sheet,minimum_box_y\n");
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

	void record(int step, double time, double stepMs, int contacts, int iterations, double platformX, double platformSpeed,
		const std::vector<pallet::Body>& bodies, const std::vector<pallet::Pose>& poses)
	{
		const Metrics m = measure(bodies, poses, platformX, platformSpeed);
		std::fprintf(m_steps, "%d,%.9g,%.9g,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%.9g\n", step, time, stepMs, contacts, iterations,
			platformX, platformSpeed, m.palletSlip, m.boxDrift, m.sheetDrift, m.relativeSpeed, m.stack.penetration, m.stack.throughSheet,
			m.stack.minimumBoxHeight);
	}
};
}
#endif
