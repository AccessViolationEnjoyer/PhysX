#ifndef PALLET_SCENE_H
#define PALLET_SCENE_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace pallet
{
const int conveyorCount = 5;
const int layerCount = 5;
const int boxesPerLayer = 12;
const int bodiesPerPallet = 1 + layerCount * boxesPerLayer + layerCount - 1;
const double beltSpeed = 0.2;
const double sheetThickness = 0.001;
const double conveyorWidth = 1.2;
const double conveyorLength = 12.0;
const double conveyorPitch = conveyorWidth + 1.0;
const double conveyorTop = 0.5;

enum BodyKind
{
	PALLET,
	BOX,
	SHEET
};

struct Body
{
	std::string name;
	BodyKind kind;
	int lane;
	int layer;
	double position[3];
	double halfSize[3];
	double mass;
};

inline std::vector<Body> createBodies()
{
	std::vector<Body> bodies;
	for(int lane = 0; lane < conveyorCount; ++lane)
	{
		const double z = (lane - 2) * conveyorPitch;
		const std::string prefix = "lane" + std::to_string(lane + 1) + "_";
		Body base = {prefix + "pallet", PALLET, lane, -1, {-3.0, conveyorTop + 0.075, z}, {0.6, 0.075, 0.5}, 10.0};
		bodies.push_back(base);
		double bottom = conveyorTop + 0.15;
		for(int layer = 0; layer < layerCount; ++layer)
		{
			for(int row = 0; row < 3; ++row)
			{
				for(int column = 0; column < 4; ++column)
				{
					Body box = {prefix + "box" + std::to_string(layer * boxesPerLayer + row * 4 + column + 1),
						BOX, lane, layer, {-3.0 + (column - 1.5) * 0.3, bottom + 0.125, z + (row - 1) * 0.3},
						{0.15, 0.125, 0.15}, 1.0};
					bodies.push_back(box);
				}
			}
			bottom += 0.25;
			if(layer + 1 < layerCount)
			{
				Body sheet = {prefix + "sheet" + std::to_string(layer + 1), SHEET, lane, layer,
					{-3.0, bottom + sheetThickness * 0.5, z}, {0.6, sheetThickness * 0.5, 0.5}, 0.1};
				bodies.push_back(sheet);
				bottom += sheetThickness;
			}
		}
	}
	return bodies;
}

struct Pose
{
	double position[3];
	double rotation[9]; // Row-major body-to-world rotation.
	double velocity[3];
};

struct Metrics
{
	double palletX = 0.0;
	double palletSpeed = 0.0;
	double penetration = 0.0;
	double minimumBoxHeight = 1.0e30;
	int throughSheet = 0;
};

// Measure geometric overlap after integration, independently of either engine's
// contact offsets. A box intrudes through a sheet when overlap exceeds 1 mm.
inline Metrics measure(const std::vector<Body>& bodies, const std::vector<Pose>& poses)
{
	Metrics metrics;
	for(size_t i = 0; i < bodies.size(); ++i)
	{
		if(bodies[i].kind == PALLET)
		{
			metrics.palletX += poses[i].position[0] / conveyorCount;
			metrics.palletSpeed += poses[i].velocity[0] / conveyorCount;
		}
		if(bodies[i].kind == BOX)
			metrics.minimumBoxHeight = std::min(metrics.minimumBoxHeight, poses[i].position[1]);
		if(bodies[i].kind != SHEET)
			continue;
		const Pose& sheet = poses[i];
		const size_t first = size_t(bodies[i].lane * bodiesPerPallet);
		for(size_t j = first; j < first + bodiesPerPallet; ++j)
		{
			const Body& box = bodies[j];
			if(box.kind != BOX || (box.layer != bodies[i].layer && box.layer != bodies[i].layer + 1))
				continue;
			double distance[3] = {}, radius[3] = {};
			for(int axis = 0; axis < 3; ++axis)
			{
				for(int k = 0; k < 3; ++k)
					distance[axis] += sheet.rotation[k * 3 + axis] * (poses[j].position[k] - sheet.position[k]);
				for(int k = 0; k < 3; ++k)
				{
					double projection = 0.0;
					for(int r = 0; r < 3; ++r)
						projection += sheet.rotation[r * 3 + axis] * poses[j].rotation[r * 3 + k];
					radius[axis] += std::abs(projection) * box.halfSize[k];
				}
			}
			if(std::abs(distance[0]) >= bodies[i].halfSize[0] + radius[0] ||
				std::abs(distance[2]) >= bodies[i].halfSize[2] + radius[2])
				continue;
			const double side = box.layer > bodies[i].layer ? 1.0 : -1.0;
			const double penetration = radius[1] + sheetThickness * 0.5 - side * distance[1];
			metrics.penetration = std::max(metrics.penetration, penetration);
			if(penetration > sheetThickness)
				++metrics.throughSheet;
		}
	}
	return metrics;
}

inline FILE* openOutput(const std::string& path)
{
	FILE* file = fopen(path.c_str(), "w");
	if(!file)
		throw std::runtime_error("Cannot write " + path);
	return file;
}

class Recorder
{
	FILE* m_steps;
	FILE* m_poses;

public:
	Recorder(const std::string& prefix, const std::vector<Body>& bodies)
	{
		m_steps = openOutput(prefix + "-steps.csv");
		m_poses = openOutput(prefix + "-poses.csv");
		fprintf(m_steps, "step,time,step_ms,solve_ms,contact_pairs,contact_points,rows,iterations,pallet_x,pallet_speed,sheet_overlap_m,through_sheet,minimum_box_y\n");
		fprintf(m_poses, "step,body,x,y,z,r00,r01,r02,r10,r11,r12,r20,r21,r22\n");
		FILE* file = openOutput(prefix + "-bodies.csv");
		fprintf(file, "body,name,kind,lane,layer,hx,hy,hz,mass\n");
		for(size_t i = 0; i < bodies.size(); ++i)
		{
			const Body& body = bodies[i];
			fprintf(file, "%zu,%s,%d,%d,%d,%.9g,%.9g,%.9g,%.9g\n", i, body.name.c_str(), int(body.kind),
				body.lane, body.layer, body.halfSize[0], body.halfSize[1], body.halfSize[2], body.mass);
		}
		fclose(file);
	}

	~Recorder()
	{
		fclose(m_steps);
		fclose(m_poses);
	}

	void record(int step, double time, double stepMs, double solveMs, int pairs, int points, int rows, int iterations,
		const std::vector<Body>& bodies, const std::vector<Pose>& poses)
	{
		const Metrics metrics = measure(bodies, poses);
		fprintf(m_steps, "%d,%.9g,%.9g,%.9g,%d,%d,%d,%d,%.9g,%.9g,%.9g,%d,%.9g\n", step, time, stepMs, solveMs,
			pairs, points, rows, iterations, metrics.palletX, metrics.palletSpeed, metrics.penetration,
			metrics.throughSheet, metrics.minimumBoxHeight);
		if(step % 10 == 0)
		{
			for(size_t i = 0; i < poses.size(); ++i)
			{
				fprintf(m_poses, "%d,%zu", step, i);
				for(int j = 0; j < 3; ++j)
					fprintf(m_poses, ",%.12g", poses[i].position[j]);
				for(int j = 0; j < 9; ++j)
					fprintf(m_poses, ",%.12g", poses[i].rotation[j]);
				fprintf(m_poses, "\n");
			}
		}
	}
};

typedef std::chrono::steady_clock Clock;
inline double elapsed(Clock::time_point start)
{
	return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}
#endif


