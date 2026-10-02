// PEEL's "PileOfLargeConvexes" (ConvexPileScene.h) in MuJoCo's Newton solver, to compare step times
// with AnvilConvexPile. Solver and contact settings match the other MuJoCo comparison scenes; the
// hull is a mesh geom, whose collision shape MuJoCo takes as the points' convex hull.
#include "ConvexPileScene.h"
#include "MujocoConveyor.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

static bool writeScene(const std::string& path, int columns, int layers)
{
	FILE* file = std::fopen(path.c_str(), "w");
	if(!file)
		return false;
	float points[3 * convexPile::pointCount];
	convexPile::hullPoints(points);
	std::fprintf(file, "<mujoco model=\"Convex pile\">\n"
		"\t<compiler angle=\"radian\"/>\n\t<size memory=\"256M\"/>\n"
		"\t<option timestep=\"0.01\" gravity=\"0 -9.81 0\" integrator=\"Euler\" solver=\"Newton\"\n"
		"\t\tcone=\"pyramidal\" jacobian=\"sparse\" iterations=\"100\" tolerance=\"1e-8\" ls_tolerance=\"0.01\"/>\n"
		"\t<default><geom condim=\"3\" friction=\"0.5 0 0\" margin=\"0.001\" gap=\"0\"\n"
		"\t\tsolref=\"0.02 1\" solimp=\"0.9999 0.9999 0.001 0.5 2\"/></default>\n"
		"\t<asset><mesh name=\"hull\" inertia=\"convex\" vertex=\"");
	for(int i = 0; i < 3 * convexPile::pointCount; ++i)
		std::fprintf(file, i ? " %.9g" : "%.9g", double(points[i]));
	std::fprintf(file, "\"/></asset>\n"
		"\t<worldbody>\n\t\t<geom name=\"floor\" type=\"plane\" size=\"20 20 0.1\" euler=\"-1.5707963267948966 0 0\"/>\n");
	for(int layer = 0; layer < layers; ++layer)
	{
		for(int row = 0; row < columns; ++row)
		{
			for(int column = 0; column < columns; ++column)
			{
				double position[3];
				convexPile::position(column, row, layer, columns, position);
				std::fprintf(file, "\t\t<body pos=\"%.9g %.9g %.9g\"><freejoint/><geom type=\"mesh\" mesh=\"hull\" mass=\"%.9g\"/></body>\n",
					position[0], position[1], position[2], convexPile::mass);
			}
		}
	}
	std::fprintf(file, "\t</worldbody>\n</mujoco>\n");
	std::fclose(file);
	return true;
}

int main(int argc, const char* const* argv)
{
	if(argc < 2)
	{
		std::printf("MujocoConvexPile output.csv [steps=800] [threads=8] [columns=5] [layers=20]\n");
		return 1;
	}
	const int steps = argc > 2 ? std::atoi(argv[2]) : 800;
	const int threads = argc > 3 ? std::atoi(argv[3]) : 8;
	const int columns = argc > 4 ? std::atoi(argv[4]) : 5;
	const int layers = argc > 5 ? std::atoi(argv[5]) : 20;
	const std::string scene = std::string(argv[1]) + ".xml";
	if(!writeScene(scene, columns, layers))
		return 1;
	char error[1024];
	mjModel* model = mj_loadXML(scene.c_str(), NULL, error, sizeof(error));
	if(!model)
		throw std::runtime_error(error);
	mjData* data = mj_makeData(model);
	mju_threadpool(data, threads > 1 ? threads : 0);
	FILE* output = std::fopen(argv[1], "w");
	if(!output)
		return 1;
	std::printf("MuJoCo %s Newton convex pile: %dx%dx%d, %d hulls of %d points, dt=%.3g, cap %d, %d workers\n",
		mj_versionString(), columns, columns, layers, int(model->nbody - 1), convexPile::pointCount, model->opt.timestep, int(model->opt.iterations), threads);
	std::fprintf(output, "step,step_ms,contacts,iterations,minimum_y,maximum_y,mean_y,maximum_speed\n");
	bool finite = true;
	for(int step = 0; step < steps; ++step)
	{
		const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		mj_step1(model, data);
		mujocoConveyor::applyRestDistance(data);
		mj_fwdActuation(model, data);
		mj_fwdAcceleration(model, data);
		mj_fwdConstraint(model, data);
		mj_sensorAcc(model, data);
		mj_checkAcc(model, data);
		mj_Euler(model, data);
		const double stepMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		int iterations = 0;
		for(int i = 0; i < std::min(data->nisland, mjNISLAND); ++i)
			iterations += data->solver_niter[i];
		double minimumY = 1e30, maximumY = -1e30, sumY = 0.0, maximumSpeed = 0.0;
		for(int body = 0; body < model->nbody - 1; ++body)
		{
			const mjtNum* qpos = data->qpos + 7 * body;
			const mjtNum* qvel = data->qvel + 6 * body;
			finite = finite && std::isfinite(qpos[1]) && std::isfinite(qvel[0]);
			minimumY = std::min(minimumY, double(qpos[1]));
			maximumY = std::max(maximumY, double(qpos[1]));
			sumY += double(qpos[1]);
			maximumSpeed = std::max(maximumSpeed, std::sqrt(double(qvel[0] * qvel[0] + qvel[1] * qvel[1] + qvel[2] * qvel[2])));
		}
		std::fprintf(output, "%d,%.9g,%d,%d,%.9g,%.9g,%.9g,%.9g\n", step + 1, stepMs, int(data->ncon), iterations, minimumY, maximumY, sumY / double(model->nbody - 1), maximumSpeed);
	}
	std::fclose(output);
	for(int i = 0; i < mjNWARNING; ++i)
	{
		if(data->warning[i].number)
			std::printf("WARNING %d count %d\n", i, data->warning[i].number);
	}
	mj_deleteData(data);
	mj_deleteModel(model);
	return finite ? 0 : 1;
}
