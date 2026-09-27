// The fixed-joint rods of FixedJointTests.cpp, simulated by MuJoCo's Newton solver for comparison.
// Cube 0 is a static body; every other cube is a free body welded to its neighbour by an
// equality constraint, MuJoCo's counterpart of a fixed joint between simulated bodies. Welded
// neighbours do not collide, as PhysX joints default to. The 100 kg cube waits far below the
// scene until it is dropped.
//
// usage: MujocoFixedJointTests scene-directory [threads=1]
//   Welds and contacts use the comparisons' shared solref 0.02 1 and constant solimp 0.9999,
//   the largest impedance MuJoCo allows (regularization 1e-4).
#include <mujoco/mujoco.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static const double cubeSize = 0.1;
static const double rodHeight = 1.0;
static const double dropHeight = 0.5;

static bool writeScene(const std::string& path, int count)
{
	FILE* file = fopen(path.c_str(), "w");
	if(!file)
		return false;
	fprintf(file, "<mujoco model=\"Fixed joint rod\">\n\t<option timestep=\"0.01\" gravity=\"0 -9.81 0\" integrator=\"Euler\" solver=\"Newton\"\n"
		"\t\tcone=\"pyramidal\" jacobian=\"sparse\" iterations=\"100\" tolerance=\"1e-8\" ls_tolerance=\"0.01\"/>\n"
		"\t<default><geom type=\"box\" size=\"%g %g %g\" condim=\"3\" friction=\"0.5 0 0\" margin=\"0.001\" solref=\"0.02 1\" solimp=\"0.9999 0.9999 0.001 0.5 2\"/>\n"
		"\t\t<equality solref=\"0.02 1\" solimp=\"0.9999 0.9999 0.001 0.5 2\"/></default>\n\t<worldbody>\n", 0.5 * cubeSize, 0.5 * cubeSize, 0.5 * cubeSize);
	fprintf(file, "\t\t<body name=\"cube0\" pos=\"0 %g 0\"><geom mass=\"0.1\"/></body>\n", rodHeight);
	for(int i = 1; i < count; ++i)
		fprintf(file, "\t\t<body name=\"cube%d\" pos=\"%g %g 0\"><freejoint/><geom mass=\"%.9g\"/></body>\n", i, i * cubeSize, rodHeight, 0.1 * std::pow(2.0, i));
	fprintf(file, "\t\t<body name=\"weight\" pos=\"0 -1000 0\"><freejoint/><geom mass=\"100\"/></body>\n\t</worldbody>\n\t<equality>\n");
	for(int i = 1; i < count; ++i)
		fprintf(file, "\t\t<weld body1=\"cube%d\" body2=\"cube%d\"/>\n", i - 1, i);
	fprintf(file, "\t</equality>\n\t<contact>\n");
	for(int i = 1; i < count; ++i)
		fprintf(file, "\t\t<exclude body1=\"cube%d\" body2=\"cube%d\"/>\n", i - 1, i);
	fprintf(file, "\t</contact>\n</mujoco>\n");
	fclose(file);
	return true;
}

struct Metrics
{
	double tipDeflection = 0.0;
	double jointGap = 0.0;
	double jointAngle = 0.0;
	double speed = 0.0;
};

// Cube i is body i + 1 (body 0 is the world), so the tip cube of a count-cube rod is body
// count; their frames are at the cube centres.
static Metrics measure(const mjData* data, int count)
{
	Metrics metrics;
	metrics.tipDeflection = rodHeight - data->xpos[3 * count + 1];
	for(int i = 2; i <= count; ++i)
	{
		const mjtNum* rotation0 = data->xmat + 9 * (i - 1);
		const mjtNum* rotation1 = data->xmat + 9 * i;
		mjtNum joint0[3], joint1[3], offset0[3] = { 0.5 * cubeSize, 0.0, 0.0 }, offset1[3] = { -0.5 * cubeSize, 0.0, 0.0 };
		mju_mulMatVec3(joint0, rotation0, offset0);
		mju_mulMatVec3(joint1, rotation1, offset1);
		mju_addTo3(joint0, data->xpos + 3 * (i - 1));
		mju_addTo3(joint1, data->xpos + 3 * i);
		mjtNum difference[3];
		mju_sub3(difference, joint0, joint1);
		metrics.jointGap = std::max(metrics.jointGap, double(mju_norm3(difference)));
		mjtNum conjugate[4], relative[4];
		mju_negQuat(conjugate, data->xquat + 4 * (i - 1));
		mju_mulQuat(relative, conjugate, data->xquat + 4 * i);
		metrics.jointAngle = std::max(metrics.jointAngle, 2.0 * std::acos(std::min(1.0, std::abs(double(relative[0])))));
	}
	for(int i = 1; i < count; ++i)
		metrics.speed = std::max(metrics.speed, double(mju_norm3(data->qvel + 6 * (i - 1))));
	return metrics;
}

static void combine(Metrics& worst, const Metrics& current)
{
	worst.tipDeflection = std::max(worst.tipDeflection, current.tipDeflection);
	worst.jointGap = std::max(worst.jointGap, current.jointGap);
	worst.jointAngle = std::max(worst.jointAngle, current.jointAngle);
	worst.speed = std::max(worst.speed, current.speed);
}

// Step time, excluding each run's first step.
static double totalMs = 0.0;
static int stepCount = 0;

static void timedStep(const mjModel* model, mjData* data)
{
	const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	mj_step(model, data);
	if(stepCount++ > 0)
		totalMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

static bool run(const std::string& directory, int count, bool drop, int threads)
{
	const std::string path = directory + "/fixed_rod_" + std::to_string(count) + ".xml";
	if(!writeScene(path, count))
		return false;
	char error[1024];
	mjModel* model = mj_loadXML(path.c_str(), NULL, error, sizeof(error));
	if(!model)
	{
		fprintf(stderr, "%s\n", error);
		return false;
	}
	mjData* data = mj_makeData(model);
	mju_threadpool(data, threads > 1 ? threads : 0);
	totalMs = 0.0;
	stepCount = 0;
	Metrics worst;
	for(int step = 0; step < 300; ++step)
	{
		timedStep(model, data);
		combine(worst, measure(data, count));
	}
	const Metrics settled = measure(data, count);
	if(!drop)
	{
		printf("fixed_rod_%d mean_step_us=%.2f tip_deflection_mm=%.4f max_joint_gap_mm=%.4f max_joint_angle_deg=%.4f final_speed_mm_s=%.4f"
			" worst_tip_deflection_mm=%.4f worst_joint_gap_mm=%.4f\n", count, 1e3 * totalMs / (stepCount - 1), settled.tipDeflection * 1e3, settled.jointGap * 1e3,
			settled.jointAngle * 180.0 / mjPI, settled.speed * 1e3, worst.tipDeflection * 1e3, worst.jointGap * 1e3);
	}
	else
	{
		// Place the weight above the settled tip at rest; its free joint is the last one.
		const int tip = count;
		mjtNum* weight = data->qpos + 7 * (count - 1);
		mju_copy3(weight, data->xpos + 3 * tip);
		weight[1] += cubeSize + dropHeight;
		weight[3] = 1.0;
		weight[4] = weight[5] = weight[6] = 0.0;
		mju_zero(data->qvel + 6 * (count - 1), 6);
		mj_forward(model, data);
		Metrics after;
		for(int step = 0; step < 300; ++step)
		{
			timedStep(model, data);
			combine(after, measure(data, count));
		}
		const Metrics loaded = measure(data, count);
		mjtNum offset[3], local[3];
		mju_sub3(offset, data->xpos + 3 * (count + 1), data->xpos + 3 * tip);
		mju_mulMatTVec3(local, data->xmat + 9 * tip, offset);
		printf("fixed_rod_drop mean_step_us=%.2f unloaded_tip_deflection_mm=%.4f peak_tip_deflection_mm=%.4f settled_tip_deflection_mm=%.4f"
			" deflection_change_mm=%.4f max_joint_gap_mm=%.4f max_joint_angle_deg=%.4f worst_joint_gap_mm=%.4f final_speed_mm_s=%.4f"
			" weight_offset_mm=%.3f,%.3f,%.3f\n", 1e3 * totalMs / (stepCount - 1), settled.tipDeflection * 1e3, after.tipDeflection * 1e3, loaded.tipDeflection * 1e3,
			(loaded.tipDeflection - settled.tipDeflection) * 1e3, loaded.jointGap * 1e3, loaded.jointAngle * 180.0 / mjPI,
			after.jointGap * 1e3, loaded.speed * 1e3, local[0] * 1e3, local[1] * 1e3, local[2] * 1e3);
	}
	for(int i = 0; i < mjNWARNING; ++i)
	{
		if(data->warning[i].number)
			printf("  WARNING %d count %d\n", i, data->warning[i].number);
	}
	mj_deleteData(data);
	mj_deleteModel(model);
	return true;
}

int main(int argc, const char* const* argv)
{
	if(argc < 2)
	{
		printf("MujocoFixedJointTests scene-directory [threads=1]\n");
		return 1;
	}
	const int threads = argc > 2 ? std::atoi(argv[2]) : 1;
	printf("MuJoCo %s Newton, constant impedance 0.9999\n", mj_versionString());
	const bool ok = run(argv[1], 10, false, threads) && run(argv[1], 20, false, threads) && run(argv[1], 10, true, threads);
	return ok ? 0 : 1;
}
