// One pallet on a platform shuttling from side to side (PalletPlatform.h), simulated by
// MuJoCo's Newton solver. The platform is a mocap body, moved to its next position after each
// step; like the conveyor belts, its contacts take its velocity over the step through the
// reference acceleration, as a PhysX kinematic body's contacts do.
#include "PalletPlatform.h"
#include "MujocoConveyor.h"
#include <cstdlib>
#include <stdexcept>

// Solver and contact settings match the pallet conveyor scene. The platform's geom is the
// first, so the shared conveyor adjustment treats it as the only belt.
static bool writeScene(const char* path)
{
	FILE* file = pallet::openOutput(path);
	if(!file)
		return false;
	fprintf(file, "<mujoco model=\"Pallet platform\">\n"
		"\t<compiler angle=\"radian\"/>\n\t<size memory=\"64M\"/>\n"
		"\t<option timestep=\"0.01\" gravity=\"0 -9.81 0\" integrator=\"Euler\" solver=\"Newton\"\n"
		"\t\tcone=\"pyramidal\" jacobian=\"sparse\" iterations=\"100\" tolerance=\"1e-8\" ls_tolerance=\"0.01\"/>\n"
		"\t<default><geom type=\"box\" condim=\"3\" friction=\"0.5 0 0\" margin=\"0.001\" gap=\"0\"\n"
		"\t\tsolref=\"0.02 1\" solimp=\"0.990099 0.9999 0.00002 0.5 2\"/></default>\n"
		"\t<worldbody>\n\t\t<light pos=\"0 10 0\" dir=\"0 -1 0\" directional=\"true\"/>\n"
		"\t\t<body name=\"platform\" mocap=\"true\" pos=\"%.9g %.9g 0\">\n"
		"\t\t\t<geom size=\"%.9g %.9g %.9g\" rgba=\"0.2 0.25 0.3 1\"/>\n\t\t</body>\n",
		platform::startX, platform::top - platform::halfExtents[1], platform::halfExtents[0], platform::halfExtents[1], platform::halfExtents[2]);
	const std::vector<pallet::Body> bodies = platform::createStack();
	for(size_t i = 0; i < bodies.size(); ++i)
	{
		const pallet::Body& body = bodies[i];
		fprintf(file, "\t\t<body name=\"%s\" pos=\"%.9g %.9g %.9g\"><freejoint/>\n"
			"\t\t\t<geom size=\"%.9g %.9g %.9g\" mass=\"%.9g\" rgba=\"%s\"/>\n\t\t</body>\n",
			body.name.c_str(), body.position[0], body.position[1], body.position[2], body.halfSize[0], body.halfSize[1], body.halfSize[2],
			body.mass, body.kind == pallet::PALLET ? "0.5 0.3 0.12 1" : body.kind == pallet::BOX ? "0.85 0.7 0.45 1" : "0.9 0.9 0.85 1");
	}
	fprintf(file, "\t</worldbody>\n</mujoco>\n");
	fclose(file);
	return true;
}

int main(int argc, const char* const* argv)
{
	if(argc == 3 && std::string(argv[1]) == "--write-scene")
		return writeScene(argv[2]) ? 0 : 1;
	if(argc < 3)
	{
		printf("MujocoPalletPlatform scene.xml output-prefix [steps=6000] [dt=.01] [iterations=100] [threads=8]\n"
			"MujocoPalletPlatform --write-scene scene.xml\n");
		return 1;
	}
	char error[1024];
	mjModel* model = mj_loadXML(argv[1], NULL, error, sizeof(error));
	if(!model)
		throw std::runtime_error(error);
	const int steps = argc > 3 ? atoi(argv[3]) : 6000;
	model->opt.timestep = argc > 4 ? atof(argv[4]) : 0.01;
	model->opt.iterations = argc > 5 ? atoi(argv[5]) : 100;
	const int threads = argc > 6 ? atoi(argv[6]) : 8;
	const std::vector<pallet::Body> bodies = platform::createStack();
	if(model->nmocap != 1 || model->nbody != int(bodies.size()) + 2 || model->nv != 6 * int(bodies.size()) || model->ngeom != int(bodies.size()) + 1)
	{
		fprintf(stderr, "The scene must hold the mocap platform and then %zu free bodies\n", bodies.size());
		return 1;
	}
	mjData* data = mj_makeData(model);
	mju_threadpool(data, threads > 1 ? threads : 0);
	platform::Recorder recorder(argv[2]);
	if(!recorder.isValid())
		return 1;
	printf("MuJoCo %s Newton: one pallet of %zu bodies on a platform shuttling %.3g m at up to %.3g m/s and %.3g m/s^2 (%.3g s strokes), %.6g s steps, cap %d, %d workers\n",
		mj_versionString(), bodies.size(), platform::stroke, platform::maximumSpeed, platform::acceleration, platform::strokeDuration(),
		model->opt.timestep, model->opt.iterations, threads);
	std::vector<pallet::Pose> poses(bodies.size());
	for(int step = 0; step < steps; ++step)
	{
		const double timestep = model->opt.timestep;
		const double next = platform::position(data->time + timestep);
		const double speed = (next - data->mocap_pos[0]) / timestep;
		const pallet::Clock::time_point start = pallet::Clock::now();
		mj_step1(model, data);
		mujocoConveyor::applyRestDistance(data);
		mujocoConveyor::applyConveyorVelocity(data, 1, speed);
		mj_fwdActuation(model, data);
		mj_fwdAcceleration(model, data);
		mj_fwdConstraint(model, data);
		mj_sensorAcc(model, data);
		mj_checkAcc(model, data);
		mj_Euler(model, data);
		data->mocap_pos[0] = next;
		const double stepMs = pallet::elapsed(start);
		int iterations = 0;
		for(int i = 0; i < std::min(data->nisland, mjNISLAND); ++i)
			iterations += data->solver_niter[i];
		for(size_t i = 0; i < bodies.size(); ++i)
		{
			mju_copy3(poses[i].position, data->qpos + 7 * i);
			mju_quat2Mat(poses[i].rotation, data->qpos + 7 * i + 3);
			mju_copy3(poses[i].velocity, data->qvel + 6 * i);
		}
		recorder.record(step + 1, data->time, stepMs, data->ncon, iterations, next, speed, bodies, poses);
	}
	for(int i = 0; i < mjNWARNING; ++i)
	{
		if(data->warning[i].number)
			printf("WARNING %d count %d\n", i, data->warning[i].number);
	}
	mj_deleteData(data);
	mj_deleteModel(model);
	return 0;
}
