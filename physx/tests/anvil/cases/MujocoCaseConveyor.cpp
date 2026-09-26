// Individual cases on ten long conveyors, simulated by MuJoCo's Newton solver.
#include "CaseScene.h"
#include "MujocoConveyor.h"
#include <cstdlib>
#include <stdexcept>

// Solver and contact settings match the pallet scene, including PhysX Anvil's default
// contact stiffening (regularization 1e-2 at first touch, 1e-4 beyond 20 um).
static bool writeScene(const char* path)
{
	FILE* file = cases::openOutput(path);
	if(!file)
		return false;
	fprintf(file, "<mujoco model=\"Case conveyors\">\n"
		"\t<compiler angle=\"radian\"/>\n\t<size memory=\"1G\"/>\n"
		"\t<option timestep=\"0.01\" gravity=\"0 -9.81 0\" integrator=\"Euler\" solver=\"Newton\"\n"
		"\t\tcone=\"pyramidal\" jacobian=\"sparse\" iterations=\"100\" tolerance=\"1e-8\" ls_tolerance=\"0.01\"/>\n"
		"\t<default><geom type=\"box\" condim=\"3\" friction=\"%.9g 0 0\" margin=\"0.001\" gap=\"0\"\n"
		"\t\tsolref=\"0.02 1\" solimp=\"0.990099 0.9999 0.00002 0.5 2\"/></default>\n"
		"\t<worldbody>\n\t\t<light pos=\"0 10 0\" dir=\"0 -1 0\" directional=\"true\"/>\n", cases::friction);
	for(int lane = 0; lane < cases::conveyorCount; ++lane)
		fprintf(file, "\t\t<geom name=\"belt%d\" pos=\"0 %.9g %.9g\" size=\"%.9g %.9g %.9g\" rgba=\"0.2 0.25 0.3 1\"/>\n",
			lane + 1, cases::conveyorTop - 0.5 * cases::beltThickness, cases::laneZ(lane),
			0.5 * cases::conveyorLength, 0.5 * cases::beltThickness, 0.5 * cases::conveyorWidth);
	for(int i = 0; i < cases::caseCount; ++i)
	{
		double position[3];
		cases::casePosition(i, position);
		fprintf(file, "\t\t<body name=\"case%d\" pos=\"%.9g %.9g %.9g\"><freejoint/>\n"
			"\t\t\t<geom size=\"%.9g %.9g %.9g\" mass=\"%.9g\" rgba=\"0.85 0.7 0.45 1\"/>\n\t\t</body>\n",
			i + 1, position[0], position[1], position[2], 0.5 * cases::caseSize, 0.5 * cases::caseSize,
			0.5 * cases::caseSize, cases::caseMass);
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
		printf("MujocoCaseConveyor scene.xml output-prefix [steps=2000] [dt=.01] [iterations=100] [threads=8] [belt-ramp=1]\n"
			"MujocoCaseConveyor --write-scene scene.xml\n");
		return 1;
	}
	char error[1024];
	mjModel* model = mj_loadXML(argv[1], NULL, error, sizeof(error));
	if(!model)
		throw std::runtime_error(error);
	const int steps = argc > 3 ? atoi(argv[3]) : 2000;
	model->opt.timestep = argc > 4 ? atof(argv[4]) : 0.01;
	model->opt.iterations = argc > 5 ? atoi(argv[5]) : 100;
	const int threads = argc > 6 ? atoi(argv[6]) : 8;
	// Seconds for the belts to reach full speed from rest; 0 starts them at full speed.
	const double ramp = argc > 7 ? atof(argv[7]) : cases::beltRamp;
	if(model->nbody != cases::caseCount + 1 || model->nv != 6 * cases::caseCount || model->ngeom != cases::conveyorCount + cases::caseCount)
	{
		fprintf(stderr, "The scene must hold %d free cases after %d belt geoms\n", cases::caseCount, cases::conveyorCount);
		return 1;
	}
	if(steps * model->opt.timestep > cases::maximumDuration())
	{
		printf("The cases reach the end of the belts after %.3g s\n", cases::maximumDuration());
		return 1;
	}
	mjData* data = mj_makeData(model);
	mju_threadpool(data, threads > 1 ? threads : 0);
	cases::Recorder recorder(argv[2]);
	if(!recorder.isValid())
		return 1;
	printf("MuJoCo %s Newton: %d cases on %d conveyors, %.6g s steps, cap %d, %d workers\n", mj_versionString(),
		cases::caseCount, cases::conveyorCount, model->opt.timestep, model->opt.iterations, threads);
	std::vector<cases::State> states(static_cast<size_t>(cases::caseCount));
	for(int step = 0; step < steps; ++step)
	{
		const cases::Clock::time_point start = cases::Clock::now();
		mj_step1(model, data);
		mujocoConveyor::applyRestDistance(data);
		mujocoConveyor::applyConveyorVelocity(data, cases::conveyorCount, cases::beltSpeedAt(data->time + model->opt.timestep, ramp));
		mj_fwdActuation(model, data);
		mj_fwdAcceleration(model, data);
		mj_fwdConstraint(model, data);
		mj_sensorAcc(model, data);
		mj_checkAcc(model, data);
		mj_Euler(model, data);
		const double stepMs = cases::elapsed(start);
		int iterations = 0;
		for(int i = 0; i < std::min(data->nisland, mjNISLAND); ++i)
			iterations = std::max(iterations, data->solver_niter[i]);
		for(int i = 0; i < cases::caseCount; ++i)
		{
			mju_copy3(states[size_t(i)].position, data->qpos + 7 * i);
			mju_copy3(states[size_t(i)].velocity, data->qvel + 6 * i);
		}
		recorder.record(step + 1, data->time, stepMs, data->ncon, data->nefc, iterations, states);
	}
	recorder.writeFinal(states);
	for(int i = 0; i < mjNWARNING; ++i)
	{
		if(data->warning[i].number)
			printf("WARNING %d count %d\n", i, data->warning[i].number);
	}
	mj_deleteData(data);
	mj_deleteModel(model);
	return 0;
}
