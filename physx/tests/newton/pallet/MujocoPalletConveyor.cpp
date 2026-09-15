#include "PalletScene.h"
#include "MujocoAdapter.h"
#include <cstdint>

static void writeScene(const char* path, const std::vector<pallet::Body>& bodies)
{
	FILE* file = pallet::openOutput(path);
	fprintf(file, "<mujoco model=\"Pallet conveyors\">\n"
		"\t<compiler angle=\"radian\"/>\n\t<size memory=\"256M\"/>\n"
		"\t<option timestep=\"0.01\" gravity=\"0 -9.81 0\" integrator=\"Euler\" solver=\"Newton\"\n"
		"\t\tcone=\"pyramidal\" jacobian=\"sparse\" iterations=\"100\" tolerance=\"1e-8\" ls_tolerance=\"0.01\"/>\n"
		"\t<default><geom type=\"box\" condim=\"3\" friction=\"0.5 0 0\" margin=\"0.001\" gap=\"0\"\n"
		"\t\tsolref=\"0.02 1\" solimp=\"0.9999 0.9999 0.001 0.5 2\"/></default>\n"
		"\t<visual><global offwidth=\"1280\" offheight=\"720\"/><headlight ambient=\"0.4 0.4 0.4\"/></visual>\n"
		"\t<worldbody>\n\t\t<light pos=\"0 10 0\" dir=\"0 -1 0\" directional=\"true\"/>\n");
	for(int lane = 0; lane < pallet::conveyorCount; ++lane)
		fprintf(file, "\t\t<geom name=\"belt%d\" pos=\"0 0.4 %.9g\" size=\"%.9g 0.1 %.9g\" rgba=\"0.2 0.25 0.3 1\"/>\n",
			lane + 1, (lane - 2) * pallet::conveyorPitch, pallet::conveyorLength * 0.5, pallet::conveyorWidth * 0.5);
	for(size_t i = 0; i < bodies.size(); ++i)
	{
		const pallet::Body& body = bodies[i];
		const int boxIndex = int(i) - body.lane * pallet::bodiesPerPallet - 1 - body.layer * (pallet::boxesPerLayer + 1);
		const bool alternate = (boxIndex % 4 + boxIndex / 4) % 2 != 0;
		const char* color = body.kind == pallet::PALLET ? "0.5 0.3 0.12 1" :
			body.kind == pallet::SHEET ? "0.1 0.7 0.85 1" : alternate ? "0.7 0.5 0.3 1" : "0.85 0.7 0.45 1";
		fprintf(file, "\t\t<body name=\"%s\" pos=\"%.9g %.9g %.9g\"><freejoint/>\n"
			"\t\t\t<geom size=\"%.9g %.9g %.9g\" mass=\"%.9g\" rgba=\"%s\"/>\n\t\t</body>\n",
			body.name.c_str(), body.position[0], body.position[1], body.position[2],
			body.halfSize[0], body.halfSize[1], body.halfSize[2], body.mass, color);
	}
	fprintf(file, "\t</worldbody>\n</mujoco>\n");
	fclose(file);
}

// Keep MuJoCo's positive contact-detection margin separate from rest distance.
// As in PhysX, the intended resting surfaces touch; the collision envelope
// must not create a visible gap between each sheet and its adjacent boxes.
static void applyRestDistance(mjData* data)
{
	for(int row = 0; row < data->nefc; ++row)
		data->efc_aref[row] -= data->efc_KBIP[4 * row] * data->efc_KBIP[4 * row + 2] * data->efc_margin[row];
}

// MuJoCo has no contact target-velocity setter. Supply the belt's prescribed
// surface velocity in its acceleration reference, before either Newton solve.
// Pyramidal rows are normal +/- mu*tangent, in geom[1]-geom[0] order.
static void applyConveyorVelocity(mjData* data)
{
	for(int i = 0; i < data->ncon; ++i)
	{
		const mjContact& contact = data->contact[i];
		const bool beltFirst = contact.geom[0] < pallet::conveyorCount;
		const bool beltSecond = contact.geom[1] < pallet::conveyorCount;
		if(contact.efc_address < 0 || (!beltFirst && !beltSecond))
			continue;
		const double speed = beltFirst ? pallet::beltSpeed : -pallet::beltSpeed;
		for(int tangent = 0; tangent < 2; ++tangent)
		{
			for(int sign = 0; sign < 2; ++sign)
			{
				const int row = contact.efc_address + tangent * 2 + sign;
				const double target = speed * (contact.frame[0] + (sign ? -1.0 : 1.0) *
					contact.friction[tangent] * contact.frame[(tangent + 1) * 3]);
				data->efc_vel[row] -= target;
				data->efc_aref[row] += data->efc_KBIP[4 * row + 1] * target;
			}
		}
	}
}

static int countContactPairs(const mjData* data)
{
	static thread_local std::vector<uint64_t> pairs;
	pairs.clear();
	newton::reserveStorage(pairs, size_t(data->ncon));
	for(int i = 0; i < data->ncon; ++i)
	{
		const mjContact& contact = data->contact[i];
		pairs.push_back((uint64_t(contact.geom[0]) << 32) | uint64_t(contact.geom[1]));
	}
	std::sort(pairs.begin(), pairs.end());
	return int(std::unique(pairs.begin(), pairs.end()) - pairs.begin());
}

int main(int argc, const char* const* argv)
{
	const std::vector<pallet::Body> bodies = pallet::createBodies();
	if(argc == 3 && std::string(argv[1]) == "--write-scene")
	{
		writeScene(argv[2], bodies);
		return 0;
	}
	if(argc < 4)
	{
		printf("MujocoPalletConveyor scene.xml output-prefix mujoco|prototype [steps=2000] [dt=.01] [iterations=100] [threads=8] [impedance=0] [audit=0] [profile=0]\n");
		return 1;
	}
	char error[1024];
	mjModel* model = mj_loadXML(argv[1], NULL, error, sizeof(error));
	if(!model)
		throw std::runtime_error(error);
	const bool prototype = std::string(argv[3]) == "prototype";
	const int steps = argc > 4 ? atoi(argv[4]) : 2000;
	model->opt.timestep = argc > 5 ? atof(argv[5]) : 0.01;
	model->opt.iterations = argc > 6 ? atoi(argv[6]) : 100;
	const int threads = argc > 7 ? atoi(argv[7]) : 8;
	const double impedance = argc > 8 ? atof(argv[8]) : 0.0;
	const bool audit = argc > 9 && atoi(argv[9]) != 0;
	if(impedance > 0.0)
	{
		for(int i = 0; i < model->ngeom; ++i)
			model->geom_solimp[i * mjNIMP] = model->geom_solimp[i * mjNIMP + 1] = impedance;
	}
	mjData* data = mj_makeData(model);
	mjThreadPool* pool = threads > 1 ? mju_threadPoolCreate(threads) : NULL;
	if(pool)
		mju_bindThreadPool(data, pool);
	mj_step1(model, data);
	newton::validateMujocoModel(model, data, bodies.size());
	pallet::Recorder recorder(argv[2], bodies);
	std::vector<pallet::Pose> poses(bodies.size());
	std::vector<double> auditAcceleration, auditForce;
	double maximumVelocityError = 0.0;
	double maximumConvergedError = 0.0;
	int cappedAuditSteps = 0;
	FILE* timingFile = pallet::openOutput(std::string(argv[2]) + "-timing.csv");
	fprintf(timingFile, "step,islands,preparation_wall_ms,solver_only_wall_ms,scatter_wall_ms,adapter_wall_ms,global_preparation_wall_ms,island_tasks_wall_ms\n");
	FILE* profileFile = argc > 10 && atoi(argv[10]) ? pallet::openOutput(std::string(argv[2]) + "-profile.csv") : NULL;
	if(profileFile)
		fprintf(profileFile, "step,preparation_ms,solve_ms,matrix_ms,factor_ms,update_ms,evaluation_ms,line_ms,backsolve_ms,updates,symbolic_ms,symbolic_analyses,preparation_wall_ms,solver_only_wall_ms,scatter_wall_ms,global_preparation_wall_ms,island_tasks_wall_ms\n");
	FILE* auditFile = audit ? pallet::openOutput(std::string(argv[2]) + "-audit.csv") : NULL;
	if(auditFile)
		fprintf(auditFile, "step,prototype_iterations,mujoco_iterations,velocity_error\n");
	printf("MEASUREMENT solve_ms is the complete solver stage: Newton adapter preparation, fused island tasks and scatter, or native mj_fwdConstraint. Isolated Newton preparation/solve wall times are unavailable (-1); island_tasks_wall_ms includes dispatch and joins.\n");
	printf("%s: %zu bodies, dt %.6g s, cap %d, %d workers, impedance %.6g; warm start on\n",
		prototype ? "Prototype Newton" : "MuJoCo Newton", bodies.size(), model->opt.timestep,
		model->opt.iterations, threads, double(model->geom_solimp[0]));
	for(int step = 0; step < steps; ++step)
	{
		const pallet::Clock::time_point start = pallet::Clock::now();
		mj_step1(model, data);
		applyRestDistance(data);
		applyConveyorVelocity(data);
		mj_fwdActuation(model, data);
		mj_fwdAcceleration(model, data);
		const pallet::Clock::time_point solveStart = pallet::Clock::now();
		int iterations = 0;
		newton::MujocoSolverProfile profile;
		profile.enabled = profileFile != NULL;
		if(prototype)
			iterations = newton::solveMujocoConstraints(model, data, pool, profile);
		else
		{
			mj_fwdConstraint(model, data);
			for(int i = 0; i < std::min(data->nisland, mjNISLAND); ++i)
				iterations = std::max(iterations, data->solver_niter[i]);
		}
		const double adapterMs = pallet::elapsed(solveStart);
		const double solveMs = adapterMs;
		double auditMs = 0.0;
		if(audit && prototype)
		{
			const pallet::Clock::time_point auditStart = pallet::Clock::now();
			auditAcceleration.assign(data->qacc, data->qacc + model->nv);
			auditForce.assign(data->efc_force, data->efc_force + data->nefc);
			mj_fwdConstraint(model, data);
			int nativeIterations = 0;
			for(int i = 0; i < std::min(data->nisland, mjNISLAND); ++i)
				nativeIterations = std::max(nativeIterations, data->solver_niter[i]);
			double velocityError = 0.0;
			for(int i = 0; i < model->nv; ++i)
				velocityError = std::max(velocityError, model->opt.timestep * std::abs(auditAcceleration[i] - data->qacc[i]));
			maximumVelocityError = std::max(maximumVelocityError, velocityError);
			fprintf(auditFile, "%d,%d,%d,%.12g\n", step + 1, iterations, nativeIterations, velocityError);
			newton::validateMujocoModel(model, data, bodies.size());
			if(iterations == model->opt.iterations || nativeIterations == model->opt.iterations)
				++cappedAuditSteps;
			else
				maximumConvergedError = std::max(maximumConvergedError, velocityError);
			mju_copy(data->qacc, auditAcceleration.data(), model->nv);
			mju_copy(data->efc_force, auditForce.data(), data->nefc);
			mj_mulJacTVec(model, data, data->qfrc_constraint, data->efc_force);
			auditMs = pallet::elapsed(auditStart);
		}
		mj_sensorAcc(model, data);
		mj_checkAcc(model, data);
		mj_Euler(model, data);
		const double stepMs = pallet::elapsed(start) - auditMs;
		fprintf(timingFile, "%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", step + 1, data->nisland,
			profile.preparationWallMs, profile.solverOnlyMs, profile.scatterWallMs, adapterMs,
			profile.globalPreparationWallMs, profile.islandTasksWallMs);
		if(profileFile)
			fprintf(profileFile, "%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%.9g,%d,%.9g,%.9g,%.9g,%.9g,%.9g\n", step + 1,
				profile.preparationMs, profile.solveMs, profile.matrixMs, profile.factorMs, profile.updateMs,
				profile.evaluationMs, profile.lineSearchMs, profile.backsolveMs, profile.updates, profile.symbolicMs, profile.symbolicAnalyses, profile.preparationWallMs, profile.solverOnlyMs, profile.scatterWallMs, profile.globalPreparationWallMs, profile.islandTasksWallMs);
		for(size_t i = 0; i < poses.size(); ++i)
		{
			mju_copy3(poses[i].position, data->qpos + i * 7);
			mju_copy3(poses[i].velocity, data->qvel + i * 6);
			mju_quat2Mat(poses[i].rotation, data->qpos + i * 7 + 3);
		}
		recorder.record(step + 1, data->time, stepMs, solveMs, countContactPairs(data), data->ncon, data->nefc, iterations, bodies, poses);
	}
	if(audit)
		printf("AUDIT maximum next-velocity difference %.12g, uncapped %.12g, capped steps %d (linear m/s or angular rad/s)\n",
			maximumVelocityError, maximumConvergedError, cappedAuditSteps);
	for(int i = 0; i < mjNWARNING; ++i)
	{
		if(data->warning[i].number)
			printf("WARNING %d count %d\n", i, data->warning[i].number);
	}
	fclose(timingFile);
	if(profileFile)
		fclose(profileFile);
	if(auditFile)
		fclose(auditFile);
	if(pool)
		mju_threadPoolDestroy(pool);
	mj_deleteData(data);
	mj_deleteModel(model);
	return 0;
}
