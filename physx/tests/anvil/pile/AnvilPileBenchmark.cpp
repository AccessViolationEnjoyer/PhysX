#include "MujocoAdapter.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static FILE* openOutput(const char* path)
{
	FILE* file = fopen(path, "w");
	if(!file)
		throw std::runtime_error("Cannot open output file");
	return file;
}

static void writeScene(const char* path, int width, int depth, int layers)
{
	FILE* file = openOutput(path);
	fprintf(file, "<mujoco model=\"Connected box pile\">\n"
		"\t<compiler angle=\"radian\"/>\n\t<size memory=\"2G\"/>\n"
		"\t<option timestep=\"0.01\" gravity=\"0 -9.81 0\" integrator=\"Euler\" solver=\"Newton\"\n"
		"\t\tcone=\"pyramidal\" jacobian=\"sparse\" iterations=\"100\" tolerance=\"1e-8\" ls_tolerance=\"0.01\"/>\n"
		"\t<default><geom type=\"box\" condim=\"3\" friction=\"0.5 0 0\" margin=\"0.001\" gap=\"0\"\n"
		"\t\tsolref=\"0.02 1\" solimp=\"0.9999 0.9999 0.001 0.5 2\"/></default>\n"
		"\t<worldbody>\n\t\t<geom name=\"floor\" type=\"plane\" size=\"10 10 0.1\" euler=\"-1.5707963267948966 0 0\"/>\n");
	for(int layer = 0; layer < layers; ++layer)
	{
		// Quarter-box staggering bridges adjacent columns through real box contacts.
		// A shared static floor alone would not join separate stacks into an island.
		const double shift = (layer % 2) * 0.075;
		for(int z = 0; z < depth; ++z)
			for(int x = 0; x < width; ++x)
			{
				const int index = (layer * depth + z) * width + x;
				fprintf(file, "\t\t<body name=\"box%d\" pos=\"%.12g %.12g %.12g\"><freejoint/>"
					"<geom size=\"0.15 0.15 0.15\" mass=\"1\"/></body>\n", index,
					(x - (width - 1) * 0.5) * 0.3 + shift, 0.15 + layer * 0.3,
					(z - (depth - 1) * 0.5) * 0.3 + shift);
			}
	}
	fprintf(file, "\t</worldbody>\n</mujoco>\n");
	fclose(file);
}

static void applyRestDistance(mjData* data)
{
	// Match the existing pallet comparison: keep the detection margin positive,
	// while the intended resting surfaces touch. Apply this to both solvers.
	for(int row = 0; row < data->nefc; ++row)
		data->efc_aref[row] -= data->efc_KBIP[4 * row] * data->efc_KBIP[4 * row + 2] * data->efc_margin[row];
}

static int nativeIterations(const mjData* data)
{
	int iterations = 0;
	for(int island = 0; island < std::min(data->nisland, mjNISLAND); ++island)
		iterations = std::max(iterations, data->solver_niter[island]);
	return iterations;
}

struct SnapshotResult
{
	double solveMs = 0.0;
	int iterations = 0;
	int activeRows = 0;
	long long nativeSolverNnz = -1;
	std::vector<mjtNum> acceleration;
};

static void benchmarkSnapshot(const mjModel* model, mjData* data,
	const std::string& prefix, int sourceStep, int threads, int repeats)
{
	const std::vector<mjtNum> warmStart(data->qacc_warmstart, data->qacc_warmstart + model->nv);
	const std::vector<mjtNum> position(data->qpos, data->qpos + model->nq);
	const std::vector<mjtNum> velocity(data->qvel, data->qvel + model->nv);
	const mjtNum sourceTime = data->time;
	SnapshotResult result[2];
	for(int method = 0; method < 2; ++method)
		result[method].acceleration.resize(model->nv);
	int largestIsland = 0;
	for(int island = 0; island < data->nisland; ++island)
		largestIsland = std::max(largestIsland, data->island_nv[island]);
	FILE* output = openOutput((prefix + "-snapshot.csv").c_str());
	fprintf(output, "method,repeat,solve_ms,iterations,islands,largest_bodies,rows,active_rows,max_linear_vel_diff,max_angular_vel_diff,native_solver_nnz\n");
	FILE* settings = openOutput((prefix + "-snapshot-settings.csv").c_str());
	fprintf(settings, "setting,value\n"
		"scope,repeated fixed host equations including prototype translation and writeback\n"
		"source_trajectory,prototype\nsource_step,%d\nsource_time,%.17g\n"
		"integrated_warmup_steps,%d\nrepeats,%d\nworkers,%d\nbodies,%d\n"
		"timestep,%.17g\niterations,%d\ntolerance,%.17g\n"
		"native_line_tolerance,%.17g\nnative_line_iterations,%d\n"
		"prototype_line_tolerance,%.17g\nnative_solver,%d\ncone,%d\njacobian,%d\n"
		"integrator,%d\nnoslip_iterations,%d\ndisable_flags,%d\nenable_flags,%d\n"
		"order,prototype first on odd repeats and mujoco first on even repeats\n"
		"warm_start,fixed captured qacc_warmstart restored before every call\n"
		"integration,none during snapshot repetitions\n"
		"detailed_phase_profiling,disabled during snapshot repetitions\n",
		sourceStep, sourceTime, sourceStep - 1, repeats, threads, int(model->nv / 6),
		model->opt.timestep, model->opt.iterations, model->opt.tolerance,
		model->opt.ls_tolerance, model->opt.ls_iterations, anvil::Settings().lineTolerance,
		model->opt.solver, model->opt.cone, model->opt.jacobian, model->opt.integrator,
		model->opt.noslip_iterations, model->opt.disableflags, model->opt.enableflags);
	fclose(settings);
	printf("Snapshot: source step %d at time %.9g from prototype trajectory; %d alternating pairs on fixed equations and warm start, no integration\n",
		sourceStep, sourceTime, repeats);
	fflush(stdout);
	for(int repeat = 0; repeat < repeats; ++repeat)
	{
		for(int order = 0; order < 2; ++order)
		{
			const int method = (repeat + order) % 2;
			SnapshotResult& current = result[method];
			// Both entry points reconstruct their acceleration and forces from these
			// unchanged host equations. Restoring the seed is outside the timer.
			mju_copy(data->qacc_warmstart, warmStart.data(), int(model->nv));
			anvil::MujocoSolverProfile profile;
			const anvil::Clock::time_point start = anvil::Clock::now();
			if(method == 0)
				current.iterations = anvil::solveMujocoConstraints(model, data, profile);
			else
				mj_fwdConstraint(model, data);
			current.solveMs = anvil::elapsed(start);
			if(method == 1)
			{
				current.iterations = nativeIterations(data);
				current.nativeSolverNnz = 0;
				for(int island = 0; island < std::min(data->nisland, mjNISLAND); ++island)
					current.nativeSolverNnz += data->solver_nnz[island];
			}
			if(std::memcmp(data->qacc_warmstart, warmStart.data(), sizeof(mjtNum) * model->nv) != 0 ||
				std::memcmp(data->qpos, position.data(), sizeof(mjtNum) * model->nq) != 0 ||
				std::memcmp(data->qvel, velocity.data(), sizeof(mjtNum) * model->nv) != 0 || data->time != sourceTime)
				throw std::runtime_error("Snapshot solve changed the fixed state or warm start");
			mju_copy(current.acceleration.data(), data->qacc, int(model->nv));
			current.activeRows = 0;
			for(int row = 0; row < data->nefc; ++row)
			{
				if(!std::isfinite(data->efc_force[row]))
					throw std::runtime_error("Nonfinite snapshot contact force");
				current.activeRows += data->efc_force[row] > 0.0;
			}
		}
		double linear = 0.0, angular = 0.0;
		for(int dof = 0; dof < model->nv; ++dof)
		{
			if(!std::isfinite(result[0].acceleration[dof]) || !std::isfinite(result[1].acceleration[dof]))
				throw std::runtime_error("Nonfinite snapshot acceleration");
			// Same initial velocity and timestep: compare the velocity increments
			// without integrating either result into the captured physical state.
			const double difference = model->opt.timestep * std::abs(result[0].acceleration[dof] - result[1].acceleration[dof]);
			if(dof % 6 < 3)
				linear = std::max(linear, difference);
			else
				angular = std::max(angular, difference);
		}
		for(int order = 0; order < 2; ++order)
		{
			const int method = (repeat + order) % 2;
			const SnapshotResult& current = result[method];
			fprintf(output, "%s,%d,%.9g,%d,%d,%d,%d,%d,%.12g,%.12g,%lld\n",
				method == 0 ? "prototype" : "mujoco", repeat + 1, current.solveMs, current.iterations,
				data->nisland, largestIsland / 6, data->nefc, current.activeRows, linear, angular, current.nativeSolverNnz);
		}
		fflush(output);
	}
	fclose(output);
}

int main(int argc, const char* const* argv)
{
	if(argc == 6 && std::strcmp(argv[1], "--write-scene") == 0)
	{
		const int width = std::atoi(argv[3]), depth = std::atoi(argv[4]), layers = std::atoi(argv[5]);
		if(width < 1 || depth < 1 || layers < 1)
			return 1;
		writeScene(argv[2], width, depth, layers);
		return 0;
	}
	if(argc < 4 || (std::strcmp(argv[3], "prototype") != 0 && std::strcmp(argv[3], "mujoco") != 0))
	{
		printf("AnvilPileBenchmark --write-scene output.xml width depth layers\n"
			"AnvilPileBenchmark scene.xml output-prefix prototype|mujoco [steps=1000] [threads=8] [auditEvery=0] [profile=0] [snapshotRepeats=0]\n");
		return 1;
	}
	const int steps = argc > 4 ? std::atoi(argv[4]) : 1000;
	const int threads = argc > 5 ? std::atoi(argv[5]) : 8;
	const int auditEvery = argc > 6 ? std::atoi(argv[6]) : 0;
	const bool profiling = argc > 7 && std::atoi(argv[7]) != 0;
	const int snapshotRepeats = argc > 8 ? std::atoi(argv[8]) : 0;
	if(snapshotRepeats < 0 || (snapshotRepeats > 0 && steps < 1))
		throw std::runtime_error("Snapshot repeats must be nonnegative and capture requires at least one step");
	// Snapshot input always comes from our ordinary live trajectory.
	const bool prototype = snapshotRepeats > 0 || std::strcmp(argv[3], "prototype") == 0;
	char error[1024];
	mjModel* model = mj_loadXML(argv[1], NULL, error, sizeof(error));
	if(!model)
		throw std::runtime_error(error);
	mjData* data = mj_makeData(model);
	mju_threadpool(data, threads > 1 ? threads : 0);
	mj_step1(model, data);
	anvil::validateMujocoModel(model, data, model->nv / 6);
	const std::string prefix(argv[2]);
	FILE* output = openOutput((prefix + "-steps.csv").c_str());
	fprintf(output, "step,time,solve_ms,step_ms,islands,largest_island_bodies,unconstrained_bodies,contacts,rows,active_rows,iterations,stationarity,projection_error,penetration_m,max_linear_speed,minimum_height,maximum_height,global_preparation_ms,island_tasks_ms,scatter_ms,worker_prepare_ms,worker_solve_ms,matrix_ms,factor_ms,update_ms,evaluation_ms,line_ms,backsolve_ms\n");
	FILE* audit = auditEvery && prototype ? openOutput((prefix + "-audit.csv").c_str()) : NULL;
	if(audit)
		fprintf(audit, "step,prototype_iterations,native_iterations,max_linear_velocity_difference,max_angular_velocity_difference\n");
	std::vector<mjtNum> rowAcceleration, savedAcceleration, savedForce;
	printf("%s: %d boxes, %d workers, dt %.6g, cap %d, tolerance %.6g; full solve includes preparation and writeback\n",
		prototype ? "Prototype" : "MuJoCo", int(model->nv / 6), threads, model->opt.timestep, model->opt.iterations, model->opt.tolerance);
	for(int step = 0; step < steps; ++step)
	{
		const anvil::Clock::time_point stepStart = anvil::Clock::now();
		mj_step1(model, data);
		applyRestDistance(data);
		mj_fwdActuation(model, data);
		mj_fwdAcceleration(model, data);
		if(snapshotRepeats > 0 && step + 1 == steps)
		{
			fflush(output);
			benchmarkSnapshot(model, data, prefix, step + 1, threads, snapshotRepeats);
			break;
		}
		anvil::MujocoSolverProfile profile;
		profile.enabled = profiling;
		const anvil::Clock::time_point solveStart = anvil::Clock::now();
		int iterations;
		if(prototype)
			iterations = anvil::solveMujocoConstraints(model, data, profile);
		else
		{
			mj_fwdConstraint(model, data);
			iterations = nativeIterations(data);
		}
		const double solveMs = anvil::elapsed(solveStart);
		const anvil::Clock::time_point diagnosticStart = anvil::Clock::now();
		if(audit && step % auditEvery == 0)
		{
			// Compare identical equations and warm starts, then restore our result
			// before integration. Reference solves are outside both timing scopes.
			savedAcceleration.assign(data->qacc, data->qacc + model->nv);
			savedForce.assign(data->efc_force, data->efc_force + data->nefc);
			mj_fwdConstraint(model, data);
			double linear = 0.0, angular = 0.0;
			for(int dof = 0; dof < model->nv; ++dof)
			{
				const double difference = model->opt.timestep * std::abs(savedAcceleration[dof] - data->qacc[dof]);
				if(dof % 6 < 3)
					linear = std::max(linear, difference);
				else
					angular = std::max(angular, difference);
			}
			fprintf(audit, "%d,%d,%d,%.12g,%.12g\n", step + 1, iterations, nativeIterations(data), linear, angular);
			mju_copy(data->qacc, savedAcceleration.data(), int(model->nv));
			mju_copy(data->efc_force, savedForce.data(), data->nefc);
			mj_mulJacTVec(model, data, data->qfrc_constraint, data->efc_force);
		}
		int largestIsland = 0, activeDofs = 0, activeRows = 0;
		for(int island = 0; island < data->nisland; ++island)
		{
			largestIsland = std::max(largestIsland, data->island_nv[island]);
			activeDofs += data->island_nv[island];
		}
		double stationarity = 0.0, projectionError = 0.0, penetration = 0.0;
		// Independent primal stationarity and nonnegative-edge force consistency.
		// These diagnostics use the physical mass and unscaled host Jacobian.
		for(int dof = 0; dof < model->nv; ++dof)
		{
			const double rootMass = std::sqrt(anvil::mujocoMassDiagonal(model, data, dof));
			const double residual = model->opt.timestep * (rootMass * (data->qacc[dof] - data->qacc_smooth[dof]) -
				data->qfrc_constraint[dof] / rootMass);
			if(!std::isfinite(residual))
				throw std::runtime_error("Nonfinite constraint result");
			stationarity = std::max(stationarity, std::abs(residual));
		}
		rowAcceleration.resize(data->nefc);
		mj_mulJacVec(model, data, rowAcceleration.data(), data->qacc);
		for(int row = 0; row < data->nefc; ++row)
		{
			if(!std::isfinite(rowAcceleration[row]) || !std::isfinite(data->efc_force[row]))
				throw std::runtime_error("Nonfinite contact result");
			const double expected = std::max(0.0, -(rowAcceleration[row] - data->efc_aref[row]) / data->efc_R[row]);
			projectionError = std::max(projectionError, model->opt.timestep * std::abs(data->efc_force[row] - expected));
			activeRows += data->efc_force[row] > 0.0;
		}
		for(int contact = 0; contact < data->ncon; ++contact)
			penetration = std::max(penetration, -data->contact[contact].dist);
		const double diagnosticMs = anvil::elapsed(diagnosticStart);
		mj_sensorAcc(model, data);
		mj_checkAcc(model, data);
		mj_Euler(model, data);
		const double stepMs = anvil::elapsed(stepStart) - diagnosticMs;
		double minimumHeight = 1.0e30, maximumHeight = -1.0e30, speed = 0.0;
		for(int body = 0; body < model->nv / 6; ++body)
		{
			minimumHeight = std::min(minimumHeight, double(data->qpos[body * 7 + 1]));
			maximumHeight = std::max(maximumHeight, double(data->qpos[body * 7 + 1]));
			speed = std::max(speed, std::sqrt(mju_dot3(data->qvel + body * 6, data->qvel + body * 6)));
		}
		fprintf(output, "%d,%.9g,%.9g,%.9g,%d,%d,%d,%d,%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
			step + 1, data->time, solveMs, stepMs, data->nisland, largestIsland / 6, int((model->nv - activeDofs) / 6),
			data->ncon, data->nefc, activeRows, iterations, stationarity, projectionError, penetration, speed,
			minimumHeight, maximumHeight, profile.globalPreparationWallMs, profile.islandTasksWallMs,
			profile.scatterWallMs, profile.preparationMs, profile.solveMs, profile.matrixMs, profile.factorMs,
			profile.updateMs, profile.evaluationMs, profile.lineSearchMs, profile.backsolveMs);
		if(step == 0 || (step + 1) % 10 == 0)
			fflush(output);
	}
	for(int warning = 0; warning < mjNWARNING; ++warning)
		if(data->warning[warning].number)
			printf("WARNING %d count %d\n", warning, data->warning[warning].number);
	fclose(output);
	if(audit)
		fclose(audit);
	mj_deleteData(data);
	mj_deleteModel(model);
	return 0;
}
