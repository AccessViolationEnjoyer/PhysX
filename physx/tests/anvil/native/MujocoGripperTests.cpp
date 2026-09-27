// The gripper of GripperTests.cpp, simulated by MuJoCo's Newton solver for comparison. The arm is
// a heavy body on a vertical slide joint whose position and velocity are set to the lift
// trajectory before every step (a mocap arm would move the fingers without a velocity, so their
// contacts would not see the lift). Each finger is a slide joint on the arm, closed by a position
// actuator with the PhysX drive's stiffness and a force limit equal to the grip. The drive's
// damping is joint damping, which the Euler integrator treats implicitly as PhysX drives do;
// the actuator's own damping would be explicit and unstable for these light fingers.
//
// usage: MujocoGripperTests scene-directory [grip-force=150] [hold-seconds=1] [pyramidal|elliptic] [boxes=4] [trace]
//   Contacts use the comparisons' shared solref 0.02 1 and constant solimp 0.9999 (regularization
//   1e-4), with contact rest distance and a 10 mm margin so fast fingers meet the boxes before
//   penetrating, as PhysX's contact offset does.
#include "MujocoConveyor.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const double timestep = 0.01;
static const double friction = 0.5;
static const double boxSize = 0.1;
static const double boxMass = 1.0;
static int boxCount = 4;
static const double fingerHalf[3] = { 0.01, 0.04, 0.06 };
static const double fingerMass = 0.5;
static const double opening = 0.03;
static const double armHeight = 0.3;
static const double lift = 0.3;
static const int closeStep = 100, liftStep = 200, liftSteps = 150;

static double armLift(int step)
{
	const double t = std::min(1.0, std::max(0.0, double(step - liftStep) / liftSteps));
	return lift * t * t * (3.0 - 2.0 * t);
}

static bool writeScene(const std::string& path, double grip, bool elliptic)
{
	FILE* file = fopen(path.c_str(), "w");
	if(!file)
		return false;
	fprintf(file, "<mujoco model=\"Gripper\">\n\t<option timestep=\"%g\" gravity=\"0 -9.81 0\" integrator=\"Euler\" solver=\"Newton\"\n"
		"\t\tcone=\"%s\" jacobian=\"sparse\" iterations=\"100\" tolerance=\"1e-8\" ls_tolerance=\"0.01\"/>\n"
		"\t<default><geom condim=\"3\" friction=\"%g 0 0\" margin=\"0.01\" solref=\"0.02 1\" solimp=\"0.9999 0.9999 0.001 0.5 2\"/></default>\n\t<worldbody>\n"
		"\t\t<geom name=\"ground\" type=\"plane\" size=\"5 5 0.1\" zaxis=\"0 1 0\"/>\n",
		timestep, elliptic ? "elliptic" : "pyramidal", friction);
	for(int i = 0; i < boxCount; ++i)
		fprintf(file, "\t\t<body name=\"box%d\" pos=\"%g %g 0\"><freejoint/><geom type=\"box\" size=\"%g %g %g\" mass=\"%g\"/></body>\n",
			i, (i - 0.5 * (boxCount - 1)) * boxSize, 0.5 * boxSize, 0.5 * boxSize, 0.5 * boxSize, 0.5 * boxSize, boxMass);
	const double fingerX = 0.5 * boxCount * boxSize + opening + fingerHalf[0];
	fprintf(file, "\t\t<body name=\"arm\" pos=\"0 %g 0\"><joint name=\"lift\" type=\"slide\" axis=\"0 1 0\"/>"
		"<inertial pos=\"0 0 0\" mass=\"1000\" diaginertia=\"1 1 1\"/>\n", armHeight);
	for(int side = 0; side < 2; ++side)
		fprintf(file, "\t\t\t<body name=\"finger%d\" pos=\"%g %g 0\"><joint name=\"grip%d\" type=\"slide\" axis=\"1 0 0\" damping=\"200\"/>"
			"<geom type=\"box\" size=\"%g %g %g\" mass=\"%g\"/></body>\n", side, (side ? 1.0 : -1.0) * fingerX, 0.5 * boxSize - armHeight,
			side, fingerHalf[0], fingerHalf[1], fingerHalf[2], fingerMass);
	fprintf(file, "\t\t</body>\n\t</worldbody>\n\t<actuator>\n");
	for(int side = 0; side < 2; ++side)
		fprintf(file, "\t\t<position joint=\"grip%d\" kp=\"5000\" forcelimited=\"true\" forcerange=\"%g %g\"/>\n", side, -grip, grip);
	fprintf(file, "\t</actuator>\n</mujoco>\n");
	fclose(file);
	return true;
}

int main(int argc, const char* const* argv)
{
	if(argc < 2)
	{
		printf("MujocoGripperTests scene-directory [grip-force=150] [hold-seconds=1] [pyramidal|elliptic] [boxes=4] [trace]\n");
		return 1;
	}
	const double grip = argc > 2 ? std::atof(argv[2]) : 150.0;
	const double holdSeconds = argc > 3 ? std::atof(argv[3]) : 1.0;
	const bool elliptic = argc > 4 && std::strcmp(argv[4], "elliptic") == 0;
	boxCount = argc > 5 ? std::atoi(argv[5]) : 4;
	const std::string path = std::string(argv[1]) + "/gripper.xml";
	if(!writeScene(path, grip, elliptic))
		return 1;
	char error[1024];
	mjModel* model = mj_loadXML(path.c_str(), NULL, error, sizeof(error));
	if(!model)
	{
		fprintf(stderr, "%s\n", error);
		return 1;
	}
	mjData* data = mj_makeData(model);
	// Bodies: world, boxes 1-4, arm 5, fingers 6-7. Degrees of freedom: four free joints, then the
	// arm's lift and the two fingers' grips.
	const int armBody = boxCount + 1, liftDof = 6 * boxCount, liftPosition = 7 * boxCount;
	const int endStep = liftStep + liftSteps + int(holdSeconds / timestep);
	double totalMs = 0.0, maximumSlip = 0.0, maximumTilt = 0.0, maximumOverlap = 0.0;
	std::vector<double> closedOffset(boxCount);
	for(int step = 0; step < endStep; ++step)
	{
		if(step == closeStep)
		{
			data->ctrl[0] = opening + 0.05;
			data->ctrl[1] = -(opening + 0.05);
		}
		// Hold the arm on its trajectory: this step's position and velocity.
		data->qpos[liftPosition] = armLift(step);
		data->qvel[liftDof] = (armLift(step + 1) - armLift(step)) / timestep;
		const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		mj_step1(model, data);
		mujocoConveyor::applyRestDistance(data);
		mj_step2(model, data);
		if(step > 0)
			totalMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		mj_kinematics(model, data);
		const double fingerY = 0.5 * (data->xpos[3 * (armBody + 1) + 1] + data->xpos[3 * (armBody + 2) + 1]);
		if(argc > 6 && step % 10 == 9 && step >= closeStep)
		{
			printf("step %3d fingers x %+.4f %+.4f", step + 1, double(data->xpos[3 * (armBody + 1)]), double(data->xpos[3 * (armBody + 2)]));
			for(int i = 0; i < boxCount; ++i)
				printf("  box%d x %+.4f y %.4f", i, double(data->xpos[3 * (i + 1)]), double(data->xpos[3 * (i + 1) + 1]));
			printf("  contacts %d\n", data->ncon);
		}
		for(int i = 0; i < boxCount; ++i)
		{
			const double boxY = data->xpos[3 * (i + 1) + 1];
			if(step + 1 == liftStep)
				closedOffset[i] = boxY - fingerY;
			if(step + 1 >= liftStep)
				maximumSlip = std::max(maximumSlip, std::abs(boxY - fingerY - closedOffset[i]));
			const double w = std::min(1.0, std::abs(double(data->xquat[4 * (i + 1)])));
			maximumTilt = std::max(maximumTilt, 2.0 * std::acos(w));
			if(i + 1 < boxCount)
				maximumOverlap = std::max(maximumOverlap, boxSize - (data->xpos[3 * (i + 2)] - data->xpos[3 * (i + 1)]));
		}
	}
	// The heavy arm gains one step of gravity after its last trajectory update; restore it.
	data->qvel[liftDof] = 0.0;
	mj_forward(model, data);
	double minimumLift = 1.0e30, speed = 0.0, creep = 0.0;
	mjtNum fingerVelocity[2][6];
	mj_objectVelocity(model, data, mjOBJ_BODY, armBody + 1, fingerVelocity[0], 0);
	mj_objectVelocity(model, data, mjOBJ_BODY, armBody + 2, fingerVelocity[1], 0);
	const double fingerSpeed = 0.5 * (fingerVelocity[0][4] + fingerVelocity[1][4]);
	for(int i = 0; i < boxCount; ++i)
	{
		minimumLift = std::min(minimumLift, double(data->xpos[3 * (i + 1) + 1]) - 0.5 * boxSize);
		speed = std::max(speed, double(mju_norm3(data->qvel + 6 * i)));
		creep = std::max(creep, std::abs(double(data->qvel[6 * i + 1]) - fingerSpeed));
	}
	const double fingerGap = data->xpos[3 * (armBody + 2)] - data->xpos[3 * (armBody + 1)] - 2.0 * fingerHalf[0];
	printf("gripper MuJoCo-%s %s boxes=%d grip_n=%.1f mean_step_us=%.2f hold_s=%.1f creep_mm_s=%.4f minimum_lift_mm=%.3f maximum_slip_mm=%.4f"
		" maximum_tilt_deg=%.4f row_width_mm=%.4f maximum_box_overlap_mm=%.4f final_speed_mm_s=%.4f\n",
		elliptic ? "elliptic" : "pyramidal", mj_versionString(), boxCount, grip, 1e3 * totalMs / (endStep - 1), holdSeconds, creep * 1e3,
		minimumLift * 1e3, maximumSlip * 1e3, maximumTilt * 180.0 / mjPI, fingerGap * 1e3, maximumOverlap * 1e3, speed * 1e3);
	for(int i = 0; i < mjNWARNING; ++i)
	{
		if(data->warning[i].number)
			printf("  WARNING %d count %d\n", i, data->warning[i].number);
	}
	mj_deleteData(data);
	mj_deleteModel(model);
	return 0;
}
