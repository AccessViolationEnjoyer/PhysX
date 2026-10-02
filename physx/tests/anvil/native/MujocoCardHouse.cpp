// The card_house test of BehaviourTests.cpp (PEEL's CardHouse) in MuJoCo's Newton solver: five levels
// of 10 g cards, 0.2 x 0.4 m and 2 mm thick, in pairs leaning 25 degrees against each other, with
// flat cards bridging neighbouring pairs, 40 cards in all, friction 0.8. The house settles for a
// second and should then stand still; the checks and their limits are the Anvil test's. Solver and
// contact settings match the other MuJoCo comparison scenes.
//
// usage: MujocoCardHouse scene.xml [threads=1]
#include "MujocoConveyor.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const double cardWidth = 0.1, cardHeight = 0.2, cardThickness = 0.001;
static const double leanAngle = 25.0 * mjPI / 180.0;

static int failures = 0;

static void check(bool condition, const char* message)
{
	if(!condition)
	{
		std::printf("FAIL %s\n", message);
		++failures;
	}
}

// PEEL's construction, level by level, as BehaviourTests.cpp builds it; rotations are about x.
static int writeScene(const std::string& path)
{
	FILE* file = std::fopen(path.c_str(), "w");
	if(!file)
		return 0;
	std::fprintf(file, "<mujoco model=\"Card house\">\n"
		"\t<option timestep=\"0.01\" gravity=\"0 -9.81 0\" integrator=\"Euler\" solver=\"Newton\"\n"
		"\t\tcone=\"pyramidal\" jacobian=\"sparse\" iterations=\"100\" tolerance=\"1e-8\" ls_tolerance=\"0.01\"/>\n"
		"\t<default><geom type=\"box\" condim=\"3\" friction=\"0.8 0 0\" margin=\"0.001\" gap=\"0\"\n"
		"\t\tsolref=\"0.02 1\" solimp=\"0.9999 0.9999 0.001 0.5 2\"/></default>\n"
		"\t<worldbody>\n\t\t<geom name=\"floor\" pos=\"0 -0.5 0\" size=\"2 0.5 2\"/>\n");
	const double angles[3] = { 0.5 * mjPI, leanAngle, -leanAngle };
	int cards = 0;
	int pairs = 5;
	double z0 = 0.0, y = cardHeight - 0.02;
	while(pairs)
	{
		double z = z0;
		for(int i = 0; i < pairs; ++i)
		{
			const double positions[3][3] = { { 0.0, y + cardHeight - 0.015, z + 0.25 }, { 0.0, y, z }, { 0.0, y, z + 0.175 } };
			for(int card = i + 1 == pairs ? 1 : 0; card < 3; ++card)
			{
				std::fprintf(file, "\t\t<body pos=\"%.9g %.9g %.9g\" quat=\"%.17g %.17g 0 0\"><freejoint/>"
					"<geom size=\"%.9g %.9g %.9g\" mass=\"0.01\"/></body>\n",
					positions[card][0], positions[card][1], positions[card][2], std::cos(0.5 * angles[card]), std::sin(0.5 * angles[card]),
					cardWidth, cardHeight, cardThickness);
				++cards;
			}
			z += 0.35;
		}
		y += 2.0 * cardHeight - 0.03;
		z0 += 0.175;
		--pairs;
	}
	std::fprintf(file, "\t</worldbody>\n</mujoco>\n");
	std::fclose(file);
	return cards;
}

int main(int argc, const char* const* argv)
{
	if(argc < 2)
	{
		std::printf("MujocoCardHouse scene.xml [threads=1]\n");
		return 1;
	}
	const int threads = argc > 2 ? std::atoi(argv[2]) : 1;
	const int cardCount = writeScene(argv[1]);
	if(!cardCount)
		return 1;
	char error[1024];
	mjModel* model = mj_loadXML(argv[1], NULL, error, sizeof(error));
	if(!model)
	{
		std::fprintf(stderr, "%s\n", error);
		return 1;
	}
	mjData* data = mj_makeData(model);
	mju_threadpool(data, threads > 1 ? threads : 0);
	std::printf("MuJoCo %s Newton card house: %d cards, friction 0.8, %d threads\n", mj_versionString(), cardCount, threads);
	const int frames = 1000, settleFrames = 100;
	// Card i is body i + 1 (body 0 is the world), framed at its centre.
	std::vector<double> settledPosition(3 * size_t(cardCount)), settledRotation(4 * size_t(cardCount));
	double maximumDisplacement = 0.0, maximumRotation = 0.0, maximumFloorPenetration = 0.0, lowestTop = 1e30;
	double totalMs = 0.0;
	for(int frame = 0; frame < frames; ++frame)
	{
		const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		mj_step1(model, data);
		mujocoConveyor::applyRestDistance(data);
		mj_step2(model, data);
		totalMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		// mj_step2 leaves the body frames at the start of the step; refresh them for the new state.
		mj_kinematics(model, data);
		for(int i = 0; i < cardCount; ++i)
		{
			const mjtNum* position = data->xpos + 3 * (i + 1);
			const mjtNum* rotation = data->xquat + 4 * (i + 1);
			if(frame + 1 == settleFrames)
			{
				mju_copy3(&settledPosition[3 * size_t(i)], position);
				mju_copy4(&settledRotation[4 * size_t(i)], rotation);
			}
			if(frame + 1 <= settleFrames)
				continue;
			mjtNum difference[3];
			mju_sub3(difference, position, &settledPosition[3 * size_t(i)]);
			maximumDisplacement = std::max(maximumDisplacement, double(mju_norm3(difference)));
			mjtNum conjugate[4], change[4];
			mju_negQuat(conjugate, &settledRotation[4 * size_t(i)]);
			mju_mulQuat(change, rotation, conjugate);
			maximumRotation = std::max(maximumRotation, 2.0 * std::acos(std::min(1.0, std::abs(double(change[0])))));
			const mjtNum* frameAxes = data->xmat + 9 * (i + 1);
			double bottom = 1e30;
			for(int corner = 0; corner < 8; ++corner)
			{
				const mjtNum local[3] = { (corner & 1 ? 1.0 : -1.0) * cardWidth, (corner & 2 ? 1.0 : -1.0) * cardHeight, (corner & 4 ? 1.0 : -1.0) * cardThickness };
				mjtNum world[3];
				mju_mulMatVec3(world, frameAxes, local);
				bottom = std::min(bottom, double(position[1] + world[1]));
			}
			maximumFloorPenetration = std::max(maximumFloorPenetration, -bottom);
		}
		if(frame + 1 > settleFrames)
			lowestTop = std::min(lowestTop, double(data->xpos[3 * cardCount + 1]));
	}
	double maximumSpeed = 0.0;
	for(int i = 0; i < cardCount; ++i)
		maximumSpeed = std::max(maximumSpeed, double(mju_norm3(data->qvel + 6 * i)));
	const double topDrop = settledPosition[3 * size_t(cardCount - 1) + 1] - lowestTop;
	std::printf("card_house cards=%d max_displacement_mm=%.4f max_rotation_deg=%.4f top_drop_mm=%.4f floor_penetration_um=%.3f final_max_speed_mm_s=%.4f"
		" mean_step_ms=%.4f total_s=%.3f\n", cardCount, maximumDisplacement * 1e3, maximumRotation * 180.0 / mjPI, topDrop * 1e3, maximumFloorPenetration * 1e6,
		maximumSpeed * 1e3, totalMs / frames, totalMs * 1e-3);
	check(maximumDisplacement < 1.0e-3, "card_house: no card moves more than 1 mm once the house has settled");
	check(maximumRotation < 0.1 * mjPI / 180.0, "card_house: no card turns more than 0.1 degree once the house has settled");
	check(maximumSpeed < 1.0e-3, "card_house: house is at rest (below 1 mm/s)");
	check(maximumFloorPenetration < 20.0e-6, "card_house: cards sink less than 20 um into the floor");
	for(int i = 0; i < mjNWARNING; ++i)
	{
		if(data->warning[i].number)
			std::printf("WARNING %d count %d\n", i, data->warning[i].number);
	}
	mj_deleteData(data);
	mj_deleteModel(model);
	return failures ? 1 : 0;
}
