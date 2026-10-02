// The card_house test of BehaviourTests.cpp (the FBF paper's card house, CardHouseScene.h) in
// MuJoCo's Newton solver, at the paper's scale and with real playing cards. The houses settle for
// two seconds and should then stand still; the checks and their limits are the Anvil test's.
// Solver and contact settings match the other MuJoCo comparison scenes.
//
// usage: MujocoCardHouse scene-prefix [threads=1]   (writes scene-prefix-<scale>.xml)
#include "CardHouseScene.h"
#include "MujocoConveyor.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* message)
{
	if(!condition)
	{
		std::printf("FAIL %s\n", message);
		++failures;
	}
}

static bool writeScene(const std::string& path, const cardHouse::Scale& scale, const std::vector<cardHouse::Card>& cards)
{
	FILE* file = std::fopen(path.c_str(), "w");
	if(!file)
		return false;
	const double floorHalf = 10.0 * scale.layout;
	std::fprintf(file, "<mujoco model=\"Card house\">\n"
		"\t<option timestep=\"0.01\" gravity=\"0 -9.81 0\" integrator=\"Euler\" solver=\"Newton\"\n"
		"\t\tcone=\"pyramidal\" jacobian=\"sparse\" iterations=\"100\" tolerance=\"1e-8\" ls_tolerance=\"0.01\"/>\n"
		"\t<default><geom type=\"box\" condim=\"3\" friction=\"%g 0 0\" margin=\"0.001\" gap=\"0\"\n"
		"\t\tsolref=\"0.02 1\" solimp=\"0.9999 0.9999 0.001 0.5 2\"/></default>\n"
		"\t<worldbody>\n\t\t<geom name=\"floor\" pos=\"0 -0.5 0\" size=\"%.9g 0.5 %.9g\"/>\n", cardHouse::friction, floorHalf, floorHalf);
	for(size_t i = 0; i < cards.size(); ++i)
	{
		const cardHouse::Card& card = cards[i];
		std::fprintf(file, "\t\t<body pos=\"%.9g %.9g %.9g\" quat=\"%.17g 0 0 %.17g\"><freejoint/><geom size=\"%.9g %.9g %.9g\" mass=\"%.9g\"/></body>\n",
			card.position[0], card.position[1], card.position[2], std::cos(0.5 * card.angle), std::sin(0.5 * card.angle),
			card.halfExtents[0], card.halfExtents[1], card.halfExtents[2], scale.mass);
	}
	std::fprintf(file, "\t</worldbody>\n</mujoco>\n");
	std::fclose(file);
	return true;
}

static void run(const std::string& prefix, const cardHouse::Scale& scale, bool checkStanding, int threads)
{
	const std::vector<cardHouse::Card> cards = cardHouse::build(scale);
	const int cardCount = int(cards.size());
	const std::string path = prefix + "-" + scale.name + ".xml";
	if(!writeScene(path, scale, cards))
	{
		++failures;
		return;
	}
	char error[1024];
	mjModel* model = mj_loadXML(path.c_str(), NULL, error, sizeof(error));
	if(!model)
	{
		std::fprintf(stderr, "%s\n", error);
		++failures;
		return;
	}
	mjData* data = mj_makeData(model);
	mju_threadpool(data, threads > 1 ? threads : 0);
	const int frames = 1000, settleFrames = 200;
	const double length = 2.0 * scale.halfLength;
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
				const double* half = cards[size_t(i)].halfExtents;
				const mjtNum local[3] = { (corner & 1 ? 1.0 : -1.0) * half[0], (corner & 2 ? 1.0 : -1.0) * half[1], (corner & 4 ? 1.0 : -1.0) * half[2] };
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
	// Settling, from the starting layout: how far each card turned and how far the top card fell.
	double maximumSettleTurn = 0.0;
	for(int i = 0; i < cardCount; ++i)
	{
		const double startQuat[4] = { std::cos(0.5 * cards[size_t(i)].angle), 0.0, 0.0, std::sin(0.5 * cards[size_t(i)].angle) };
		mjtNum conjugate[4], change[4];
		mju_negQuat(conjugate, startQuat);
		mju_mulQuat(change, &settledRotation[4 * size_t(i)], conjugate);
		maximumSettleTurn = std::max(maximumSettleTurn, 2.0 * std::acos(std::min(1.0, std::abs(double(change[0])))));
	}
	const double settledTop = settledPosition[3 * size_t(cardCount - 1) + 1];
	const double topDrop = settledTop - lowestTop;
	std::printf("card_house %s settle_turn_deg=%.3f top_settle_fall_mm=%.3f top_start_height_mm=%.3f\n", scale.name, maximumSettleTurn * 180.0 / mjPI,
		(cards.back().position[1] - settledTop) * 1e3, cards.back().position[1] * 1e3);
	std::printf("card_house %s cards=%d max_displacement_mm=%.4f max_rotation_deg=%.4f top_drop_mm=%.4f floor_penetration_um=%.3f final_max_speed_mm_s=%.4f"
		" mean_step_ms=%.4f total_s=%.3f\n", scale.name, cardCount, maximumDisplacement * 1e3, maximumRotation * 180.0 / mjPI, topDrop * 1e3, maximumFloorPenetration * 1e6,
		maximumSpeed * 1e3, totalMs / frames, totalMs * 1e-3);
	if(checkStanding)
		check(settledTop > cardHouse::topLevelBase(scale), "card_house: the house stands (the top card stays on the top level while settling)");
	check(maximumDisplacement < 0.0025 * length, "card_house: no card moves more than 0.25% of its length once the house has settled");
	check(maximumRotation < 0.1 * mjPI / 180.0, "card_house: no card turns more than 0.1 degree once the house has settled");
	check(maximumSpeed < 0.0025 * length, "card_house: house is at rest (below 0.25% of a card length per second)");
	check(maximumFloorPenetration < 20.0e-6, "card_house: cards sink less than 20 um into the floor");
	for(int i = 0; i < mjNWARNING; ++i)
	{
		if(data->warning[i].number)
			std::printf("WARNING %d count %d\n", i, data->warning[i].number);
	}
	mj_deleteData(data);
	mj_deleteModel(model);
}

int main(int argc, const char* const* argv)
{
	if(argc < 2)
	{
		std::printf("MujocoCardHouse scene-prefix [threads=1]\n");
		return 1;
	}
	const int threads = argc > 2 ? std::atoi(argv[2]) : 1;
	std::printf("MuJoCo %s Newton card house, friction %g, %d threads\n", mj_versionString(), cardHouse::friction, threads);
	run(argv[1], cardHouse::paper, true, threads);
	run(argv[1], cardHouse::playingCards, true, threads);
	return failures ? 1 : 0;
}
