#ifndef ANVIL_CARD_HOUSE_SCENE_H
#define ANVIL_CARD_HOUSE_SCENE_H

// The card house of the FBF paper's examples (github.com/matthcsong/fbf-sca-2026,
// paper_examples/card-house/run.py), shared by the PhysX and MuJoCo runners. Five levels of tents,
// two cards leaning together at 65 degrees from horizontal, with flat bridge cards tilted 1 degree
// across the tops of neighbouring tents carrying the next level: 40 cards. Cards start slightly
// apart (the lowest tents 6.7 cm above the ground at the paper's scale) and fall into place.
//
// The paper's cards are 2.5 x 1.25 m and 40 mm thick, 25 kg (density 200). The playing-card scale
// keeps the layout, scaled by card length, with real cards: 88 x 63 mm, 0.3 mm thick, 1.8 g.
// Friction is the paper's 0.8: a tent holds at its top only above tan 25 = 0.47.
//
// y is up (the paper's z); cards lean in the x-y plane (rotations about z), the paper's x-z plane.
#include <cmath>
#include <vector>

namespace cardHouse
{
static const double pi = 3.14159265358979323846;
static const double leanFromHorizontal = 65.0 * pi / 180.0;
static const double bridgeTilt = 1.0 * pi / 180.0;
static const int levels = 5;
static const double friction = 0.8;

// The paper's layout, in its units.
static const double tentHalfGap = 0.55;
static const double tentWidth = 4.0 * tentHalfGap;
static const double tentHeight = (std::tan(leanFromHorizontal) + std::tan(3.0 * pi / 180.0)) * tentWidth * 0.5;

struct Scale
{
	const char* name;
	double halfLength, halfDepth, halfThickness;
	double layout;  // Paper units to metres.
	double mass;
};

static const Scale paper = { "fbf", 1.25, 0.625, 0.02, 1.0, 2.5 * 1.25 * 0.04 * 200.0 };
static const Scale playingCards = { "playing_cards", 0.044, 0.0315, 0.00015, 0.088 / 2.5, 0.0018 };

struct Card
{
	double position[3];
	double angle;  // About z, radians.
	double halfExtents[3];
	int level, tent;
	char role;  // 'B'ridge, 'L'eft or 'R'ight card of a tent.
};

// The height (metres) of the base of the top level, which a standing house's top card stays above:
// it starts half a level higher and only falls through the starting gaps.
inline double topLevelBase(const Scale& scale, int levelCount = levels)
{
	return (levelCount - 1) * tentHeight * scale.layout;
}

// The paper's construction, level by level: a bridge (above the ground level) and then the tent's
// left and right cards, whose tops lean towards each other.
inline std::vector<Card> build(const Scale& scale, int levelCount = levels)
{
	std::vector<Card> cards;
	const double k = scale.layout;
	for(int i = 0; i < levelCount; ++i)
	{
		for(int j = 0; j < levelCount - i; ++j)
		{
			const double x = (j - (levelCount - i) / 2.0 + 0.5) * tentWidth;
			if(i != 0)
			{
				const Card bridge = { { x * k, i * tentHeight * k, 0.0 }, bridgeTilt, { scale.halfLength, scale.halfThickness, scale.halfDepth }, i, j, 'B' };
				cards.push_back(bridge);
			}
			const double y = (i + 0.5) * tentHeight * k;
			const double lean = pi * 0.5 - leanFromHorizontal;
			const Card left = { { (x - tentHalfGap) * k, y, 0.0 }, -lean, { scale.halfThickness, scale.halfLength, scale.halfDepth }, i, j, 'L' };
			const Card right = { { (x + tentHalfGap) * k, y, 0.0 }, lean, { scale.halfThickness, scale.halfLength, scale.halfDepth }, i, j, 'R' };
			cards.push_back(left);
			cards.push_back(right);
		}
	}
	return cards;
}
}

#endif
