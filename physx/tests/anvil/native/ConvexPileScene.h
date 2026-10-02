#ifndef ANVIL_CONVEX_PILE_SCENE_H
#define ANVIL_CONVEX_PILE_SCENE_H

// PEEL's "PileOfLargeConvexes" (Pierre Terdiman's PEEL, TestScenes_Performance.cpp), shared by the
// PhysX and MuJoCo runners: columns of copies of one convex hull, the hull of 32 random points on a
// sphere of radius 0.4 m (PEEL's amplitude 4 at scale 0.1), 1 kg each, stacked 0.8 m apart from
// 0.4 m above the ground. The points use PEEL's generator: ICE BasicRandom seeded with 42, each
// point three RandomFloat components normalised to unit length.
#include <cmath>
#include <cstdint>

namespace convexPile
{
static const double amplitude = 0.4;
static const double spacing = 2.0 * amplitude;
static const int pointCount = 32;
static const double mass = 1.0;

// ICE's BasicRandom: RandomFloat is uniform in [-0.5, 0.5].
class BasicRandom
{
public:
	explicit BasicRandom(uint32_t seed) : mValue(seed) {}
	float randomFloat()
	{
		mValue = mValue * 2147001325u + 715136305u;
		return float(mValue & 0xffff) / 65535.0f - 0.5f;
	}

private:
	uint32_t mValue;
};

// The hull's points, x y z per point.
inline void hullPoints(float* points)
{
	BasicRandom random(42);
	for(int i = 0; i < pointCount; ++i)
	{
		const float x = random.randomFloat(), y = random.randomFloat(), z = random.randomFloat();
		const float scale = float(amplitude) / std::sqrt(x * x + y * y + z * z);
		points[3 * i] = x * scale;
		points[3 * i + 1] = y * scale;
		points[3 * i + 2] = z * scale;
	}
}

// PEEL's placement, including its offset of half a column from the centre; y is up.
inline void position(int column, int row, int layer, int columns, double* position)
{
	position[0] = (double(column) - double(columns) * 0.5) * spacing;
	position[1] = amplitude + spacing * double(layer);
	position[2] = (double(row) - double(columns) * 0.5) * spacing;
}
}

#endif
