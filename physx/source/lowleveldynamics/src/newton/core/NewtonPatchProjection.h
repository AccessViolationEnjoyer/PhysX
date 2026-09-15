#ifndef NEWTON_PATCH_PROJECTION_H
#define NEWTON_PATCH_PROJECTION_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace newton
{
// The numerical core uses IEEE binary64. MSVC's std::isfinite calls the CRT
// classifier in this hot projection loop; inspect the exponent bits directly
// without aliasing or floating-point operations. Signed zeros and subnormals
// remain finite, and every infinity/NaN payload is rejected.
inline bool isFiniteDouble(double value)
{
	static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559 &&
		std::numeric_limits<double>::digits == 53 && std::numeric_limits<double>::max_exponent == 1024,
		"Newton finite classification requires IEEE binary64");
	std::uint64_t bits;
	std::memcpy(&bits, &value, sizeof(bits));
	return (bits & 0x7ff0000000000000ull) != 0x7ff0000000000000ull;
}

// Minimize sum(0.5 * R * lambda^2 + s * lambda), with
// 0 <= normal[i] <= cap[i] and |tangent[j]| <= friction[j] * sum(normal).
// Inputs are row residuals s = J*v + freeVelocity and positive diagonal R.
// A native patch has at most two anchors, hence at most four tangent rows.
struct PatchProjectionInput
{
	int normalCount;
	int tangentCount;
	const double* normalVelocity;
	const double* normalRegularization;
	const double* normalCap;
	const double* tangentVelocity;
	const double* tangentRegularization;
	const double* friction;
};

// Impulse arrays are required for nonempty row sets. Derivative arrays are
// optional. All arrays belong to the caller and must not overlap each other or
// the input arrays. This helper neither allocates nor retains storage.
struct PatchProjectionOutput
{
	double* normalImpulse;
	double* tangentImpulse;
	double* normalDiagonal;
	double* tangentDiagonal;
	double* tangentCoupling;
};

struct PatchProjectionResult
{
	double normalSum;
	double normalShift;
	double cost;
	double normalInverseRegularization;
	double boundedTangentRegularization;
	double inverseCoupling;

	// Curvature is H = -d(lambda)/ds. With w = normalDiagonal,
	// u = tangentCoupling, A = normalInverseRegularization,
	// B = boundedTangentRegularization and D = 1 + A*B:
	// H_nn = diag(w) - (B/D)*w*w^T,
	// H_nt = -w*u^T/D,
	// H_tt = diag(tangentDiagonal) + (A/D)*u*u^T.
	// J^T*H*J therefore needs only the diagonal row terms and two
	// body-local weighted Jacobian sums, not a dense patch matrix.
};

template<bool ValidateInput>
inline bool projectPatchInternal(const PatchProjectionInput& input, const PatchProjectionOutput& output,
	PatchProjectionResult& result)
{
	if(ValidateInput)
	{
		if(input.normalCount < 0 || input.tangentCount < 0 || input.tangentCount > 4)
			return false;
		if(input.normalCount && (!input.normalVelocity || !input.normalRegularization || !input.normalCap || !output.normalImpulse))
			return false;
		if(input.tangentCount && (!input.tangentVelocity || !input.tangentRegularization || !input.friction || !output.tangentImpulse))
			return false;

		const double infinity = std::numeric_limits<double>::infinity();
		for(int i = 0; i < input.normalCount; ++i)
		{
			if(!isFiniteDouble(input.normalVelocity[i]) || !isFiniteDouble(input.normalRegularization[i]) ||
				input.normalRegularization[i] <= 0.0 || input.normalCap[i] < 0.0 ||
				(!isFiniteDouble(input.normalCap[i]) && input.normalCap[i] != infinity))
				return false;
		}
		for(int j = 0; j < input.tangentCount; ++j)
		{
			if(!isFiniteDouble(input.tangentVelocity[j]) || !isFiniteDouble(input.tangentRegularization[j]) ||
				input.tangentRegularization[j] <= 0.0 || !isFiniteDouble(input.friction[j]) || input.friction[j] < 0.0)
				return false;
		}
	}
	const double infinity = std::numeric_limits<double>::infinity();
	// For a common normal shift q, n_i = clamp((q-s_i)/R_i,0,cap_i).
	// Stationarity gives q = sum(mu_j * max(|s_j|-R_j*mu_j*N(q),0)).
	// The left side increases and the right side decreases. Walk its finite
	// linear pieces from q=0, stopping at either the root or the next row event.
	double shift = 0.0;
	double solutionBase = 0.0;
	double solutionStep = 0.0;
	unsigned int activeTangents = (1u << input.tangentCount) - 1u;
	const unsigned long long maximumPieces = 2ull * static_cast<unsigned long long>(input.normalCount) +
		static_cast<unsigned long long>(input.tangentCount) + 2ull;
	bool solved = false;
	for(unsigned long long piece = 0; piece < maximumPieces; ++piece)
	{
		double normalSum = 0.0;
		double normalSlope = 0.0;
		double nextEvent = infinity;
		int tangentEvent = -1;
		for(int i = 0; i < input.normalCount; ++i)
		{
			const double velocity = input.normalVelocity[i];
			const double regularization = input.normalRegularization[i];
			const double cap = input.normalCap[i];
			const double upperEvent = velocity + regularization * cap;
			const double impulse = (shift - velocity) / regularization;
			normalSum += impulse <= 0.0 ? 0.0 : (impulse >= cap ? cap : impulse);
			// Right derivative at entry/cap boundaries for the monotone walk.
			if(shift >= velocity && shift < upperEvent && cap > 0.0)
				normalSlope += 1.0 / regularization;
			if(velocity > shift && velocity < nextEvent)
				nextEvent = velocity;
			if(upperEvent > shift && upperEvent < nextEvent)
				nextEvent = upperEvent;
		}

		double tangentShift = 0.0;
		double tangentSlope = 0.0;
		for(int j = 0; j < input.tangentCount; ++j)
		{
			const double friction = input.friction[j];
			const double slope = input.tangentRegularization[j] * friction;
			const double excess = std::abs(input.tangentVelocity[j]) - slope * normalSum;
			if((activeTangents & (1u << j)) && excess > 0.0 && friction > 0.0)
			{
				tangentShift += friction * excess;
				tangentSlope += friction * slope;
				if(normalSlope > 0.0 && slope > 0.0)
				{
					const double event = shift + (excess / slope) / normalSlope;
					if(event < nextEvent)
					{
						nextEvent = event;
						tangentEvent = j;
					}
				}
			}
			else
				activeTangents &= ~(1u << j);
		}
		const double denominator = 1.0 + normalSlope * tangentSlope;
		if(!isFiniteDouble(normalSum) || !isFiniteDouble(tangentShift) || !isFiniteDouble(denominator))
			return false;
		const double step = (tangentShift - shift) / denominator;
		if(step <= 0.0)
		{
			solutionBase = shift;
			solved = true;
			break;
		}
		const double root = shift + step;
		const bool rootInPiece = root <= nextEvent;
		const double nextShift = rootInPiece ? root : nextEvent;
		if(!isFiniteDouble(nextShift))
			return false;
		if(rootInPiece)
		{
			solutionBase = shift;
			solutionStep = step;
			shift = nextShift;
			solved = true;
			break;
		}
		shift = nextShift;
		// A tangent event may round to the current q. Consume it explicitly,
		// so a sub-ulp event cannot be mistaken for convergence or revisited.
		if(tangentEvent >= 0)
			activeTangents &= ~(1u << tangentEvent);
	}
	if(!solved)
		return false;

	result.normalSum = 0.0;
	result.normalShift = shift;
	result.cost = 0.0;
	result.normalInverseRegularization = 0.0;
	result.boundedTangentRegularization = 0.0;
	for(int i = 0; i < input.normalCount; ++i)
	{
		const double velocity = input.normalVelocity[i];
		const double regularization = input.normalRegularization[i];
		const double cap = input.normalCap[i];
		// Preserve the root increment separately: rounding q before subtracting a
		// nearby s can lose accuracy when normal regularization is very small.
		const double unconstrained = ((solutionBase - velocity) / regularization) + solutionStep / regularization;
		const double impulse = unconstrained <= 0.0 ? 0.0 : (unconstrained >= cap ? cap : unconstrained);
		const double diagonal = impulse > 0.0 && impulse < cap ? 1.0 / regularization : 0.0;
		output.normalImpulse[i] = impulse;
		if(output.normalDiagonal)
			output.normalDiagonal[i] = diagonal;
		result.normalSum += impulse;
		result.normalInverseRegularization += diagonal;
		result.cost -= impulse * (velocity + 0.5 * regularization * impulse);
	}
	for(int j = 0; j < input.tangentCount; ++j)
	{
		const double velocity = input.tangentVelocity[j];
		const double regularization = input.tangentRegularization[j];
		const double friction = input.friction[j];
		const double limit = friction * result.normalSum;
		const double unconstrained = std::abs(velocity) / regularization;
		const double sign = velocity < 0.0 ? -1.0 : 1.0;
		const bool bounded = unconstrained >= limit;
		const double impulse = -sign * (bounded ? limit : unconstrained);
		output.tangentImpulse[j] = impulse;
		if(output.tangentDiagonal)
			output.tangentDiagonal[j] = bounded ? 0.0 : 1.0 / regularization;
		if(output.tangentCoupling)
			output.tangentCoupling[j] = bounded ? sign * friction : 0.0;
		if(bounded)
			result.boundedTangentRegularization += regularization * friction * friction;
		result.cost -= impulse * (velocity + 0.5 * regularization * impulse);
	}
	const double denominator = 1.0 + result.normalInverseRegularization * result.boundedTangentRegularization;
	result.inverseCoupling = 1.0 / denominator;
	return isFiniteDouble(result.normalSum) && isFiniteDouble(result.cost) && isFiniteDouble(denominator);
}

inline bool projectPatch(const PatchProjectionInput& input, const PatchProjectionOutput& output,
	PatchProjectionResult& result)
{
	return projectPatchInternal<true>(input, output, result);
}

// Internal prepared-input entry. Counts/pointers, finite residuals, positive R,
// nonnegative caps and finite nonnegative friction must already be validated.
// Keep arithmetic/result checks: a failed projection must not publish impulses.
inline bool projectPatchUnchecked(const PatchProjectionInput& input, const PatchProjectionOutput& output,
	PatchProjectionResult& result)
{
	return projectPatchInternal<false>(input, output, result);
}
}

#endif
