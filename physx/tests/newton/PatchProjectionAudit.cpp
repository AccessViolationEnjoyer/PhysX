#include "NewtonPatchProjection.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

namespace
{
const int MAX_ROWS = 64;
struct Sample
{
	int normalCount;
	int tangentCount;
	std::array<double, MAX_ROWS> normalVelocity, normalRegularization, normalCap;
	std::array<double, MAX_ROWS> tangentVelocity, tangentRegularization, friction;
};
struct Answer
{
	std::array<double, MAX_ROWS> normal, tangent, normalDiagonal, tangentDiagonal, tangentCoupling;
	newton::PatchProjectionResult result;
};

bool project(const Sample& sample, Answer& answer)
{
	const newton::PatchProjectionInput input = {sample.normalCount, sample.tangentCount,
		sample.normalVelocity.data(), sample.normalRegularization.data(), sample.normalCap.data(),
		sample.tangentVelocity.data(), sample.tangentRegularization.data(), sample.friction.data()};
	const newton::PatchProjectionOutput output = {answer.normal.data(), answer.tangent.data(),
		answer.normalDiagonal.data(), answer.tangentDiagonal.data(), answer.tangentCoupling.data()};
	return newton::projectPatch(input, output, answer.result);
}

void dump(const Sample& sample, int trial)
{
	std::printf("FAIL trial=%d normals=%d tangents=%d\n", trial, sample.normalCount, sample.tangentCount);
	for(int i = 0; i < sample.normalCount; ++i)
		std::printf("n %.17g %.17g %.17g\n", sample.normalVelocity[i], sample.normalRegularization[i], sample.normalCap[i]);
	for(int j = 0; j < sample.tangentCount; ++j)
		std::printf("t %.17g %.17g %.17g\n", sample.tangentVelocity[j], sample.tangentRegularization[j], sample.friction[j]);
}

bool checkOptimality(const Sample& sample, const Answer& answer, double& maximumError)
{
	double normalSum = 0.0;
	for(int i = 0; i < sample.normalCount; ++i)
	{
		if(answer.normal[i] < 0.0 || answer.normal[i] > sample.normalCap[i])
			return false;
		normalSum += answer.normal[i];
	}
	double shift = 0.0;
	for(int j = 0; j < sample.tangentCount; ++j)
	{
		const double bound = sample.friction[j] * normalSum;
		const double freeImpulse = -sample.tangentVelocity[j] / sample.tangentRegularization[j];
		const double reference = std::max(-bound, std::min(bound, freeImpulse));
		const double error = std::abs(reference - answer.tangent[j]) / std::max(1.0, std::abs(reference));
		maximumError = std::max(maximumError, error);
		if(error > 1e-11)
			return false;
		shift += sample.friction[j] * std::max(0.0, std::abs(sample.tangentVelocity[j]) - sample.tangentRegularization[j] * bound);
	}
	for(int i = 0; i < sample.normalCount; ++i)
	{
		const double residual = sample.normalRegularization[i] * answer.normal[i] + sample.normalVelocity[i] - shift;
		const double scale = std::max(1.0, std::max(std::abs(sample.normalVelocity[i]), std::abs(shift)));
		double error = std::abs(residual) / scale;
		if(answer.normal[i] == 0.0)
			error = std::max(0.0, -residual) / scale;
		if(answer.normal[i] == sample.normalCap[i])
			error = std::min(error, std::max(0.0, residual) / scale);
		maximumError = std::max(maximumError, error);
		if(error > 2e-10)
		{
			std::printf("KKT normal=%d impulse=%.17g shift=%.17g helperShift=%.17g residual=%.17g\n", i, answer.normal[i], shift, answer.result.normalShift, residual);
			return false;
		}
	}
	return std::isfinite(answer.result.cost) && answer.result.cost >= -1e-10;
}

// Independent exhaustive active-set reference for small patches. All normal
// states (zero/free/capped) and tangent states (free/bounded) are enumerated.
bool enumerateReference(const Sample& sample, const Answer& answer)
{
	int normalStates = 1;
	for(int i = 0; i < sample.normalCount; ++i)
		normalStates *= 3;
	const int tangentStates = 1 << sample.tangentCount;
	for(int normalState = 0; normalState < normalStates; ++normalState)
	{
		int state = normalState;
		std::array<int, MAX_ROWS> normalMode;
		double coefficient = 0.0;
		double constant = 0.0;
		for(int i = 0; i < sample.normalCount; ++i)
		{
			normalMode[i] = state % 3;
			state /= 3;
			if(normalMode[i] == 1)
			{
				coefficient += 1.0 / sample.normalRegularization[i];
				constant -= sample.normalVelocity[i] / sample.normalRegularization[i];
			}
			else if(normalMode[i] == 2)
				constant += sample.normalCap[i];
		}
		if(!std::isfinite(constant))
			continue;
		for(int tangentState = 0; tangentState < tangentStates; ++tangentState)
		{
			double tangentConstant = 0.0;
			double tangentCoefficient = 0.0;
			for(int j = 0; j < sample.tangentCount; ++j)
			{
				if(tangentState & (1 << j))
				{
					tangentConstant += sample.friction[j] * std::abs(sample.tangentVelocity[j]);
					tangentCoefficient += sample.friction[j] * sample.friction[j] * sample.tangentRegularization[j];
				}
			}
			const double normalSum = (constant + coefficient * tangentConstant) / (1.0 + coefficient * tangentCoefficient);
			const double shift = tangentConstant - tangentCoefficient * normalSum;
			if(normalSum < -1e-12 || shift < -1e-12)
				continue;
			Answer reference = {};
			bool valid = true;
			for(int i = 0; i < sample.normalCount; ++i)
			{
				const double freeImpulse = (shift - sample.normalVelocity[i]) / sample.normalRegularization[i];
				if(normalMode[i] == 0)
				{
					reference.normal[i] = 0.0;
					valid = valid && freeImpulse <= 1e-10;
				}
				else if(normalMode[i] == 1)
				{
					reference.normal[i] = freeImpulse;
					valid = valid && freeImpulse >= 0.0 && freeImpulse <= sample.normalCap[i];
				}
				else
				{
					reference.normal[i] = sample.normalCap[i];
					valid = valid && freeImpulse >= sample.normalCap[i] - 1e-10;
				}
			}
			for(int j = 0; j < sample.tangentCount; ++j)
			{
				const double freeImpulse = std::abs(sample.tangentVelocity[j]) / sample.tangentRegularization[j];
				const double bound = sample.friction[j] * normalSum;
				const bool bounded = (tangentState & (1 << j)) != 0;
				valid = valid && (bounded ? freeImpulse >= bound - 1e-10 : freeImpulse <= bound + 1e-10);
				reference.tangent[j] = (sample.tangentVelocity[j] < 0.0 ? 1.0 : -1.0) * (bounded ? bound : freeImpulse);
			}
			if(!valid)
				continue;
			for(int i = 0; i < sample.normalCount; ++i)
				if(std::abs(reference.normal[i] - answer.normal[i]) > 2e-9 * std::max(1.0, std::abs(reference.normal[i])))
					return false;
			for(int j = 0; j < sample.tangentCount; ++j)
				if(std::abs(reference.tangent[j] - answer.tangent[j]) > 2e-9 * std::max(1.0, std::abs(reference.tangent[j])))
					return false;
			return true;
		}
	}
	return false;
}

bool samePiece(const Sample& sample, const Answer& a, const Answer& b)
{
	for(int i = 0; i < sample.normalCount; ++i)
		if(a.normalDiagonal[i] != b.normalDiagonal[i])
			return false;
	for(int j = 0; j < sample.tangentCount; ++j)
		if(a.tangentDiagonal[j] != b.tangentDiagonal[j] || a.tangentCoupling[j] != b.tangentCoupling[j])
			return false;
	return true;
}

bool checkDerivative(Sample& sample, const Answer& answer, int& checked, double& maximumError)
{
	const int rows = sample.normalCount + sample.tangentCount;
	for(int column = 0; column < rows; ++column)
	{
		double& velocity = column < sample.normalCount ? sample.normalVelocity[column] : sample.tangentVelocity[column - sample.normalCount];
		const double original = velocity;
		const double step = 1e-5 * std::max(1.0, std::abs(original));
		Answer plus, minus;
		velocity = original + step;
		const bool plusValid = project(sample, plus);
		velocity = original - step;
		const bool minusValid = project(sample, minus);
		velocity = original;
		if(!plusValid || !minusValid)
			return false;
		if(!samePiece(sample, answer, plus) || !samePiece(sample, answer, minus))
			continue;
		for(int row = 0; row < rows; ++row)
		{
			const double wn = row < sample.normalCount ? answer.normalDiagonal[row] : 0.0;
			const double un = row >= sample.normalCount ? answer.tangentCoupling[row - sample.normalCount] : 0.0;
			const double wm = column < sample.normalCount ? answer.normalDiagonal[column] : 0.0;
			const double um = column >= sample.normalCount ? answer.tangentCoupling[column - sample.normalCount] : 0.0;
			double predicted = answer.result.inverseCoupling *
				(-answer.result.boundedTangentRegularization * wn * wm - wn * um - un * wm + answer.result.normalInverseRegularization * un * um);
			if(row == column)
				predicted += row < sample.normalCount ? answer.normalDiagonal[row] : answer.tangentDiagonal[row - sample.normalCount];
			const double plusImpulse = row < sample.normalCount ? plus.normal[row] : plus.tangent[row - sample.normalCount];
			const double minusImpulse = row < sample.normalCount ? minus.normal[row] : minus.tangent[row - sample.normalCount];
			const double observed = -(plusImpulse - minusImpulse) / (2.0 * step);
			const double error = std::abs(predicted - observed) / std::max(1.0, std::abs(predicted));
			maximumError = std::max(maximumError, error);
			if(error > 3e-6)
			{
				std::printf("derivative row=%d col=%d expected=%.17g observed=%.17g\n", row, column, predicted, observed);
				return false;
			}
		}
		const double impulse = column < sample.normalCount ? answer.normal[column] : answer.tangent[column - sample.normalCount];
		const double gradient = (plus.result.cost - minus.result.cost) / (2.0 * step);
		if(std::abs(gradient + impulse) > 2e-6 * std::max(1.0, std::abs(impulse)))
			return false;
		++checked;
	}
	return true;
}
}

int main()
{
	std::mt19937 random(927481u);
	std::uniform_real_distribution<double> unit(0.0, 1.0);
	double maximumKkt = 0.0;
	double maximumDerivative = 0.0;
	int derivatives = 0;
	for(int trial = 0; trial < 20000; ++trial)
	{
		Sample sample = {};
		sample.normalCount = trial < 300 ? 1 + trial % 4 : trial % 33;
		sample.tangentCount = trial % 5;
		for(int i = 0; i < sample.normalCount; ++i)
		{
			sample.normalVelocity[i] = 20.0 * unit(random) - 10.0;
			sample.normalRegularization[i] = std::pow(10.0, (trial < 1000 ? 4.0 : 9.0) * unit(random) - (trial < 1000 ? 2.0 : 6.0));
			sample.normalCap[i] = unit(random) < 0.4 ? (std::numeric_limits<double>::max)() : 3.0 * unit(random);
			if(trial % 11 == 0)
				sample.normalCap[i] = 0.0;
		}
		for(int j = 0; j < sample.tangentCount; ++j)
		{
			sample.tangentVelocity[j] = 20.0 * unit(random) - 10.0;
			sample.tangentRegularization[j] = std::pow(10.0, 4.0 * unit(random) - 2.0);
			sample.friction[j] = trial % 13 == 0 ? 0.0 : 2.0 * unit(random);
		}
		Answer answer;
		if(!project(sample, answer) || !checkOptimality(sample, answer, maximumKkt) ||
			(trial < 300 && !enumerateReference(sample, answer)) ||
			(trial < 1000 && !checkDerivative(sample, answer, derivatives, maximumDerivative)))
		{
			dump(sample, trial);
			return 1;
		}
	}
	std::printf("PASS: 20000 capped patch KKT checks; 300 exhaustive active-set references; %d derivative columns. Max KKT %.3g, curvature finite-difference error %.3g.\n", derivatives, maximumKkt, maximumDerivative);
	return 0;
}
