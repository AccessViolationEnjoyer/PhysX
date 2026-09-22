#ifndef NEWTON_MATH_H
#define NEWTON_MATH_H

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <vector>
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#define NEWTON_AVX2_FMA 1
#endif
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#define NEWTON_X86_SIMD 1
#if defined(NEWTON_AVX2_FMA)
#include <immintrin.h>
#else
#include <emmintrin.h>
#endif
#endif

namespace newton
{
#if defined(_MSC_VER)
#define NEWTON_FORCE_INLINE __forceinline
#define NEWTON_RESTRICT __restrict
#else
#define NEWTON_FORCE_INLINE inline __attribute__((always_inline))
#define NEWTON_RESTRICT __restrict__
#endif

NEWTON_FORCE_INLINE void addScaled6(double* NEWTON_RESTRICT destination, const double* NEWTON_RESTRICT source, double scale)
{
#if defined(NEWTON_AVX2_FMA)
	const __m256d multiplier = _mm256_set1_pd(scale);
	_mm256_storeu_pd(destination, _mm256_fmadd_pd(multiplier, _mm256_loadu_pd(source), _mm256_loadu_pd(destination)));
	_mm_storeu_pd(destination + 4, _mm_fmadd_pd(_mm256_castpd256_pd128(multiplier), _mm_loadu_pd(source + 4), _mm_loadu_pd(destination + 4)));
#elif defined(NEWTON_X86_SIMD)
	const __m128d multiplier = _mm_set1_pd(scale);
	_mm_storeu_pd(destination, _mm_add_pd(_mm_loadu_pd(destination), _mm_mul_pd(multiplier, _mm_loadu_pd(source))));
	_mm_storeu_pd(destination + 2, _mm_add_pd(_mm_loadu_pd(destination + 2), _mm_mul_pd(multiplier, _mm_loadu_pd(source + 2))));
	_mm_storeu_pd(destination + 4, _mm_add_pd(_mm_loadu_pd(destination + 4), _mm_mul_pd(multiplier, _mm_loadu_pd(source + 4))));
#else
	for(int i = 0; i < 6; ++i)
	{
		destination[i] += scale * source[i];
	}
#endif
}

NEWTON_FORCE_INLINE void subtractScaled6(double* NEWTON_RESTRICT destination, const double* NEWTON_RESTRICT source, double scale)
{
#if defined(NEWTON_AVX2_FMA)
	const __m256d multiplier = _mm256_set1_pd(scale);
	_mm256_storeu_pd(destination, _mm256_fnmadd_pd(multiplier, _mm256_loadu_pd(source), _mm256_loadu_pd(destination)));
	_mm_storeu_pd(destination + 4, _mm_fnmadd_pd(_mm256_castpd256_pd128(multiplier), _mm_loadu_pd(source + 4), _mm_loadu_pd(destination + 4)));
#elif defined(NEWTON_X86_SIMD)
	const __m128d multiplier = _mm_set1_pd(scale);
	_mm_storeu_pd(destination, _mm_sub_pd(_mm_loadu_pd(destination), _mm_mul_pd(multiplier, _mm_loadu_pd(source))));
	_mm_storeu_pd(destination + 2, _mm_sub_pd(_mm_loadu_pd(destination + 2), _mm_mul_pd(multiplier, _mm_loadu_pd(source + 2))));
	_mm_storeu_pd(destination + 4, _mm_sub_pd(_mm_loadu_pd(destination + 4), _mm_mul_pd(multiplier, _mm_loadu_pd(source + 4))));
#else
	for(int i = 0; i < 6; ++i)
	{
		destination[i] -= scale * source[i];
	}
#endif
}

NEWTON_FORCE_INLINE void subtractMatrixVector6(double* NEWTON_RESTRICT destination, const double* NEWTON_RESTRICT matrix, const double* NEWTON_RESTRICT vector)
{
#if defined(NEWTON_AVX2_FMA)
	__m256d first = _mm256_loadu_pd(destination);
	__m128d second = _mm_loadu_pd(destination + 4);
	for(int column = 0; column < 6; ++column)
	{
		const __m256d multiplier = _mm256_set1_pd(vector[column]);
		first = _mm256_fnmadd_pd(multiplier, _mm256_loadu_pd(matrix + 6 * column), first);
		second = _mm_fnmadd_pd(_mm256_castpd256_pd128(multiplier), _mm_loadu_pd(matrix + 6 * column + 4), second);
	}
	_mm256_storeu_pd(destination, first);
	_mm_storeu_pd(destination + 4, second);
#elif defined(NEWTON_X86_SIMD)
	__m128d first = _mm_loadu_pd(destination);
	__m128d second = _mm_loadu_pd(destination + 2);
	__m128d third = _mm_loadu_pd(destination + 4);
	for(int column = 0; column < 6; ++column)
	{
		const __m128d multiplier = _mm_set1_pd(vector[column]);
		first = _mm_sub_pd(first, _mm_mul_pd(multiplier, _mm_loadu_pd(matrix + 6 * column)));
		second = _mm_sub_pd(second, _mm_mul_pd(multiplier, _mm_loadu_pd(matrix + 6 * column + 2)));
		third = _mm_sub_pd(third, _mm_mul_pd(multiplier, _mm_loadu_pd(matrix + 6 * column + 4)));
	}
	_mm_storeu_pd(destination, first);
	_mm_storeu_pd(destination + 2, second);
	_mm_storeu_pd(destination + 4, third);
#else
	for(int column = 0; column < 6; ++column)
	{
		for(int row = 0; row < 6; ++row)
		{
			destination[row] -= matrix[6 * column + row] * vector[column];
		}
	}
#endif
}

NEWTON_FORCE_INLINE double dot6(const double* NEWTON_RESTRICT left, const double* NEWTON_RESTRICT right)
{
#if defined(NEWTON_AVX2_FMA)
	const __m256d product = _mm256_mul_pd(_mm256_loadu_pd(left), _mm256_loadu_pd(right));
	const __m128d halves = _mm_add_pd(_mm256_castpd256_pd128(product), _mm256_extractf128_pd(product, 1));
	const __m128d sum = _mm_add_pd(halves, _mm_unpackhi_pd(halves, halves));
	const __m128d end = _mm_mul_pd(_mm_loadu_pd(left + 4), _mm_loadu_pd(right + 4));
	return _mm_cvtsd_f64(sum) + _mm_cvtsd_f64(end) + _mm_cvtsd_f64(_mm_unpackhi_pd(end, end));
#elif defined(NEWTON_X86_SIMD)
	const __m128d first = _mm_mul_pd(_mm_loadu_pd(left), _mm_loadu_pd(right));
	const __m128d second = _mm_mul_pd(_mm_loadu_pd(left + 2), _mm_loadu_pd(right + 2));
	const __m128d third = _mm_mul_pd(_mm_loadu_pd(left + 4), _mm_loadu_pd(right + 4));
	double result = _mm_cvtsd_f64(first);
	result += _mm_cvtsd_f64(_mm_unpackhi_pd(first, first));
	result += _mm_cvtsd_f64(second);
	result += _mm_cvtsd_f64(_mm_unpackhi_pd(second, second));
	result += _mm_cvtsd_f64(third);
	result += _mm_cvtsd_f64(_mm_unpackhi_pd(third, third));
	return result;
#else
	double result = left[0] * right[0];
	for(int i = 1; i < 6; ++i)
	{
		result += left[i] * right[i];
	}
	return result;
#endif
}

NEWTON_FORCE_INLINE void subtractProduct6(double* NEWTON_RESTRICT destination, const double* NEWTON_RESTRICT left, const double* NEWTON_RESTRICT right)
{
#if defined(NEWTON_AVX2_FMA)
	// Retain the complete result so each left column is loaded only once.
	__m256d first0 = _mm256_loadu_pd(destination);
	__m128d second0 = _mm_loadu_pd(destination + 4);
	__m256d first1 = _mm256_loadu_pd(destination + 6);
	__m128d second1 = _mm_loadu_pd(destination + 10);
	__m256d first2 = _mm256_loadu_pd(destination + 12);
	__m128d second2 = _mm_loadu_pd(destination + 16);
	__m256d first3 = _mm256_loadu_pd(destination + 18);
	__m128d second3 = _mm_loadu_pd(destination + 22);
	__m256d first4 = _mm256_loadu_pd(destination + 24);
	__m128d second4 = _mm_loadu_pd(destination + 28);
	__m256d first5 = _mm256_loadu_pd(destination + 30);
	__m128d second5 = _mm_loadu_pd(destination + 34);
	for(int inner = 0; inner < 6; ++inner)
	{
		const __m256d leftFirst = _mm256_loadu_pd(left + 6 * inner);
		const __m128d leftSecond = _mm_loadu_pd(left + 6 * inner + 4);
		__m256d multiplier = _mm256_set1_pd(right[6 * inner]);
		first0 = _mm256_fnmadd_pd(multiplier, leftFirst, first0);
		second0 = _mm_fnmadd_pd(_mm256_castpd256_pd128(multiplier), leftSecond, second0);
		multiplier = _mm256_set1_pd(right[6 * inner + 1]);
		first1 = _mm256_fnmadd_pd(multiplier, leftFirst, first1);
		second1 = _mm_fnmadd_pd(_mm256_castpd256_pd128(multiplier), leftSecond, second1);
		multiplier = _mm256_set1_pd(right[6 * inner + 2]);
		first2 = _mm256_fnmadd_pd(multiplier, leftFirst, first2);
		second2 = _mm_fnmadd_pd(_mm256_castpd256_pd128(multiplier), leftSecond, second2);
		multiplier = _mm256_set1_pd(right[6 * inner + 3]);
		first3 = _mm256_fnmadd_pd(multiplier, leftFirst, first3);
		second3 = _mm_fnmadd_pd(_mm256_castpd256_pd128(multiplier), leftSecond, second3);
		multiplier = _mm256_set1_pd(right[6 * inner + 4]);
		first4 = _mm256_fnmadd_pd(multiplier, leftFirst, first4);
		second4 = _mm_fnmadd_pd(_mm256_castpd256_pd128(multiplier), leftSecond, second4);
		multiplier = _mm256_set1_pd(right[6 * inner + 5]);
		first5 = _mm256_fnmadd_pd(multiplier, leftFirst, first5);
		second5 = _mm_fnmadd_pd(_mm256_castpd256_pd128(multiplier), leftSecond, second5);
	}
	_mm256_storeu_pd(destination, first0);
	_mm_storeu_pd(destination + 4, second0);
	_mm256_storeu_pd(destination + 6, first1);
	_mm_storeu_pd(destination + 10, second1);
	_mm256_storeu_pd(destination + 12, first2);
	_mm_storeu_pd(destination + 16, second2);
	_mm256_storeu_pd(destination + 18, first3);
	_mm_storeu_pd(destination + 22, second3);
	_mm256_storeu_pd(destination + 24, first4);
	_mm_storeu_pd(destination + 28, second4);
	_mm256_storeu_pd(destination + 30, first5);
	_mm_storeu_pd(destination + 34, second5);
#elif defined(NEWTON_X86_SIMD)
	for(int column = 0; column < 6; ++column)
	{
		__m128d first = _mm_loadu_pd(destination + 6 * column);
		__m128d second = _mm_loadu_pd(destination + 6 * column + 2);
		__m128d third = _mm_loadu_pd(destination + 6 * column + 4);
		for(int inner = 0; inner < 6; ++inner)
		{
			const __m128d multiplier = _mm_set1_pd(right[6 * inner + column]);
			first = _mm_sub_pd(first, _mm_mul_pd(multiplier, _mm_loadu_pd(left + 6 * inner)));
			second = _mm_sub_pd(second, _mm_mul_pd(multiplier, _mm_loadu_pd(left + 6 * inner + 2)));
			third = _mm_sub_pd(third, _mm_mul_pd(multiplier, _mm_loadu_pd(left + 6 * inner + 4)));
		}
		_mm_storeu_pd(destination + 6 * column, first);
		_mm_storeu_pd(destination + 6 * column + 2, second);
		_mm_storeu_pd(destination + 6 * column + 4, third);
	}
#else
	for(int column = 0; column < 6; ++column)
	{
		for(int inner = 0; inner < 6; ++inner)
		{
			for(int row = 0; row < 6; ++row)
			{
				destination[6 * column + row] -= left[6 * inner + row] * right[6 * inner + column];
			}
		}
	}
#endif
}

NEWTON_FORCE_INLINE void subtractLowerOuterProduct6(double* destination, const double* vectors)
{
#if defined(NEWTON_AVX2_FMA)
	__m256d column0 = _mm256_loadu_pd(destination);
	__m128d column0End = _mm_loadu_pd(destination + 4);
	__m256d column1 = _mm256_loadu_pd(destination + 7);
	__m128d column1End = _mm_load_sd(destination + 11);
	__m256d column2 = _mm256_loadu_pd(destination + 14);
	__m128d column3 = _mm_loadu_pd(destination + 21);
	__m128d column3End = _mm_load_sd(destination + 23);
	__m128d column4 = _mm_loadu_pd(destination + 28);
	__m128d column5 = _mm_load_sd(destination + 35);
	for(int vector = 0; vector < 6; ++vector)
	{
		const double* value = vectors + 6 * vector;
		const __m256d first = _mm256_loadu_pd(value);
		const __m128d last = _mm_loadu_pd(value + 4);
		column0 = _mm256_fnmadd_pd(_mm256_set1_pd(value[0]), first, column0);
		column0End = _mm_fnmadd_pd(_mm_set1_pd(value[0]), last, column0End);
		column1 = _mm256_fnmadd_pd(_mm256_set1_pd(value[1]), _mm256_loadu_pd(value + 1), column1);
		column1End = _mm_fnmadd_sd(_mm_set_sd(value[1]), _mm_set_sd(value[5]), column1End);
		column2 = _mm256_fnmadd_pd(_mm256_set1_pd(value[2]), _mm256_loadu_pd(value + 2), column2);
		column3 = _mm_fnmadd_pd(_mm_set1_pd(value[3]), _mm_loadu_pd(value + 3), column3);
		column3End = _mm_fnmadd_sd(_mm_set_sd(value[3]), _mm_set_sd(value[5]), column3End);
		column4 = _mm_fnmadd_pd(_mm_set1_pd(value[4]), last, column4);
		column5 = _mm_fnmadd_sd(_mm_set_sd(value[5]), _mm_set_sd(value[5]), column5);
	}
	_mm256_storeu_pd(destination, column0);
	_mm_storeu_pd(destination + 4, column0End);
	_mm256_storeu_pd(destination + 7, column1);
	_mm_store_sd(destination + 11, column1End);
	_mm256_storeu_pd(destination + 14, column2);
	_mm_storeu_pd(destination + 21, column3);
	_mm_store_sd(destination + 23, column3End);
	_mm_storeu_pd(destination + 28, column4);
	_mm_store_sd(destination + 35, column5);
#else
	for(int column = 0; column < 6; ++column)
	{
		for(int row = column; row < 6; ++row)
		{
			double product = 0.0;
			for(int inner = 0; inner < 6; ++inner)
			{
				product += vectors[6 * inner + row] * vectors[6 * inner + column];
			}
			destination[6 * column + row] -= product;
		}
	}
#endif
}

NEWTON_FORCE_INLINE double subtractDot6(double value, const double* left, const double* right)
{
#if defined(NEWTON_AVX2_FMA)
	const __m256d product = _mm256_mul_pd(_mm256_loadu_pd(left), _mm256_loadu_pd(right));
	const __m128d halves = _mm_add_pd(_mm256_castpd256_pd128(product), _mm256_extractf128_pd(product, 1));
	const __m128d sum = _mm_add_pd(halves, _mm_unpackhi_pd(halves, halves));
	const __m128d end = _mm_mul_pd(_mm_loadu_pd(left + 4), _mm_loadu_pd(right + 4));
	value -= _mm_cvtsd_f64(sum);
	value -= _mm_cvtsd_f64(end);
	value -= _mm_cvtsd_f64(_mm_unpackhi_pd(end, end));
	return value;
#elif defined(NEWTON_X86_SIMD)
	const __m128d first = _mm_mul_pd(_mm_loadu_pd(left), _mm_loadu_pd(right));
	const __m128d second = _mm_mul_pd(_mm_loadu_pd(left + 2), _mm_loadu_pd(right + 2));
	const __m128d third = _mm_mul_pd(_mm_loadu_pd(left + 4), _mm_loadu_pd(right + 4));
	value -= _mm_cvtsd_f64(first);
	value -= _mm_cvtsd_f64(_mm_unpackhi_pd(first, first));
	value -= _mm_cvtsd_f64(second);
	value -= _mm_cvtsd_f64(_mm_unpackhi_pd(second, second));
	value -= _mm_cvtsd_f64(third);
	value -= _mm_cvtsd_f64(_mm_unpackhi_pd(third, third));
	return value;
#else
	for(int i = 0; i < 6; ++i)
	{
		value -= left[i] * right[i];
	}
	return value;
#endif
}

NEWTON_FORCE_INLINE void updateCholesky6(double* NEWTON_RESTRICT factor, double* NEWTON_RESTRICT work, double inverseC, double signedSC, double c, double s)
{
#if defined(NEWTON_AVX2_FMA)
	const __m256d inverse = _mm256_set1_pd(inverseC), update = _mm256_set1_pd(signedSC);
	const __m256d cosine = _mm256_set1_pd(c), sine = _mm256_set1_pd(s);
	const __m256d oldWork = _mm256_loadu_pd(work);
	const __m256d nextFactor = _mm256_fmadd_pd(update, oldWork, _mm256_mul_pd(inverse, _mm256_loadu_pd(factor)));
	_mm256_storeu_pd(factor, nextFactor);
	_mm256_storeu_pd(work, _mm256_fmsub_pd(cosine, oldWork, _mm256_mul_pd(sine, nextFactor)));
	const __m128d inverseEnd = _mm256_castpd256_pd128(inverse), updateEnd = _mm256_castpd256_pd128(update);
	const __m128d cosineEnd = _mm256_castpd256_pd128(cosine), sineEnd = _mm256_castpd256_pd128(sine);
	const __m128d oldWorkEnd = _mm_loadu_pd(work + 4);
	const __m128d nextFactorEnd = _mm_fmadd_pd(updateEnd, oldWorkEnd, _mm_mul_pd(inverseEnd, _mm_loadu_pd(factor + 4)));
	_mm_storeu_pd(factor + 4, nextFactorEnd);
	_mm_storeu_pd(work + 4, _mm_fmsub_pd(cosineEnd, oldWorkEnd, _mm_mul_pd(sineEnd, nextFactorEnd)));
#elif defined(NEWTON_X86_SIMD)
	const __m128d inverse = _mm_set1_pd(inverseC), update = _mm_set1_pd(signedSC);
	const __m128d cosine = _mm_set1_pd(c), sine = _mm_set1_pd(s);
	for(int axis = 0; axis < 6; axis += 2)
	{
		const __m128d oldWork = _mm_loadu_pd(work + axis);
		const __m128d nextFactor = _mm_add_pd(_mm_mul_pd(inverse, _mm_loadu_pd(factor + axis)), _mm_mul_pd(update, oldWork));
		_mm_storeu_pd(factor + axis, nextFactor);
		_mm_storeu_pd(work + axis, _mm_sub_pd(_mm_mul_pd(cosine, oldWork), _mm_mul_pd(sine, nextFactor)));
	}
#else
	for(int axis = 0; axis < 6; ++axis)
	{
		factor[axis] = inverseC * factor[axis] + signedSC * work[axis];
		work[axis] = c * work[axis] - s * factor[axis];
	}
#endif
}

template<int Rows, int Columns>
class Matrix
{
public:
	Matrix() {}

	template<int R = Rows, int C = Columns>
	Matrix(double x, double y, double z, typename std::enable_if<R == 3 && C == 1, int>::type = 0)
	{
		m_values[0] = x;
		m_values[1] = y;
		m_values[2] = z;
	}

	double& operator()(int row, int column) { return m_values[column * Rows + row]; }
	double operator()(int row, int column) const { return m_values[column * Rows + row]; }
	double& operator[](int index) { return m_values[index]; }
	double operator[](int index) const { return m_values[index]; }
	double* data() { return m_values; }
	const double* data() const { return m_values; }

	void setZero() { std::fill(m_values, m_values + Rows * Columns, 0.0); }
	void setConstant(double value) { std::fill(m_values, m_values + Rows * Columns, value); }
	void setIdentity()
	{
		setZero();
		const int diagonalCount = Rows < Columns ? Rows : Columns;
		for(int i = 0; i < diagonalCount; ++i)
		{
			(*this)(i, i) = 1.0;
		}
	}
	static Matrix Zero() { Matrix result; result.setZero(); return result; }
	static Matrix Ones()
	{
		Matrix result;
		std::fill(result.m_values, result.m_values + Rows * Columns, 1.0);
		return result;
	}
	static Matrix Unit(int axis)
	{
		Matrix result = Zero();
		result[axis] = 1.0;
		return result;
	}

	Matrix& operator+=(const Matrix& other)
	{
		for(int i = 0; i < Rows * Columns; ++i)
		{
			m_values[i] += other.m_values[i];
		}
		return *this;
	}
	Matrix& operator-=(const Matrix& other)
	{
		for(int i = 0; i < Rows * Columns; ++i)
		{
			m_values[i] -= other.m_values[i];
		}
		return *this;
	}
	Matrix& operator*=(double scale)
	{
		for(int i = 0; i < Rows * Columns; ++i)
		{
			m_values[i] *= scale;
		}
		return *this;
	}
	Matrix& operator/=(double scale)
	{
		const double inverse = 1.0 / scale;
		return *this *= inverse;
	}
	Matrix operator-() const
	{
		Matrix result;
		for(int i = 0; i < Rows * Columns; ++i)
		{
			result.m_values[i] = -m_values[i];
		}
		return result;
	}
	Matrix cwiseProduct(const Matrix& other) const
	{
		Matrix result;
		for(int i = 0; i < Rows * Columns; ++i)
		{
			result.m_values[i] = m_values[i] * other.m_values[i];
		}
		return result;
	}
	Matrix cwiseQuotient(const Matrix& other) const
	{
		Matrix result;
		for(int i = 0; i < Rows * Columns; ++i)
		{
			result.m_values[i] = m_values[i] / other.m_values[i];
		}
		return result;
	}
	Matrix cwiseSqrt() const
	{
		Matrix result;
		for(int i = 0; i < Rows * Columns; ++i)
		{
			result.m_values[i] = std::sqrt(m_values[i]);
		}
		return result;
	}
	Matrix cwiseInverse() const
	{
		Matrix result;
		for(int i = 0; i < Rows * Columns; ++i)
		{
			result.m_values[i] = 1.0 / m_values[i];
		}
		return result;
	}
	Matrix cwiseAbs() const
	{
		Matrix result;
		for(int i = 0; i < Rows * Columns; ++i)
		{
			result.m_values[i] = std::abs(m_values[i]);
		}
		return result;
	}
	Matrix cwiseMax(double limit) const
	{
		Matrix result;
		for(int i = 0; i < Rows * Columns; ++i)
		{
			result.m_values[i] = std::max(m_values[i], limit);
		}
		return result;
	}
	Matrix cwiseMin(double limit) const
	{
		Matrix result;
		for(int i = 0; i < Rows * Columns; ++i)
		{
			result.m_values[i] = std::min(m_values[i], limit);
		}
		return result;
	}
	double dot(const Matrix& other) const
	{
		double result = 0.0;
		for(int i = 0; i < Rows * Columns; ++i)
		{
			result += m_values[i] * other.m_values[i];
		}
		return result;
	}
	double squaredNorm() const { return dot(*this); }
	double norm() const { return std::sqrt(squaredNorm()); }
	template<int R = Rows, int C = Columns>
	typename std::enable_if<R == 3 && C == 1, Matrix>::type cross(const Matrix& other) const
	{
		return Matrix(m_values[1] * other[2] - m_values[2] * other[1], m_values[2] * other[0] - m_values[0] * other[2], m_values[0] * other[1] - m_values[1] * other[0]);
	}
	double maxCoeff() const
	{
		double result = m_values[0];
		for(int i = 1; i < Rows * Columns; ++i)
		{
			result = std::max(result, m_values[i]);
		}
		return result;
	}
	bool equals(const Matrix& other) const
	{
		for(int i = 0; i < Rows * Columns; ++i)
		{
			if(m_values[i] != other.m_values[i])
			{
				return false;
			}
		}
		return true;
	}
	bool isZero(double tolerance) const
	{
		for(int i = 0; i < Rows * Columns; ++i)
		{
			if(std::abs(m_values[i]) > tolerance)
			{
				return false;
			}
		}
		return true;
	}

	template<int R = Rows, int C = Columns>
	typename std::enable_if<C == 1, Matrix<Rows, Rows>>::type asDiagonal() const
	{
		Matrix<Rows, Rows> result = Matrix<Rows, Rows>::Zero();
		for(int i = 0; i < Rows; ++i)
		{
			result(i, i) = m_values[i];
		}
		return result;
	}

	template<int Count>
	Matrix<Count, 1> head() const
	{
		Matrix<Count, 1> result;
		for(int i = 0; i < Count; ++i)
		{
			result[i] = m_values[i];
		}
		return result;
	}

	class Column
	{
	public:
		Column(Matrix& matrix, int column) : m_matrix(matrix), m_column(column) {}
		operator Matrix<Rows, 1>() const
		{
			Matrix<Rows, 1> result;
			for(int row = 0; row < Rows; ++row)
			{
				result[row] = m_matrix(row, m_column);
			}
			return result;
		}
		Column& operator=(const Matrix<Rows, 1>& value)
		{
			for(int row = 0; row < Rows; ++row)
			{
				m_matrix(row, m_column) = value[row];
			}
			return *this;
		}
		Column& operator-=(const Matrix<Rows, 1>& value)
		{
			for(int row = 0; row < Rows; ++row)
			{
				m_matrix(row, m_column) -= value[row];
			}
			return *this;
		}
		Column& operator+=(const Matrix<Rows, 1>& value)
		{
			for(int row = 0; row < Rows; ++row)
			{
				m_matrix(row, m_column) += value[row];
			}
			return *this;
		}
		Column& operator/=(double scale)
		{
			for(int row = 0; row < Rows; ++row)
			{
				m_matrix(row, m_column) /= scale;
			}
			return *this;
		}
		void setZero()
		{
			for(int row = 0; row < Rows; ++row)
			{
				m_matrix(row, m_column) = 0.0;
			}
		}
	private:
		Matrix& m_matrix;
		int m_column;
	};

	class ConstColumn
	{
	public:
		ConstColumn(const Matrix& matrix, int column) : m_matrix(matrix), m_column(column) {}
		operator Matrix<Rows, 1>() const
		{
			Matrix<Rows, 1> result;
			for(int row = 0; row < Rows; ++row)
			{
				result[row] = m_matrix(row, m_column);
			}
			return result;
		}
	private:
		const Matrix& m_matrix;
		int m_column;
	};

	class Row
	{
	public:
		Row(Matrix& matrix, int row) : m_matrix(matrix), m_row(row) {}
		operator Matrix<Columns, 1>() const
		{
			Matrix<Columns, 1> result;
			for(int column = 0; column < Columns; ++column)
			{
				result[column] = m_matrix(m_row, column);
			}
			return result;
		}
		Matrix<Columns, 1> transpose() const { return operator Matrix<Columns, 1>(); }
		double squaredNorm() const
		{
			double result = 0.0;
			for(int column = 0; column < Columns; ++column)
			{
				result += m_matrix(m_row, column) * m_matrix(m_row, column);
			}
			return result;
		}
		Row& operator=(const Matrix<Columns, 1>& value)
		{
			for(int column = 0; column < Columns; ++column)
			{
				m_matrix(m_row, column) = value[column];
			}
			return *this;
		}
		void setZero()
		{
			for(int column = 0; column < Columns; ++column)
			{
				m_matrix(m_row, column) = 0.0;
			}
		}
	private:
		Matrix& m_matrix;
		int m_row;
	};

	class ConstRow
	{
	public:
		ConstRow(const Matrix& matrix, int row) : m_matrix(matrix), m_row(row) {}
		Matrix<Columns, 1> transpose() const
		{
			Matrix<Columns, 1> result;
			for(int column = 0; column < Columns; ++column)
			{
				result[column] = m_matrix(m_row, column);
			}
			return result;
		}
	private:
		const Matrix& m_matrix;
		int m_row;
	};

	Column col(int column) { return Column(*this, column); }
	ConstColumn col(int column) const { return ConstColumn(*this, column); }
	Row row(int row) { return Row(*this, row); }
	ConstRow row(int row) const { return ConstRow(*this, row); }

	Matrix<Columns, Rows> transpose() const
	{
		Matrix<Columns, Rows> result;
		for(int column = 0; column < Columns; ++column)
		{
			for(int row = 0; row < Rows; ++row)
			{
				result(column, row) = (*this)(row, column);
			}
		}
		return result;
	}

private:
	double m_values[Rows * Columns];
};

template<int Rows, int Columns>
inline Matrix<Rows, Columns> operator+(Matrix<Rows, Columns> left, const Matrix<Rows, Columns>& right)
{
	return left += right;
}

template<int Rows, int Columns>
inline Matrix<Rows, Columns> operator-(Matrix<Rows, Columns> left, const Matrix<Rows, Columns>& right)
{
	return left -= right;
}

template<int Rows, int Columns>
inline Matrix<Rows, Columns> operator*(Matrix<Rows, Columns> value, double scale)
{
	return value *= scale;
}

template<int Rows, int Columns>
inline Matrix<Rows, Columns> operator*(double scale, Matrix<Rows, Columns> value)
{
	return value *= scale;
}

template<int Rows, int Columns>
inline Matrix<Rows, Columns> operator/(Matrix<Rows, Columns> value, double scale)
{
	return value /= scale;
}

template<int Rows, int Inner, int Columns>
inline Matrix<Rows, Columns> operator*(const Matrix<Rows, Inner>& left, const Matrix<Inner, Columns>& right)
{
	Matrix<Rows, Columns> result;
	result.setZero();
	for(int column = 0; column < Columns; ++column)
	{
		for(int inner = 0; inner < Inner; ++inner)
		{
			const double value = right(inner, column);
			for(int row = 0; row < Rows; ++row)
			{
				result(row, column) += left(row, inner) * value;
			}
		}
	}
	return result;
}

template<int Size>
inline Matrix<Size, Size> diagonalMatrix(const Matrix<Size, 1>& diagonal)
{
	Matrix<Size, Size> result = Matrix<Size, Size>::Zero();
	for(int i = 0; i < Size; ++i)
	{
		result(i, i) = diagonal[i];
	}
	return result;
}

template<int Size>
inline Matrix<Size, 1> loadVector(const double* values)
{
	Matrix<Size, 1> result;
	for(int i = 0; i < Size; ++i)
	{
		result[i] = values[i];
	}
	return result;
}

template<int Size>
inline void storeVector(double* values, const Matrix<Size, 1>& vector)
{
	for(int i = 0; i < Size; ++i)
	{
		values[i] = vector[i];
	}
}

class VectorStorage
{
public:
	VectorStorage() {}
	explicit VectorStorage(int size) { resize(size); }
	VectorStorage(const VectorStorage&) = default;
	VectorStorage(VectorStorage&&) noexcept = default;
	static VectorStorage Zero(int size)
	{
		VectorStorage result(size);
		result.setZero();
		return result;
	}
	void resize(int size)
	{
		const std::uint32_t requestedSize = std::uint32_t(size);
		const std::uint32_t currentCapacity = std::uint32_t(m_values.capacity());
		if(requestedSize > currentCapacity)
		{
			m_values.reserve(std::max(requestedSize, 2u * currentCapacity));
		}
		m_values.resize(requestedSize);
	}
	int size() const { return int(m_values.size()); }
	double* data() { return m_values.data(); }
	const double* data() const { return m_values.data(); }
	double& operator[](int index) { return m_values[std::uint32_t(index)]; }
	double operator[](int index) const { return m_values[std::uint32_t(index)]; }
	std::vector<double>::iterator begin() { return m_values.begin(); }
	std::vector<double>::iterator end() { return m_values.end(); }
	std::vector<double>::const_iterator begin() const { return m_values.begin(); }
	std::vector<double>::const_iterator end() const { return m_values.end(); }
	void setZero() { std::fill(m_values.begin(), m_values.end(), 0.0); }
	void setZero(int size) { resize(size); setZero(); }
	void setOnes() { std::fill(m_values.begin(), m_values.end(), 1.0); }
	void setConstant(double value) { std::fill(m_values.begin(), m_values.end(), value); }
	VectorStorage& operator+=(const VectorStorage& other)
	{
		const std::uint32_t count = std::uint32_t(m_values.size());
		for(std::uint32_t i = 0; i < count; ++i)
		{
			m_values[i] += other.m_values[i];
		}
		return *this;
	}
	VectorStorage& operator*=(double scale)
	{
		const std::uint32_t count = std::uint32_t(m_values.size());
		for(std::uint32_t i = 0; i < count; ++i)
		{
			m_values[i] *= scale;
		}
		return *this;
	}
	void swap(VectorStorage& other) noexcept { m_values.swap(other.m_values); }
	double dot(const VectorStorage& other) const
	{
		double result = 0.0;
		const std::uint32_t count = std::uint32_t(m_values.size());
		for(std::uint32_t i = 0; i < count; ++i)
		{
			result += m_values[i] * other.m_values[i];
		}
		return result;
	}
	double squaredNorm() const { return dot(*this); }
	double sum() const
	{
		double result = 0.0;
		const std::uint32_t count = std::uint32_t(m_values.size());
		for(std::uint32_t i = 0; i < count; ++i)
		{
			result += m_values[i];
		}
		return result;
	}
	double minCoeff() const
	{
		double result = m_values[0];
		const std::uint32_t count = std::uint32_t(m_values.size());
		for(std::uint32_t i = 1; i < count; ++i)
		{
			result = std::min(result, m_values[i]);
		}
		return result;
	}
	double infinityNorm() const
	{
		double result = 0.0;
		const std::uint32_t count = std::uint32_t(m_values.size());
		for(std::uint32_t i = 0; i < count; ++i)
		{
			result = std::max(result, std::abs(m_values[i]));
		}
		return result;
	}
	template<int Count>
	Matrix<Count, 1> segment(int first) const { return loadVector<Count>(m_values.data() + first); }
	VectorStorage& operator=(const VectorStorage&) = default;
	VectorStorage& operator=(VectorStorage&&) noexcept = default;
private:
	std::vector<double> m_values;
};

class SparseStorage
{
public:
	SparseStorage() : m_rows(0), m_columns(0) {}
	SparseStorage(int rows, int columns) : m_rows(0), m_columns(0) { resize(rows, columns); }
	void resize(int rows, int columns)
	{
		m_rows = rows;
		m_columns = columns;
		const std::uint32_t requestedSize = std::uint32_t(columns + 1);
		const std::uint32_t currentCapacity = std::uint32_t(m_outer.capacity());
		if(requestedSize > currentCapacity)
		{
			m_outer.reserve(std::max(requestedSize, 2u * currentCapacity));
		}
		m_outer.resize(requestedSize);
		std::fill(m_outer.begin(), m_outer.end(), 0);
		m_inner.clear();
		m_values.clear();
	}
	void reserve(int entries)
	{
		const std::uint32_t requestedSize = std::uint32_t(entries);
		const std::uint32_t currentCapacity = std::uint32_t(m_inner.capacity());
		if(requestedSize > currentCapacity)
		{
			const std::uint32_t capacity = std::max(requestedSize, 2u * currentCapacity);
			m_inner.reserve(capacity);
			m_values.reserve(capacity);
		}
	}
	void resizeNonZeros(int entries)
	{
		reserve(entries);
		m_inner.resize(std::uint32_t(entries));
		m_values.resize(std::uint32_t(entries));
	}
	void setZero()
	{
		std::fill(m_outer.begin(), m_outer.end(), 0);
		m_inner.clear();
		m_values.clear();
	}
	int rows() const { return m_rows; }
	int cols() const { return m_columns; }
	int outerSize() const { return m_columns; }
	int nonZeros() const { return int(m_values.size()); }
	int* outerIndexPtr() { return m_outer.data(); }
	const int* outerIndexPtr() const { return m_outer.data(); }
	int* innerIndexPtr() { return m_inner.data(); }
	const int* innerIndexPtr() const { return m_inner.data(); }
	double* valuePtr() { return m_values.data(); }
	const double* valuePtr() const { return m_values.data(); }

	class InnerIterator
	{
	public:
		InnerIterator(const SparseStorage& matrix, int column)
			: m_matrix(matrix), m_entry(matrix.m_outer[std::uint32_t(column)]), m_end(matrix.m_outer[std::uint32_t(column + 1)]) {}
		operator bool() const { return m_entry < m_end; }
		InnerIterator& operator++() { ++m_entry; return *this; }
		int index() const { return m_matrix.m_inner[std::uint32_t(m_entry)]; }
		int row() const { return index(); }
		double value() const { return m_matrix.m_values[std::uint32_t(m_entry)]; }
	private:
		const SparseStorage& m_matrix;
		int m_entry;
		int m_end;
	};

private:
	int m_rows;
	int m_columns;
	std::vector<int> m_outer;
	std::vector<int> m_inner;
	std::vector<double> m_values;
};

inline bool cholesky6(const Matrix<6, 6>& input, Matrix<6, 6>& lower)
{
	lower.setZero();
	for(int column = 0; column < 6; ++column)
	{
		double diagonal = input(column, column);
		for(int inner = 0; inner < column; ++inner)
		{
			diagonal -= lower(column, inner) * lower(column, inner);
		}
		if(!(diagonal > 0.0))
		{
			return false;
		}
		lower(column, column) = std::sqrt(diagonal);
		const double inverse = 1.0 / lower(column, column);
		for(int row = column + 1; row < 6; ++row)
		{
			double value = input(row, column);
			for(int inner = 0; inner < column; ++inner)
			{
				value -= lower(row, inner) * lower(column, inner);
			}
			lower(row, column) = value * inverse;
		}
	}
	return true;
}

// Jacobi diagonalization of a small symmetric matrix. Eigenvalues are sorted
// ascending so rank updates remain deterministic across implementations.
template<int Size>
inline void symmetricEigen(const Matrix<Size, Size>& input, Matrix<Size, 1>& values, Matrix<Size, Size>& vectors)
{
	Matrix<Size, Size> matrix = input;
	vectors.setZero();
	for(int i = 0; i < Size; ++i)
	{
		vectors(i, i) = 1.0;
	}
	for(int sweep = 0; sweep < 16; ++sweep)
	{
		int p = 0, q = 1;
		double largest = 0.0;
		for(int column = 0; column < Size; ++column)
		{
			for(int row = column + 1; row < Size; ++row)
			{
				if(std::abs(matrix(row, column)) > largest)
				{
					largest = std::abs(matrix(row, column));
					p = column;
					q = row;
				}
			}
		}
		if(largest <= 1.0e-15 * std::max(1.0, std::max(std::abs(matrix(p, p)), std::abs(matrix(q, q)))))
		{
			break;
		}
		const double app = matrix(p, p), aqq = matrix(q, q), apq = matrix(q, p);
		const double tau = (aqq - app) / (2.0 * apq);
		const double tangent = (tau >= 0.0 ? 1.0 : -1.0) /
			(std::abs(tau) + std::sqrt(1.0 + tau * tau));
		const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
		const double sine = tangent * cosine;
		for(int k = 0; k < Size; ++k)
		{
			if(k != p && k != q)
			{
				const double mkp = matrix(std::max(k, p), std::min(k, p));
				const double mkq = matrix(std::max(k, q), std::min(k, q));
				const double first = cosine * mkp - sine * mkq;
				const double second = sine * mkp + cosine * mkq;
				matrix(std::max(k, p), std::min(k, p)) = first;
				matrix(std::max(k, q), std::min(k, q)) = second;
			}
		}
		matrix(p, p) = app - tangent * apq;
		matrix(q, q) = aqq + tangent * apq;
		matrix(q, p) = 0.0;
		for(int k = 0; k < Size; ++k)
		{
			const double vkp = vectors(k, p), vkq = vectors(k, q);
			vectors(k, p) = cosine * vkp - sine * vkq;
			vectors(k, q) = sine * vkp + cosine * vkq;
		}
	}
	for(int i = 0; i < Size; ++i)
	{
		values[i] = matrix(i, i);
	}
	for(int i = 0; i < Size; ++i)
	{
		for(int j = i + 1; j < Size; ++j)
		{
			if(values[j] < values[i])
			{
				std::swap(values[i], values[j]);
				for(int row = 0; row < Size; ++row)
				{
					std::swap(vectors(row, i), vectors(row, j));
				}
			}
		}
	}
}
}

#endif
