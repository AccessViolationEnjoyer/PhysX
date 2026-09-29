#ifndef ANVIL_MATH_H
#define ANVIL_MATH_H

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <vector>
// ANVIL_NO_SIMD selects the scalar reference kernels.
// WebAssembly never takes the AVX2 path: Emscripten emulates it more slowly than SIMD128.
#if !defined(ANVIL_NO_SIMD) && !defined(__wasm__) && defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#define ANVIL_AVX2_FMA 1
#endif
#if !defined(ANVIL_NO_SIMD) && (defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__))
#define ANVIL_X86_SIMD 1
#if defined(ANVIL_AVX2_FMA)
#include <immintrin.h>
#else
#include <emmintrin.h>
#endif
#endif
// WebAssembly builds use 128-bit SIMD (Emscripten -msimd128). With -mrelaxed-simd,
// multiply-adds may fuse, as they do in AVX2 builds.
#if !defined(ANVIL_NO_SIMD) && defined(__wasm_simd128__)
#define ANVIL_WASM_SIMD 1
#include <wasm_simd128.h>
#endif
// Kernels without AVX2 use pairs of doubles from SSE2 or WebAssembly SIMD.
#if defined(ANVIL_X86_SIMD) || defined(ANVIL_WASM_SIMD)
#define ANVIL_SIMD128 1
#endif

namespace anvil
{
#if defined(_MSC_VER)
#define ANVIL_FORCE_INLINE __forceinline
#define ANVIL_RESTRICT __restrict
#else
#define ANVIL_FORCE_INLINE inline __attribute__((always_inline))
#define ANVIL_RESTRICT __restrict__
#endif
// Fully unrolls the next loop, so small fixed arrays stay in registers under Clang, which
// otherwise leaves short nested loops rolled in WebAssembly.
#if defined(__clang__)
#define ANVIL_UNROLL _Pragma("clang loop unroll(full)")
#else
#define ANVIL_UNROLL
#endif

#if defined(ANVIL_SIMD128)
// Two-double vector operations shared by the SSE2 and WebAssembly kernels.
namespace simd
{
#if defined(ANVIL_WASM_SIMD)
typedef v128_t Double2;
ANVIL_FORCE_INLINE Double2 load(const double* values) { return wasm_v128_load(values); }
ANVIL_FORCE_INLINE void store(double* values, Double2 value) { wasm_v128_store(values, value); }
ANVIL_FORCE_INLINE Double2 splat(double value) { return wasm_f64x2_splat(value); }
ANVIL_FORCE_INLINE Double2 zero() { return wasm_f64x2_const(0.0, 0.0); }
ANVIL_FORCE_INLINE Double2 make(double low, double high) { return wasm_f64x2_make(low, high); }
ANVIL_FORCE_INLINE Double2 absolute(Double2 value) { return wasm_f64x2_abs(value); }
ANVIL_FORCE_INLINE Double2 add(Double2 left, Double2 right) { return wasm_f64x2_add(left, right); }
ANVIL_FORCE_INLINE Double2 subtract(Double2 left, Double2 right) { return wasm_f64x2_sub(left, right); }
ANVIL_FORCE_INLINE Double2 multiply(Double2 left, Double2 right) { return wasm_f64x2_mul(left, right); }
ANVIL_FORCE_INLINE Double2 divide(Double2 left, Double2 right) { return wasm_f64x2_div(left, right); }
ANVIL_FORCE_INLINE Double2 squareRoot(Double2 value) { return wasm_f64x2_sqrt(value); }
// Pseudo-maximum: left < right ? right : left, which matches SSE2 for ordered values.
ANVIL_FORCE_INLINE Double2 maximum(Double2 left, Double2 right) { return wasm_f64x2_pmax(left, right); }
// Pseudo-minimum: right < left ? right : left, which matches SSE2 for ordered values.
ANVIL_FORCE_INLINE Double2 minimum(Double2 left, Double2 right) { return wasm_f64x2_pmin(left, right); }
ANVIL_FORCE_INLINE Double2 greater(Double2 left, Double2 right) { return wasm_f64x2_gt(left, right); }
ANVIL_FORCE_INLINE Double2 greaterEqual(Double2 left, Double2 right) { return wasm_f64x2_ge(left, right); }
ANVIL_FORCE_INLINE Double2 less(Double2 left, Double2 right) { return wasm_f64x2_lt(left, right); }
ANVIL_FORCE_INLINE Double2 notEqual(Double2 left, Double2 right) { return wasm_f64x2_ne(left, right); }
ANVIL_FORCE_INLINE Double2 bitAnd(Double2 left, Double2 right) { return wasm_v128_and(left, right); }
ANVIL_FORCE_INLINE Double2 bitOr(Double2 left, Double2 right) { return wasm_v128_or(left, right); }
ANVIL_FORCE_INLINE int mask(Double2 value) { return int(wasm_i64x2_bitmask(value)); }
ANVIL_FORCE_INLINE bool any(Double2 value) { return wasm_v128_any_true(value); }
ANVIL_FORCE_INLINE double low(Double2 value) { return wasm_f64x2_extract_lane(value, 0); }
ANVIL_FORCE_INLINE double high(Double2 value) { return wasm_f64x2_extract_lane(value, 1); }
#if defined(__wasm_relaxed_simd__)
ANVIL_FORCE_INLINE Double2 multiplyAdd(Double2 left, Double2 right, Double2 addend) { return wasm_f64x2_relaxed_madd(left, right, addend); }
ANVIL_FORCE_INLINE Double2 negativeMultiplyAdd(Double2 left, Double2 right, Double2 addend) { return wasm_f64x2_relaxed_nmadd(left, right, addend); }
#else
ANVIL_FORCE_INLINE Double2 multiplyAdd(Double2 left, Double2 right, Double2 addend) { return wasm_f64x2_add(wasm_f64x2_mul(left, right), addend); }
ANVIL_FORCE_INLINE Double2 negativeMultiplyAdd(Double2 left, Double2 right, Double2 addend) { return wasm_f64x2_sub(addend, wasm_f64x2_mul(left, right)); }
#endif
typedef v128_t Float4;
ANVIL_FORCE_INLINE Float4 loadFloat(const float* values) { return wasm_v128_load(values); }
ANVIL_FORCE_INLINE void storeFloat(float* values, Float4 value) { wasm_v128_store(values, value); }
ANVIL_FORCE_INLINE Float4 splatFloat(float value) { return wasm_f32x4_splat(value); }
ANVIL_FORCE_INLINE Float4 multiplyFloat(Float4 left, Float4 right) { return wasm_f32x4_mul(left, right); }
#if defined(__wasm_relaxed_simd__)
ANVIL_FORCE_INLINE Float4 negativeMultiplyAddFloat(Float4 left, Float4 right, Float4 addend) { return wasm_f32x4_relaxed_nmadd(left, right, addend); }
#else
ANVIL_FORCE_INLINE Float4 negativeMultiplyAddFloat(Float4 left, Float4 right, Float4 addend) { return wasm_f32x4_sub(addend, wasm_f32x4_mul(left, right)); }
#endif
// Lanes 0-1 and 2-3 converted to doubles.
ANVIL_FORCE_INLINE Double2 promoteLow(Float4 value) { return wasm_f64x2_promote_low_f32x4(value); }
ANVIL_FORCE_INLINE Double2 promoteHigh(Float4 value) { return wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(value, value, 2, 3, 2, 3)); }
// Two doubles as the low floats of a vector whose high floats are zero, and four doubles as floats.
ANVIL_FORCE_INLINE Float4 demoteLow(Double2 value) { return wasm_f32x4_demote_f64x2_zero(value); }
ANVIL_FORCE_INLINE Float4 demote(Double2 low, Double2 high) { return wasm_i32x4_shuffle(demoteLow(low), demoteLow(high), 0, 1, 4, 5); }
// Two floats as the low lanes of a vector whose high lanes are zero, or that repeat them.
ANVIL_FORCE_INLINE Float4 loadFloatPair(const float* values) { return wasm_v128_load64_zero(values); }
ANVIL_FORCE_INLINE Float4 loadFloatPairTwice(const float* values) { return wasm_v128_load64_splat(values); }
ANVIL_FORCE_INLINE void storeFloatPair(float* values, Float4 value) { wasm_v128_store64_lane(values, value, 0); }
// The low two lanes of each vector.
ANVIL_FORCE_INLINE Float4 joinLowFloat(Float4 low, Float4 high) { return wasm_i32x4_shuffle(low, high, 0, 1, 4, 5); }
#else
typedef __m128d Double2;
ANVIL_FORCE_INLINE Double2 load(const double* values) { return _mm_loadu_pd(values); }
ANVIL_FORCE_INLINE void store(double* values, Double2 value) { _mm_storeu_pd(values, value); }
ANVIL_FORCE_INLINE Double2 splat(double value) { return _mm_set1_pd(value); }
ANVIL_FORCE_INLINE Double2 zero() { return _mm_setzero_pd(); }
ANVIL_FORCE_INLINE Double2 make(double low, double high) { return _mm_set_pd(high, low); }
ANVIL_FORCE_INLINE Double2 absolute(Double2 value) { return _mm_andnot_pd(_mm_set1_pd(-0.0), value); }
ANVIL_FORCE_INLINE Double2 add(Double2 left, Double2 right) { return _mm_add_pd(left, right); }
ANVIL_FORCE_INLINE Double2 subtract(Double2 left, Double2 right) { return _mm_sub_pd(left, right); }
ANVIL_FORCE_INLINE Double2 multiply(Double2 left, Double2 right) { return _mm_mul_pd(left, right); }
ANVIL_FORCE_INLINE Double2 divide(Double2 left, Double2 right) { return _mm_div_pd(left, right); }
ANVIL_FORCE_INLINE Double2 squareRoot(Double2 value) { return _mm_sqrt_pd(value); }
ANVIL_FORCE_INLINE Double2 maximum(Double2 left, Double2 right) { return _mm_max_pd(left, right); }
ANVIL_FORCE_INLINE Double2 minimum(Double2 left, Double2 right) { return _mm_min_pd(left, right); }
ANVIL_FORCE_INLINE Double2 greater(Double2 left, Double2 right) { return _mm_cmpgt_pd(left, right); }
ANVIL_FORCE_INLINE Double2 greaterEqual(Double2 left, Double2 right) { return _mm_cmpge_pd(left, right); }
ANVIL_FORCE_INLINE Double2 less(Double2 left, Double2 right) { return _mm_cmplt_pd(left, right); }
ANVIL_FORCE_INLINE Double2 notEqual(Double2 left, Double2 right) { return _mm_cmpneq_pd(left, right); }
ANVIL_FORCE_INLINE Double2 bitAnd(Double2 left, Double2 right) { return _mm_and_pd(left, right); }
ANVIL_FORCE_INLINE Double2 bitOr(Double2 left, Double2 right) { return _mm_or_pd(left, right); }
ANVIL_FORCE_INLINE int mask(Double2 value) { return _mm_movemask_pd(value); }
ANVIL_FORCE_INLINE bool any(Double2 value) { return _mm_movemask_pd(value) != 0; }
ANVIL_FORCE_INLINE double low(Double2 value) { return _mm_cvtsd_f64(value); }
ANVIL_FORCE_INLINE double high(Double2 value) { return _mm_cvtsd_f64(_mm_unpackhi_pd(value, value)); }
ANVIL_FORCE_INLINE Double2 multiplyAdd(Double2 left, Double2 right, Double2 addend) { return _mm_add_pd(_mm_mul_pd(left, right), addend); }
ANVIL_FORCE_INLINE Double2 negativeMultiplyAdd(Double2 left, Double2 right, Double2 addend) { return _mm_sub_pd(addend, _mm_mul_pd(left, right)); }
typedef __m128 Float4;
ANVIL_FORCE_INLINE Float4 loadFloat(const float* values) { return _mm_loadu_ps(values); }
ANVIL_FORCE_INLINE void storeFloat(float* values, Float4 value) { _mm_storeu_ps(values, value); }
ANVIL_FORCE_INLINE Float4 splatFloat(float value) { return _mm_set1_ps(value); }
ANVIL_FORCE_INLINE Float4 multiplyFloat(Float4 left, Float4 right) { return _mm_mul_ps(left, right); }
ANVIL_FORCE_INLINE Float4 negativeMultiplyAddFloat(Float4 left, Float4 right, Float4 addend) { return _mm_sub_ps(addend, _mm_mul_ps(left, right)); }
ANVIL_FORCE_INLINE Double2 promoteLow(Float4 value) { return _mm_cvtps_pd(value); }
ANVIL_FORCE_INLINE Double2 promoteHigh(Float4 value) { return _mm_cvtps_pd(_mm_movehl_ps(value, value)); }
ANVIL_FORCE_INLINE Float4 demoteLow(Double2 value) { return _mm_cvtpd_ps(value); }
ANVIL_FORCE_INLINE Float4 demote(Double2 low, Double2 high) { return _mm_movelh_ps(_mm_cvtpd_ps(low), _mm_cvtpd_ps(high)); }
ANVIL_FORCE_INLINE Float4 loadFloatPair(const float* values) { return _mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(values))); }
ANVIL_FORCE_INLINE Float4 loadFloatPairTwice(const float* values) { const Float4 pair = loadFloatPair(values); return _mm_movelh_ps(pair, pair); }
ANVIL_FORCE_INLINE void storeFloatPair(float* values, Float4 value) { _mm_storel_epi64(reinterpret_cast<__m128i*>(values), _mm_castps_si128(value)); }
ANVIL_FORCE_INLINE Float4 joinLowFloat(Float4 low, Float4 high) { return _mm_movelh_ps(low, high); }
#endif
ANVIL_FORCE_INLINE double sum(Double2 value) { return low(value) + high(value); }
}
#endif

ANVIL_FORCE_INLINE void addScaled6(double* ANVIL_RESTRICT destination, const double* ANVIL_RESTRICT source, double scale)
{
#if defined(ANVIL_AVX2_FMA)
	const __m256d multiplier = _mm256_set1_pd(scale);
	_mm256_storeu_pd(destination, _mm256_fmadd_pd(multiplier, _mm256_loadu_pd(source), _mm256_loadu_pd(destination)));
	_mm_storeu_pd(destination + 4, _mm_fmadd_pd(_mm256_castpd256_pd128(multiplier), _mm_loadu_pd(source + 4), _mm_loadu_pd(destination + 4)));
#elif defined(ANVIL_SIMD128)
	const simd::Double2 multiplier = simd::splat(scale);
	simd::store(destination, simd::multiplyAdd(multiplier, simd::load(source), simd::load(destination)));
	simd::store(destination + 2, simd::multiplyAdd(multiplier, simd::load(source + 2), simd::load(destination + 2)));
	simd::store(destination + 4, simd::multiplyAdd(multiplier, simd::load(source + 4), simd::load(destination + 4)));
#else
	for(int i = 0; i < 6; ++i)
	{
		destination[i] += scale * source[i];
	}
#endif
}

ANVIL_FORCE_INLINE void subtractScaled6(double* ANVIL_RESTRICT destination, const double* ANVIL_RESTRICT source, double scale)
{
#if defined(ANVIL_AVX2_FMA)
	const __m256d multiplier = _mm256_set1_pd(scale);
	_mm256_storeu_pd(destination, _mm256_fnmadd_pd(multiplier, _mm256_loadu_pd(source), _mm256_loadu_pd(destination)));
	_mm_storeu_pd(destination + 4, _mm_fnmadd_pd(_mm256_castpd256_pd128(multiplier), _mm_loadu_pd(source + 4), _mm_loadu_pd(destination + 4)));
#elif defined(ANVIL_SIMD128)
	const simd::Double2 multiplier = simd::splat(scale);
	simd::store(destination, simd::negativeMultiplyAdd(multiplier, simd::load(source), simd::load(destination)));
	simd::store(destination + 2, simd::negativeMultiplyAdd(multiplier, simd::load(source + 2), simd::load(destination + 2)));
	simd::store(destination + 4, simd::negativeMultiplyAdd(multiplier, simd::load(source + 4), simd::load(destination + 4)));
#else
	for(int i = 0; i < 6; ++i)
	{
		destination[i] -= scale * source[i];
	}
#endif
}

ANVIL_FORCE_INLINE void subtractMatrixVector6(double* ANVIL_RESTRICT destination, const double* ANVIL_RESTRICT matrix, const double* ANVIL_RESTRICT vector)
{
#if defined(ANVIL_AVX2_FMA)
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
#elif defined(ANVIL_SIMD128)
	simd::Double2 first = simd::load(destination);
	simd::Double2 second = simd::load(destination + 2);
	simd::Double2 third = simd::load(destination + 4);
	for(int column = 0; column < 6; ++column)
	{
		const simd::Double2 multiplier = simd::splat(vector[column]);
		first = simd::negativeMultiplyAdd(multiplier, simd::load(matrix + 6 * column), first);
		second = simd::negativeMultiplyAdd(multiplier, simd::load(matrix + 6 * column + 2), second);
		third = simd::negativeMultiplyAdd(multiplier, simd::load(matrix + 6 * column + 4), third);
	}
	simd::store(destination, first);
	simd::store(destination + 2, second);
	simd::store(destination + 4, third);
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

ANVIL_FORCE_INLINE double dot6(const double* ANVIL_RESTRICT left, const double* ANVIL_RESTRICT right)
{
#if defined(ANVIL_AVX2_FMA)
	const __m256d product = _mm256_mul_pd(_mm256_loadu_pd(left), _mm256_loadu_pd(right));
	const __m128d halves = _mm_add_pd(_mm256_castpd256_pd128(product), _mm256_extractf128_pd(product, 1));
	const __m128d sum = _mm_add_pd(halves, _mm_unpackhi_pd(halves, halves));
	const __m128d end = _mm_mul_pd(_mm_loadu_pd(left + 4), _mm_loadu_pd(right + 4));
	return _mm_cvtsd_f64(sum) + _mm_cvtsd_f64(end) + _mm_cvtsd_f64(_mm_unpackhi_pd(end, end));
#elif defined(ANVIL_SIMD128)
	const simd::Double2 first = simd::multiply(simd::load(left), simd::load(right));
	const simd::Double2 second = simd::multiplyAdd(simd::load(left + 2), simd::load(right + 2), first);
	return simd::sum(simd::multiplyAdd(simd::load(left + 4), simd::load(right + 4), second));
#else
	double result = left[0] * right[0];
	for(int i = 1; i < 6; ++i)
	{
		result += left[i] * right[i];
	}
	return result;
#endif
}

// Number of entries that compare unequal to zero.
ANVIL_FORCE_INLINE int nonzeroCount6(const double* values)
{
#if defined(ANVIL_AVX2_FMA)
	const int head = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(values), _mm256_setzero_pd(), _CMP_NEQ_UQ));
	const int tail = _mm_movemask_pd(_mm_cmp_pd(_mm_loadu_pd(values + 4), _mm_setzero_pd(), _CMP_NEQ_UQ));
	return int(_mm_popcnt_u32(unsigned(head | (tail << 4))));
#elif defined(ANVIL_SIMD128)
	const simd::Double2 zero = simd::zero();
	int count = 0;
	for(int i = 0; i < 6; i += 2)
	{
		const int nonzero = simd::mask(simd::notEqual(simd::load(values + i), zero));
		count += (nonzero & 1) + (nonzero >> 1);
	}
	return count;
#else
	int count = 0;
	for(int i = 0; i < 6; ++i)
	{
		count += values[i] != 0.0;
	}
	return count;
#endif
}

ANVIL_FORCE_INLINE double dot6Pair(const double* ANVIL_RESTRICT left0, const double* ANVIL_RESTRICT right0,
									const double* ANVIL_RESTRICT left1, const double* ANVIL_RESTRICT right1)
{
#if defined(ANVIL_AVX2_FMA)
	const __m256d first = _mm256_fmadd_pd(_mm256_loadu_pd(left1), _mm256_loadu_pd(right1),
									_mm256_mul_pd(_mm256_loadu_pd(left0), _mm256_loadu_pd(right0)));
	const __m128d end = _mm_fmadd_pd(_mm_loadu_pd(left1 + 4), _mm_loadu_pd(right1 + 4),
								_mm_mul_pd(_mm_loadu_pd(left0 + 4), _mm_loadu_pd(right0 + 4)));
	const __m128d halves = _mm_add_pd(_mm256_castpd256_pd128(first), _mm256_extractf128_pd(first, 1));
	const __m128d sum = _mm_add_pd(halves, _mm_unpackhi_pd(halves, halves));
	return _mm_cvtsd_f64(sum) + _mm_cvtsd_f64(end) + _mm_cvtsd_f64(_mm_unpackhi_pd(end, end));
#elif defined(ANVIL_SIMD128)
	simd::Double2 sum = simd::multiplyAdd(simd::load(left1), simd::load(right1), simd::multiply(simd::load(left0), simd::load(right0)));
	sum = simd::multiplyAdd(simd::load(left0 + 2), simd::load(right0 + 2), sum);
	sum = simd::multiplyAdd(simd::load(left1 + 2), simd::load(right1 + 2), sum);
	sum = simd::multiplyAdd(simd::load(left0 + 4), simd::load(right0 + 4), sum);
	sum = simd::multiplyAdd(simd::load(left1 + 4), simd::load(right1 + 4), sum);
	return simd::sum(sum);
#else
	return dot6(left0, right0) + dot6(left1, right1);
#endif
}

ANVIL_FORCE_INLINE void subtractProduct6(double* ANVIL_RESTRICT destination, const double* ANVIL_RESTRICT left, const double* ANVIL_RESTRICT right)
{
#if defined(ANVIL_AVX2_FMA)
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
#elif defined(ANVIL_SIMD128)
	// Three result columns share each loaded left column: nine accumulators fit the
	// sixteen SSE2 registers and V8's WebAssembly register allocation.
	for(int column = 0; column < 6; column += 3)
	{
		double* target = destination + 6 * column;
		simd::Double2 first0 = simd::load(target), second0 = simd::load(target + 2), third0 = simd::load(target + 4);
		simd::Double2 first1 = simd::load(target + 6), second1 = simd::load(target + 8), third1 = simd::load(target + 10);
		simd::Double2 first2 = simd::load(target + 12), second2 = simd::load(target + 14), third2 = simd::load(target + 16);
		for(int inner = 0; inner < 6; ++inner)
		{
			const double* source = left + 6 * inner;
			const simd::Double2 leftFirst = simd::load(source), leftSecond = simd::load(source + 2), leftThird = simd::load(source + 4);
			simd::Double2 multiplier = simd::splat(right[6 * inner + column]);
			first0 = simd::negativeMultiplyAdd(multiplier, leftFirst, first0);
			second0 = simd::negativeMultiplyAdd(multiplier, leftSecond, second0);
			third0 = simd::negativeMultiplyAdd(multiplier, leftThird, third0);
			multiplier = simd::splat(right[6 * inner + column + 1]);
			first1 = simd::negativeMultiplyAdd(multiplier, leftFirst, first1);
			second1 = simd::negativeMultiplyAdd(multiplier, leftSecond, second1);
			third1 = simd::negativeMultiplyAdd(multiplier, leftThird, third1);
			multiplier = simd::splat(right[6 * inner + column + 2]);
			first2 = simd::negativeMultiplyAdd(multiplier, leftFirst, first2);
			second2 = simd::negativeMultiplyAdd(multiplier, leftSecond, second2);
			third2 = simd::negativeMultiplyAdd(multiplier, leftThird, third2);
		}
		simd::store(target, first0);
		simd::store(target + 2, second0);
		simd::store(target + 4, third0);
		simd::store(target + 6, first1);
		simd::store(target + 8, second1);
		simd::store(target + 10, third1);
		simd::store(target + 12, first2);
		simd::store(target + 14, second2);
		simd::store(target + 16, third2);
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

ANVIL_FORCE_INLINE void subtractLowerOuterProduct6(double* destination, const double* vectors)
{
#if defined(ANVIL_AVX2_FMA)
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
#elif defined(ANVIL_SIMD128)
	// Even-aligned row pairs: columns 0 and 1 over rows 0-5, then columns 2 and 3 over
	// rows 2-5 and columns 4 and 5 over rows 4 and 5. Column 1's row 0 and column 3's
	// row 2 are upper entries, which are never read.
	simd::Double2 column00 = simd::load(destination), column01 = simd::load(destination + 2), column02 = simd::load(destination + 4);
	simd::Double2 column10 = simd::load(destination + 6), column11 = simd::load(destination + 8), column12 = simd::load(destination + 10);
	for(int vector = 0; vector < 6; ++vector)
	{
		const double* value = vectors + 6 * vector;
		const simd::Double2 rows0 = simd::load(value), rows2 = simd::load(value + 2), rows4 = simd::load(value + 4);
		const simd::Double2 scale0 = simd::splat(value[0]), scale1 = simd::splat(value[1]);
		column00 = simd::negativeMultiplyAdd(scale0, rows0, column00);
		column01 = simd::negativeMultiplyAdd(scale0, rows2, column01);
		column02 = simd::negativeMultiplyAdd(scale0, rows4, column02);
		column10 = simd::negativeMultiplyAdd(scale1, rows0, column10);
		column11 = simd::negativeMultiplyAdd(scale1, rows2, column11);
		column12 = simd::negativeMultiplyAdd(scale1, rows4, column12);
	}
	simd::store(destination, column00);
	simd::store(destination + 2, column01);
	simd::store(destination + 4, column02);
	simd::store(destination + 6, column10);
	simd::store(destination + 8, column11);
	simd::store(destination + 10, column12);
	simd::Double2 column21 = simd::load(destination + 14), column22 = simd::load(destination + 16);
	simd::Double2 column31 = simd::load(destination + 20), column32 = simd::load(destination + 22);
	simd::Double2 column42 = simd::load(destination + 28), column52 = simd::load(destination + 34);
	for(int vector = 0; vector < 6; ++vector)
	{
		const double* value = vectors + 6 * vector;
		const simd::Double2 rows2 = simd::load(value + 2), rows4 = simd::load(value + 4);
		const simd::Double2 scale2 = simd::splat(value[2]), scale3 = simd::splat(value[3]);
		column21 = simd::negativeMultiplyAdd(scale2, rows2, column21);
		column22 = simd::negativeMultiplyAdd(scale2, rows4, column22);
		column31 = simd::negativeMultiplyAdd(scale3, rows2, column31);
		column32 = simd::negativeMultiplyAdd(scale3, rows4, column32);
		column42 = simd::negativeMultiplyAdd(simd::splat(value[4]), rows4, column42);
		column52 = simd::negativeMultiplyAdd(simd::splat(value[5]), rows4, column52);
	}
	simd::store(destination + 14, column21);
	simd::store(destination + 16, column22);
	simd::store(destination + 20, column31);
	simd::store(destination + 22, column32);
	simd::store(destination + 28, column42);
	simd::store(destination + 34, column52);
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

// Single-precision 6x6 blocks. With 256-bit vectors, and in the scalar kernels, a block stores
// each column in eight floats: rows 6 and 7 are zero padding that stays zero through every
// kernel below, so a column is one AVX register. With 128-bit vectors a block is 36 floats
// without padding: rows 0-3 of the six columns, then rows 4-5 of the six columns, so two
// columns' last rows share a vector and a block update takes nine vectors rather than twelve.
// Every entry receives the same operations in both layouts.
#if defined(ANVIL_SIMD128) && !defined(ANVIL_AVX2_FMA)
#define ANVIL_PACKED_FLOAT_BLOCKS 1
enum { FLOAT_BLOCK_SIZE = 36, FLOAT_BLOCK_ALIGNMENT = 16 };
ANVIL_FORCE_INLINE int floatBlockIndex(int row, int column) { return row < 4 ? 4 * column + row : 20 + 2 * column + row; }
#else
enum { FLOAT_BLOCK_SIZE = 48, FLOAT_BLOCK_ALIGNMENT = 32 };
ANVIL_FORCE_INLINE int floatBlockIndex(int row, int column) { return 8 * column + row; }
#endif

// destination -= left * right^T.
ANVIL_FORCE_INLINE void subtractProductFloat6(float* ANVIL_RESTRICT destination, const float* ANVIL_RESTRICT left, const float* ANVIL_RESTRICT right)
{
#if defined(ANVIL_AVX2_FMA)
	__m256 column0 = _mm256_loadu_ps(destination);
	__m256 column1 = _mm256_loadu_ps(destination + 8);
	__m256 column2 = _mm256_loadu_ps(destination + 16);
	__m256 column3 = _mm256_loadu_ps(destination + 24);
	__m256 column4 = _mm256_loadu_ps(destination + 32);
	__m256 column5 = _mm256_loadu_ps(destination + 40);
	for(int inner = 0; inner < 6; ++inner)
	{
		const __m256 source = _mm256_loadu_ps(left + 8 * inner);
		const float* scale = right + 8 * inner;
		column0 = _mm256_fnmadd_ps(_mm256_set1_ps(scale[0]), source, column0);
		column1 = _mm256_fnmadd_ps(_mm256_set1_ps(scale[1]), source, column1);
		column2 = _mm256_fnmadd_ps(_mm256_set1_ps(scale[2]), source, column2);
		column3 = _mm256_fnmadd_ps(_mm256_set1_ps(scale[3]), source, column3);
		column4 = _mm256_fnmadd_ps(_mm256_set1_ps(scale[4]), source, column4);
		column5 = _mm256_fnmadd_ps(_mm256_set1_ps(scale[5]), source, column5);
	}
	_mm256_storeu_ps(destination, column0);
	_mm256_storeu_ps(destination + 8, column1);
	_mm256_storeu_ps(destination + 16, column2);
	_mm256_storeu_ps(destination + 24, column3);
	_mm256_storeu_ps(destination + 32, column4);
	_mm256_storeu_ps(destination + 40, column5);
#elif defined(ANVIL_SIMD128)
	// Nine accumulators hold the whole result: rows 0-3 of each column, and rows 4-5 of each
	// pair of columns, whose multipliers join the two columns' scales.
	simd::Float4 head0 = simd::loadFloat(destination), head1 = simd::loadFloat(destination + 4), head2 = simd::loadFloat(destination + 8);
	simd::Float4 head3 = simd::loadFloat(destination + 12), head4 = simd::loadFloat(destination + 16), head5 = simd::loadFloat(destination + 20);
	simd::Float4 tail01 = simd::loadFloat(destination + 24), tail23 = simd::loadFloat(destination + 28), tail45 = simd::loadFloat(destination + 32);
	for(int inner = 0; inner < 6; ++inner)
	{
		const simd::Float4 sourceHead = simd::loadFloat(left + 4 * inner);
		const simd::Float4 sourceTail = simd::loadFloatPairTwice(left + 24 + 2 * inner);
		// Row j of the right block's column is the scale of the destination's column j.
		const float* scaleHead = right + 4 * inner;
		const float* scaleTail = right + 24 + 2 * inner;
		const simd::Float4 scale0 = simd::splatFloat(scaleHead[0]), scale1 = simd::splatFloat(scaleHead[1]);
		const simd::Float4 scale2 = simd::splatFloat(scaleHead[2]), scale3 = simd::splatFloat(scaleHead[3]);
		const simd::Float4 scale4 = simd::splatFloat(scaleTail[0]), scale5 = simd::splatFloat(scaleTail[1]);
		head0 = simd::negativeMultiplyAddFloat(scale0, sourceHead, head0);
		head1 = simd::negativeMultiplyAddFloat(scale1, sourceHead, head1);
		head2 = simd::negativeMultiplyAddFloat(scale2, sourceHead, head2);
		head3 = simd::negativeMultiplyAddFloat(scale3, sourceHead, head3);
		head4 = simd::negativeMultiplyAddFloat(scale4, sourceHead, head4);
		head5 = simd::negativeMultiplyAddFloat(scale5, sourceHead, head5);
		tail01 = simd::negativeMultiplyAddFloat(simd::joinLowFloat(scale0, scale1), sourceTail, tail01);
		tail23 = simd::negativeMultiplyAddFloat(simd::joinLowFloat(scale2, scale3), sourceTail, tail23);
		tail45 = simd::negativeMultiplyAddFloat(simd::joinLowFloat(scale4, scale5), sourceTail, tail45);
	}
	simd::storeFloat(destination, head0);
	simd::storeFloat(destination + 4, head1);
	simd::storeFloat(destination + 8, head2);
	simd::storeFloat(destination + 12, head3);
	simd::storeFloat(destination + 16, head4);
	simd::storeFloat(destination + 20, head5);
	simd::storeFloat(destination + 24, tail01);
	simd::storeFloat(destination + 28, tail23);
	simd::storeFloat(destination + 32, tail45);
#else
	for(int column = 0; column < 6; ++column)
	{
		for(int inner = 0; inner < 6; ++inner)
		{
			for(int row = 0; row < 6; ++row)
			{
				destination[8 * column + row] -= left[8 * inner + row] * right[8 * inner + column];
			}
		}
	}
#endif
}

// value = value * inverse(lower)^T for a lower triangular factor with the given reciprocal
// diagonal, one column at a time.
ANVIL_FORCE_INLINE void solveTransposedLowerFloat6(float* ANVIL_RESTRICT value, const float* ANVIL_RESTRICT lower, const double* ANVIL_RESTRICT inverseDiagonal)
{
#if defined(ANVIL_AVX2_FMA)
	__m256 solved[6];
	for(int column = 0; column < 6; ++column)
	{
		__m256 current = _mm256_loadu_ps(value + 8 * column);
		for(int inner = 0; inner < column; ++inner)
		{
			current = _mm256_fnmadd_ps(_mm256_set1_ps(lower[8 * inner + column]), solved[inner], current);
		}
		solved[column] = _mm256_mul_ps(current, _mm256_set1_ps(float(inverseDiagonal[column])));
		_mm256_storeu_ps(value + 8 * column, solved[column]);
	}
#elif defined(ANVIL_SIMD128)
	// Each column's rows 4-5 are solved as a pair of floats.
	simd::Float4 solvedHead[6], solvedTail[6];
	for(int column = 0; column < 6; ++column)
	{
		simd::Float4 head = simd::loadFloat(value + 4 * column), tail = simd::loadFloatPair(value + 24 + 2 * column);
		for(int inner = 0; inner < column; ++inner)
		{
			const simd::Float4 multiplier = simd::splatFloat(lower[floatBlockIndex(column, inner)]);
			head = simd::negativeMultiplyAddFloat(multiplier, solvedHead[inner], head);
			tail = simd::negativeMultiplyAddFloat(multiplier, solvedTail[inner], tail);
		}
		const simd::Float4 inverse = simd::splatFloat(float(inverseDiagonal[column]));
		solvedHead[column] = simd::multiplyFloat(head, inverse);
		solvedTail[column] = simd::multiplyFloat(tail, inverse);
		simd::storeFloat(value + 4 * column, solvedHead[column]);
		simd::storeFloatPair(value + 24 + 2 * column, solvedTail[column]);
	}
#else
	for(int column = 0; column < 6; ++column)
	{
		float* current = value + 8 * column;
		for(int inner = 0; inner < column; ++inner)
		{
			const float scale = lower[8 * inner + column];
			for(int row = 0; row < 6; ++row)
			{
				current[row] -= scale * value[8 * inner + row];
			}
		}
		const float inverse = float(inverseDiagonal[column]);
		for(int row = 0; row < 6; ++row)
		{
			current[row] *= inverse;
		}
	}
#endif
}

// destination -= matrix * vector for a single-precision block, in double precision.
ANVIL_FORCE_INLINE void subtractFloatMatrixVector6(double* ANVIL_RESTRICT destination, const float* ANVIL_RESTRICT matrix, const double* ANVIL_RESTRICT vector)
{
#if defined(ANVIL_AVX2_FMA)
	__m256d first = _mm256_loadu_pd(destination);
	__m128d second = _mm_loadu_pd(destination + 4);
	for(int column = 0; column < 6; ++column)
	{
		const __m256d multiplier = _mm256_set1_pd(vector[column]);
		first = _mm256_fnmadd_pd(multiplier, _mm256_cvtps_pd(_mm_loadu_ps(matrix + 8 * column)), first);
		second = _mm_fnmadd_pd(_mm256_castpd256_pd128(multiplier), _mm_cvtps_pd(_mm_loadu_ps(matrix + 8 * column + 4)), second);
	}
	_mm256_storeu_pd(destination, first);
	_mm_storeu_pd(destination + 4, second);
#elif defined(ANVIL_SIMD128)
	simd::Double2 first = simd::load(destination);
	simd::Double2 second = simd::load(destination + 2);
	simd::Double2 third = simd::load(destination + 4);
	for(int column = 0; column < 6; ++column)
	{
		const simd::Double2 multiplier = simd::splat(vector[column]);
		const simd::Float4 head = simd::loadFloat(matrix + 4 * column), tail = simd::loadFloatPair(matrix + 24 + 2 * column);
		first = simd::negativeMultiplyAdd(multiplier, simd::promoteLow(head), first);
		second = simd::negativeMultiplyAdd(multiplier, simd::promoteHigh(head), second);
		third = simd::negativeMultiplyAdd(multiplier, simd::promoteLow(tail), third);
	}
	simd::store(destination, first);
	simd::store(destination + 2, second);
	simd::store(destination + 4, third);
#else
	for(int column = 0; column < 6; ++column)
	{
		for(int row = 0; row < 6; ++row)
		{
			destination[row] -= double(matrix[8 * column + row]) * vector[column];
		}
	}
#endif
}

// A double-precision six-row column as a column of a single-precision block.
ANVIL_FORCE_INLINE void convertColumnFloat6(float* ANVIL_RESTRICT block, int column, const double* ANVIL_RESTRICT source)
{
#if defined(ANVIL_AVX2_FMA)
	float* destination = block + 8 * column;
	_mm_storeu_ps(destination, _mm256_cvtpd_ps(_mm256_loadu_pd(source)));
	_mm_storeu_ps(destination + 4, _mm_cvtpd_ps(_mm_loadu_pd(source + 4)));
#elif defined(ANVIL_SIMD128)
	simd::storeFloat(block + 4 * column, simd::demote(simd::load(source), simd::load(source + 2)));
	simd::storeFloatPair(block + 24 + 2 * column, simd::demoteLow(simd::load(source + 4)));
#else
	float* destination = block + 8 * column;
	for(int row = 0; row < 6; ++row)
	{
		destination[row] = float(source[row]);
	}
	destination[6] = destination[7] = 0.0f;
#endif
}

// A column of a single-precision block as six doubles.
ANVIL_FORCE_INLINE void promoteColumnFloat6(double* ANVIL_RESTRICT destination, const float* ANVIL_RESTRICT block, int column)
{
#if defined(ANVIL_AVX2_FMA)
	const float* source = block + 8 * column;
	_mm256_storeu_pd(destination, _mm256_cvtps_pd(_mm_loadu_ps(source)));
	_mm_storeu_pd(destination + 4, _mm_cvtps_pd(_mm_loadu_ps(source + 4)));
#elif defined(ANVIL_SIMD128)
	const simd::Float4 head = simd::loadFloat(block + 4 * column);
	simd::store(destination, simd::promoteLow(head));
	simd::store(destination + 2, simd::promoteHigh(head));
	simd::store(destination + 4, simd::promoteLow(simd::loadFloatPair(block + 24 + 2 * column)));
#else
	const float* source = block + 8 * column;
	for(int row = 0; row < 6; ++row)
	{
		destination[row] = double(source[row]);
	}
#endif
}

// Dot product of a single-precision block's column and a double vector, in double precision.
ANVIL_FORCE_INLINE double dotFloat6(const float* ANVIL_RESTRICT block, int column, const double* ANVIL_RESTRICT right)
{
#if defined(ANVIL_AVX2_FMA)
	const float* left = block + 8 * column;
	const __m256d product = _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(left)), _mm256_loadu_pd(right));
	const __m128d halves = _mm_add_pd(_mm256_castpd256_pd128(product), _mm256_extractf128_pd(product, 1));
	const __m128d sum = _mm_add_pd(halves, _mm_unpackhi_pd(halves, halves));
	const __m128d end = _mm_mul_pd(_mm_cvtps_pd(_mm_loadu_ps(left + 4)), _mm_loadu_pd(right + 4));
	return _mm_cvtsd_f64(sum) + _mm_cvtsd_f64(end) + _mm_cvtsd_f64(_mm_unpackhi_pd(end, end));
#elif defined(ANVIL_SIMD128)
	const simd::Float4 head = simd::loadFloat(block + 4 * column), tail = simd::loadFloatPair(block + 24 + 2 * column);
	const simd::Double2 first = simd::multiply(simd::promoteLow(head), simd::load(right));
	const simd::Double2 second = simd::multiplyAdd(simd::promoteHigh(head), simd::load(right + 2), first);
	return simd::sum(simd::multiplyAdd(simd::promoteLow(tail), simd::load(right + 4), second));
#else
	const float* left = block + 8 * column;
	double result = double(left[0]) * right[0];
	for(int i = 1; i < 6; ++i)
	{
		result += double(left[i]) * right[i];
	}
	return result;
#endif
}

ANVIL_FORCE_INLINE double subtractDot6(double value, const double* left, const double* right)
{
#if defined(ANVIL_AVX2_FMA)
	const __m256d product = _mm256_mul_pd(_mm256_loadu_pd(left), _mm256_loadu_pd(right));
	const __m128d halves = _mm_add_pd(_mm256_castpd256_pd128(product), _mm256_extractf128_pd(product, 1));
	const __m128d sum = _mm_add_pd(halves, _mm_unpackhi_pd(halves, halves));
	const __m128d end = _mm_mul_pd(_mm_loadu_pd(left + 4), _mm_loadu_pd(right + 4));
	value -= _mm_cvtsd_f64(sum);
	value -= _mm_cvtsd_f64(end);
	value -= _mm_cvtsd_f64(_mm_unpackhi_pd(end, end));
	return value;
#elif defined(ANVIL_SIMD128)
	const simd::Double2 first = simd::multiply(simd::load(left), simd::load(right));
	const simd::Double2 second = simd::multiplyAdd(simd::load(left + 2), simd::load(right + 2), first);
	return value - simd::sum(simd::multiplyAdd(simd::load(left + 4), simd::load(right + 4), second));
#else
	for(int i = 0; i < 6; ++i)
	{
		value -= left[i] * right[i];
	}
	return value;
#endif
}

ANVIL_FORCE_INLINE void updateCholesky6(double* ANVIL_RESTRICT factor, double* ANVIL_RESTRICT work, double inverseC, double signedSC, double c, double s)
{
#if defined(ANVIL_AVX2_FMA)
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
#elif defined(ANVIL_SIMD128)
	const simd::Double2 inverse = simd::splat(inverseC), update = simd::splat(signedSC);
	const simd::Double2 cosine = simd::splat(c), sine = simd::splat(s);
	for(int axis = 0; axis < 6; axis += 2)
	{
		const simd::Double2 oldWork = simd::load(work + axis);
		const simd::Double2 nextFactor = simd::multiplyAdd(update, oldWork, simd::multiply(inverse, simd::load(factor + axis)));
		simd::store(factor + axis, nextFactor);
		simd::store(work + axis, simd::negativeMultiplyAdd(sine, nextFactor, simd::multiply(cosine, oldWork)));
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
class alignas((Rows == 6 && Columns == 6) ? 32 : 8) Matrix
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
#if defined(ANVIL_AVX2_FMA)
		std::uint32_t i = 0;
		for(; i + 4 <= count; i += 4)
		{
			const __m256d products = _mm256_mul_pd(_mm256_loadu_pd(m_values.data() + i), _mm256_loadu_pd(other.m_values.data() + i));
			const __m128d low = _mm256_castpd256_pd128(products);
			const __m128d high = _mm256_extractf128_pd(products, 1);
			result += _mm_cvtsd_f64(low);
			result += _mm_cvtsd_f64(_mm_unpackhi_pd(low, low));
			result += _mm_cvtsd_f64(high);
			result += _mm_cvtsd_f64(_mm_unpackhi_pd(high, high));
		}
		for(; i < count; ++i)
#elif defined(ANVIL_SIMD128)
		std::uint32_t i = 0;
		for(; i + 2 <= count; i += 2)
		{
			const simd::Double2 products = simd::multiply(simd::load(m_values.data() + i), simd::load(other.m_values.data() + i));
			result += simd::low(products);
			result += simd::high(products);
		}
		for(; i < count; ++i)
#else
		for(std::uint32_t i = 0; i < count; ++i)
#endif
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

// Also returns each pivot's reciprocal for solves that multiply instead of divide.
inline bool cholesky6(const Matrix<6, 6>& input, Matrix<6, 6>& lower, double* inverseDiagonal)
{
	lower.setZero();
	for(int column = 0; column < 6; ++column)
	{
		double diagonal = input(column, column);
		for(int inner = 0; inner < column; ++inner)
		{
			diagonal -= lower(column, inner) * lower(column, inner);
		}
		if(diagonal <= 0.0)
		{
			return false;
		}
		lower(column, column) = std::sqrt(diagonal);
		const double inverse = 1.0 / lower(column, column);
		if(inverseDiagonal)
		{
			inverseDiagonal[column] = inverse;
		}
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

inline bool cholesky6(const Matrix<6, 6>& input, Matrix<6, 6>& lower)
{
	return cholesky6(input, lower, NULL);
}

// The same factor as the strict lower triangle of a row-major 6x6 array (other entries are
// untouched) and the inverse diagonal, which is all a triangular solve needs.
inline bool cholesky6Packed(const Matrix<6, 6>& input, double* strictLower, double* inverseDiagonal)
{
	ANVIL_UNROLL
	for(int column = 0; column < 6; ++column)
	{
		double diagonal = input(column, column);
		ANVIL_UNROLL
		for(int inner = 0; inner < column; ++inner)
		{
			diagonal -= strictLower[column * 6 + inner] * strictLower[column * 6 + inner];
		}
		if(diagonal <= 0.0)
		{
			return false;
		}
		const double inverse = 1.0 / std::sqrt(diagonal);
		inverseDiagonal[column] = inverse;
		ANVIL_UNROLL
		for(int row = column + 1; row < 6; ++row)
		{
			double value = input(row, column);
			ANVIL_UNROLL
			for(int inner = 0; inner < column; ++inner)
			{
				value -= strictLower[row * 6 + inner] * strictLower[column * 6 + inner];
			}
			strictLower[row * 6 + column] = value * inverse;
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
