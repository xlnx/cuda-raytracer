#pragma once

#include "trait.hpp"
#include "vec.hpp"
// #if !defined( KOISHI_USE_CUDA )
#include <cmath>
// #endif

namespace koishi
{
namespace vm
{
#if defined( KOISHI_USE_CUDA )

#define KOISHI_HOST_DEVICE \
	__host__ __device__
#define KOISHI_MATH_NAMESP

#else

#define KOISHI_HOST_DEVICE \
	inline
#define KOISHI_MATH_NAMESP \
	std

#endif

namespace __func
{
template <typename T, typename = typename std::enable_if<
						trait::is_in<T, int, uint, float, double>::value>::type>
KOISHI_HOST_DEVICE T min( T a, T b )
{
	return a < b ? a : b;
}

template <typename T, typename = typename std::enable_if<
						trait::is_in<T, int, uint, float, double>::value>::type>
KOISHI_HOST_DEVICE T max( T a, T b )
{
	return a > b ? a : b;
}

template <typename T, typename = typename std::enable_if<
						trait::is_in<T, float, double>::value>::type>
KOISHI_HOST_DEVICE T radians( T deg )
{
	return T( M_PI ) * deg / T( 180. );
}

template <typename T, typename = typename std::enable_if<
						trait::is_in<T, float, double>::value>::type>
KOISHI_HOST_DEVICE T degrees( T deg )
{
	return deg * T( 180. ) / T( M_PI );
}

#if !defined( KOISHI_USE_CUDA )
template <typename T, typename = typename std::enable_if<
						trait::is_in<T, float, double>::value>::type>
KOISHI_HOST_DEVICE T rsqrt( T a )
{
	return T( 1.0 ) / KOISHI_MATH_NAMESP::sqrt( a );
}

#endif

}  // namespace __func

#define KOISHI_COMPWISE_OP( op, ... )                                                               \
	namespace vec1                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a, const T &b )                                      \
	{                                                                                               \
		return { a.x op b.x };                                                                      \
	}                                                                                               \
	}                                                                                               \
	namespace vec2                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a, const T &b )                                      \
	{                                                                                               \
		return { a.x op b.x, a.y op b.y };                                                          \
	}                                                                                               \
	}                                                                                               \
	namespace vec3                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a, const T &b )                                      \
	{                                                                                               \
		return { a.x op b.x, a.y op b.y, a.z op b.z };                                              \
	}                                                                                               \
	}                                                                                               \
	namespace vec4                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a, const T &b )                                      \
	{                                                                                               \
		return { a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w };                                  \
	}                                                                                               \
	}

KOISHI_COMPWISE_OP( +, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( -, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( *, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( /, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( %, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( &, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( |, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( <<, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( >>, KOISHI_VEC_INT, KOISHI_VEC_UINT )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( op, ... )                                                               \
	namespace vec1                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a )                                                  \
	{                                                                                               \
		return { op a.x };                                                                          \
	}                                                                                               \
	}                                                                                               \
	namespace vec2                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a )                                                  \
	{                                                                                               \
		return { op a.x, op a.y };                                                                  \
	}                                                                                               \
	}                                                                                               \
	namespace vec3                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a )                                                  \
	{                                                                                               \
		return { op a.x, op a.y, op a.z };                                                          \
	}                                                                                               \
	}                                                                                               \
	namespace vec4                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a )                                                  \
	{                                                                                               \
		return { op a.x, op a.y, op a.z, op a.w };                                                  \
	}                                                                                               \
	}

KOISHI_COMPWISE_OP( +, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( -, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( ~, KOISHI_VEC_INT, KOISHI_VEC_UINT )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( op, ... )                                                               \
	namespace vec1                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T &operator op( T &a, const T &b )                                           \
	{                                                                                               \
		return a.x op b.x, a;                                                                       \
	}                                                                                               \
	}                                                                                               \
	namespace vec2                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T &operator op( T &a, const T &b )                                           \
	{                                                                                               \
		return a.x op b.x, a.y op b.y, a;                                                           \
	}                                                                                               \
	}                                                                                               \
	namespace vec3                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T &operator op( T &a, const T &b )                                           \
	{                                                                                               \
		return a.x op b.x, a.y op b.y, a.z op b.z, a;                                               \
	}                                                                                               \
	}                                                                                               \
	namespace vec4                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T &operator op( T &a, const T &b )                                           \
	{                                                                                               \
		return a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w, a;                                   \
	}                                                                                               \
	}

KOISHI_COMPWISE_OP( +=, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( -=, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( *=, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( /=, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( %=, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( &=, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( |=, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( <<=, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( >>=, KOISHI_VEC_INT, KOISHI_VEC_UINT )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( op, ... )                                                               \
	namespace vec1                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a, typename trait::com<T>::type b )                  \
	{                                                                                               \
		return { a.x op b };                                                                        \
	}                                                                                               \
	}                                                                                               \
	namespace vec2                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a, typename trait::com<T>::type b )                  \
	{                                                                                               \
		return { a.x op b, a.y op b };                                                              \
	}                                                                                               \
	}                                                                                               \
	namespace vec3                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a, typename trait::com<T>::type b )                  \
	{                                                                                               \
		return { a.x op b, a.y op b, a.z op b };                                                    \
	}                                                                                               \
	}                                                                                               \
	namespace vec4                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a, typename trait::com<T>::type b )                  \
	{                                                                                               \
		return { a.x op b, a.y op b, a.z op b, a.w op b };                                          \
	}                                                                                               \
	}                                                                                               \
	namespace vec1                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( typename trait::com<T>::type a, const T &b )                  \
	{                                                                                               \
		return { a op b.x };                                                                        \
	}                                                                                               \
	}                                                                                               \
	namespace vec2                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( typename trait::com<T>::type a, const T &b )                  \
	{                                                                                               \
		return { a op b.x, a op b.y };                                                              \
	}                                                                                               \
	}                                                                                               \
	namespace vec3                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( typename trait::com<T>::type a, const T &b )                  \
	{                                                                                               \
		return { a op b.x, a op b.y, a op b.z };                                                    \
	}                                                                                               \
	}                                                                                               \
	namespace vec4                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( typename trait::com<T>::type a, const T &b )                  \
	{                                                                                               \
		return { a op b.x, a op b.y, a op b.z, a op b.w };                                          \
	}                                                                                               \
	}

KOISHI_COMPWISE_OP( +, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( -, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( *, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( /, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( %, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( &, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( |, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( <<, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( >>, KOISHI_VEC_INT, KOISHI_VEC_UINT )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( op, ... )                                                               \
	namespace vec1                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T &operator op( T &a, typename trait::com<T>::type b )                       \
	{                                                                                               \
		return a.x op b, a;                                                                         \
	}                                                                                               \
	}                                                                                               \
	namespace vec2                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T &operator op( T &a, typename trait::com<T>::type b )                       \
	{                                                                                               \
		return a.x op b, a.y op b, a;                                                               \
	}                                                                                               \
	}                                                                                               \
	namespace vec3                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T &operator op( T &a, typename trait::com<T>::type b )                       \
	{                                                                                               \
		return a.x op b, a.y op b, a.z op b, a;                                                     \
	}                                                                                               \
	}                                                                                               \
	namespace vec4                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T &operator op( T &a, typename trait::com<T>::type b )                       \
	{                                                                                               \
		return a.x op b, a.y op b, a.z op b, a.w op b, a;                                           \
	}                                                                                               \
	}

KOISHI_COMPWISE_OP( +=, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( -=, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( *=, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( /=, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( %=, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( &=, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( |=, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( <<=, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( >>=, KOISHI_VEC_INT, KOISHI_VEC_UINT )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( fn, name, ... )                                                         \
	namespace vec1                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a, const T &b )                                             \
	{                                                                                               \
		return { fn( a.x, b.x ) };                                                                  \
	}                                                                                               \
	}                                                                                               \
	namespace vec2                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a, const T &b )                                             \
	{                                                                                               \
		return { fn( a.x, b.x ), fn( a.y, b.y ) };                                                  \
	}                                                                                               \
	}                                                                                               \
	namespace vec3                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a, const T &b )                                             \
	{                                                                                               \
		return { fn( a.x, b.x ), fn( a.y, b.y ), fn( a.z, b.z ) };                                  \
	}                                                                                               \
	}                                                                                               \
	namespace vec4                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a, const T &b )                                             \
	{                                                                                               \
		return { fn( a.x, b.x ), fn( a.y, b.y ), fn( a.z, b.z ), fn( a.w, b.w ) };                  \
	}                                                                                               \
	}

KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::pow, pow, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::atan2, atan2, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( __func::max, max, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( __func::min, min, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( ... )                                                                   \
	namespace vec1                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE typename trait::com<T>::type dot( const T &a, const T &b )                   \
	{                                                                                               \
		return a.x * b.x;                                                                           \
	}                                                                                               \
	}                                                                                               \
	namespace vec2                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE typename trait::com<T>::type dot( const T &a, const T &b )                   \
	{                                                                                               \
		return a.x * b.x + a.y * b.y;                                                               \
	}                                                                                               \
	}                                                                                               \
	namespace vec3                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE typename trait::com<T>::type dot( const T &a, const T &b )                   \
	{                                                                                               \
		return a.x * b.x + a.y * b.y + a.z * b.z;                                                   \
	}                                                                                               \
	}                                                                                               \
	namespace vec4                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE typename trait::com<T>::type dot( const T &a, const T &b )                   \
	{                                                                                               \
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;                                       \
	}                                                                                               \
	}

KOISHI_COMPWISE_OP( KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( ... )                                                                   \
	namespace vec3                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T cross( const T &a, const T &b )                                            \
	{                                                                                               \
		return { a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y };             \
	}                                                                                               \
	}

KOISHI_COMPWISE_OP( KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( ... )                                                                   \
	namespace vec1                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T reflect( const T &I, const T &N )                                          \
	{                                                                                               \
		return I - N * dot( N, I ) * typename trait::com<T>::type( 2 );                             \
	}                                                                                               \
	}                                                                                               \
	namespace vec2                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T reflect( const T &I, const T &N )                                          \
	{                                                                                               \
		return I - N * dot( N, I ) * typename trait::com<T>::type( 2 );                             \
	}                                                                                               \
	}                                                                                               \
	namespace vec3                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T reflect( const T &I, const T &N )                                          \
	{                                                                                               \
		return I - N * dot( N, I ) * typename trait::com<T>::type( 2 );                             \
	}                                                                                               \
	}                                                                                               \
	namespace vec4                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T reflect( const T &I, const T &N )                                          \
	{                                                                                               \
		return I - N * dot( N, I ) * typename trait::com<T>::type( 2 );                             \
	}                                                                                               \
	}

KOISHI_COMPWISE_OP( KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( fn, name, ... )                                                         \
	namespace vec1                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a )                                                         \
	{                                                                                               \
		return { fn( a.x ) };                                                                       \
	}                                                                                               \
	}                                                                                               \
	namespace vec2                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a )                                                         \
	{                                                                                               \
		return { fn( a.x ), fn( a.y ) };                                                            \
	}                                                                                               \
	}                                                                                               \
	namespace vec3                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a )                                                         \
	{                                                                                               \
		return { fn( a.x ), fn( a.y ), fn( a.z ) };                                                 \
	}                                                                                               \
	}                                                                                               \
	namespace vec4                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a )                                                         \
	{                                                                                               \
		return { fn( a.x ), fn( a.y ), fn( a.z ), fn( a.w ) };                                      \
	}                                                                                               \
	}

KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::sin, sin, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::cos, cos, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::tan, tan, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::asin, asin, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::acos, acos, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::atan, atan, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::exp, exp, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::log, log, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::exp2, exp2, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::log2, log2, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::sqrt, sqrt, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( __func::radians, radians, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( __func::degrees, degrees, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
#if defined( KOISHI_USE_CUDA )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::rsqrt, rsqrt, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
#else
KOISHI_COMPWISE_OP( __func::rsqrt, rsqrt, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
#endif
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::abs, abs, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::floor, floor, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::ceil, ceil, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

// #undef KOISHI_COMPWISE_OP
// #define KOISHI_COMPWISE_OP( fn, name, ... )                                                              \
// 	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value>::type> \
// 	KOISHI_HOST_DEVICE T name( T a )                                                                     \
// 	{                                                                                                    \
// 		return fn( a );                                                                                  \
// 	}

// KOISHI_COMPWISE_OP( __func::radians, radians, float, double )
// KOISHI_COMPWISE_OP( __func::degrees, degrees, float, double )
// #if !defined( KOISHI_USE_CUDA )
// KOISHI_COMPWISE_OP( __func::rsqrt, rsqrt, float, double )
// #endif

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( sqrt, ... )                                                             \
	namespace vec1                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE typename trait::com<T>::type length( const T &a )                            \
	{                                                                                               \
		return sqrt( a.x * a.x );                                                                   \
	}                                                                                               \
	}                                                                                               \
	namespace vec2                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE typename trait::com<T>::type length( const T &a )                            \
	{                                                                                               \
		return sqrt( a.x * a.x + a.y * a.y );                                                       \
	}                                                                                               \
	}                                                                                               \
	namespace vec3                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE typename trait::com<T>::type length( const T &a )                            \
	{                                                                                               \
		return sqrt( a.x * a.x + a.y * a.y + a.z * a.z );                                           \
	}                                                                                               \
	}                                                                                               \
	namespace vec4                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE typename trait::com<T>::type length( const T &a )                            \
	{                                                                                               \
		return sqrt( a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w );                               \
	}                                                                                               \
	}

KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::sqrt, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( rsqrt, ... )                                                            \
	namespace vec1                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T normalize( const T &a )                                                    \
	{                                                                                               \
		return a * rsqrt( a.x * a.x );                                                              \
	}                                                                                               \
	}                                                                                               \
	namespace vec2                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T normalize( const T &a )                                                    \
	{                                                                                               \
		return a * rsqrt( a.x * a.x + a.y * a.y );                                                  \
	}                                                                                               \
	}                                                                                               \
	namespace vec3                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T normalize( const T &a )                                                    \
	{                                                                                               \
		return a * rsqrt( a.x * a.x + a.y * a.y + a.z * a.z );                                      \
	}                                                                                               \
	}                                                                                               \
	namespace vec4                                                                                  \
	{                                                                                               \
	template <typename T, typename = typename std::enable_if<trait::is_in<T, __VA_ARGS__>::value && \
															 trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T normalize( const T &a )                                                    \
	{                                                                                               \
		return a * rsqrt( a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w );                          \
	}                                                                                               \
	}

#if defined( KOISHI_USE_CUDA )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::rsqrt, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
#else
KOISHI_COMPWISE_OP( __func::rsqrt, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
#endif

using namespace vec1;
using namespace vec2;
using namespace vec3;
using namespace vec4;

#undef KOISHI_MATH_NAMESP

#undef KOISHI_COMPWISE_OP

}  // namespace vm

}  // namespace koishi

using namespace koishi::vm;
using namespace koishi::vm::__func;
