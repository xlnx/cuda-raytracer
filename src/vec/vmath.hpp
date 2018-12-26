#pragma once

#include "trait.hpp"
#include "vec.hpp"
// #if !defined( KOISHI_USE_CUDA )
#include <cmath>
// #endif

#if !defined( KOISHI_USE_CUDA )

namespace koishi
{
namespace vec
{
#endif

#if defined( KOISHI_USE_CUDA )

#define KOISHI_HOST \
	__host__
#define KOISHI_DEVICE \
	__device__
#define KOISHI_HOST_DEVICE \
	__host__ __device__
#define KOISHI_NOINLINE \
	__noinline__

#define KOISHI_MATH_NAMESP

#define KOISHI_RESTRICT __restrict__

#else

#define KOISHI_HOST

#define KOISHI_DEVICE

#define KOISHI_HOST_DEVICE

#define KOISHI_NOINLINE

#define KOISHI_MATH_NAMESP \
	std

#define KOISHI_RESTRICT

#endif

namespace __func
{
template <typename T, typename = typename std::enable_if<
						koishi::trait::is_in<T, int, uint, float, double>::value>::type>
KOISHI_HOST_DEVICE T min( T a, T b )
{
	return a < b ? a : b;
}

template <typename T, typename = typename std::enable_if<
						koishi::trait::is_in<T, int, uint, float, double>::value>::type>
KOISHI_HOST_DEVICE T max( T a, T b )
{
	return a > b ? a : b;
}

template <typename T, typename = typename std::enable_if<
						koishi::trait::is_in<T, float, double>::value>::type>
KOISHI_HOST_DEVICE T radians( T deg )
{
	return T( M_PI ) * deg / T( 180. );
}

template <typename T, typename = typename std::enable_if<
						koishi::trait::is_in<T, float, double>::value>::type>
KOISHI_HOST_DEVICE T degrees( T deg )
{
	return deg * T( 180. ) / T( M_PI );
}

#if !defined( KOISHI_USE_CUDA )
template <typename T, typename = typename std::enable_if<
						koishi::trait::is_in<T, float, double>::value>::type>
KOISHI_HOST_DEVICE T rsqrt( T a )
{
	return T( 1.0 ) / KOISHI_MATH_NAMESP::sqrt( a );
}

#endif

}  // namespace __func

#define KOISHI_COMPWISE_OP( op, ... )                                                                                                                                                    \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const T &a, const T &b )                                                                                         \
	{                                                                                                                                                                                    \
		return { a.x op b.x };                                                                                                                                                           \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const T &a, const T &b )                                                                                         \
	{                                                                                                                                                                                    \
		return { a.x op b.x, a.y op b.y };                                                                                                                                               \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const T &a, const T &b )                                                                                         \
	{                                                                                                                                                                                    \
		return { a.x op b.x, a.y op b.y, a.z op b.z };                                                                                                                                   \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const T &a, const T &b )                                                                                         \
	{                                                                                                                                                                                    \
		return { a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w };                                                                                                                       \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const typename unnormalized_type<T>::type &a, const T &b )                                                       \
	{                                                                                                                                                                                    \
		return { a.x op b.x };                                                                                                                                                           \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const typename unnormalized_type<T>::type &a, const T &b )                                                       \
	{                                                                                                                                                                                    \
		return { a.x op b.x, a.y op b.y };                                                                                                                                               \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const typename unnormalized_type<T>::type &a, const T &b )                                                       \
	{                                                                                                                                                                                    \
		return { a.x op b.x, a.y op b.y, a.z op b.z };                                                                                                                                   \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const typename unnormalized_type<T>::type &a, const T &b )                                                       \
	{                                                                                                                                                                                    \
		return { a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w };                                                                                                                       \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const T &a, const typename unnormalized_type<T>::type &b )                                                       \
	{                                                                                                                                                                                    \
		return { a.x op b.x };                                                                                                                                                           \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const T &a, const typename unnormalized_type<T>::type &b )                                                       \
	{                                                                                                                                                                                    \
		return { a.x op b.x, a.y op b.y };                                                                                                                                               \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const T &a, const typename unnormalized_type<T>::type &b )                                                       \
	{                                                                                                                                                                                    \
		return { a.x op b.x, a.y op b.y, a.z op b.z };                                                                                                                                   \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const T &a, const typename unnormalized_type<T>::type &b )                                                       \
	{                                                                                                                                                                                    \
		return { a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w };                                                                                                                       \
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
#define KOISHI_COMPWISE_OP( op, ... )                                                                                                                                                    \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE T operator op( const T &a )                                                                                                                                       \
	{                                                                                                                                                                                    \
		return T( typename unnormalized_type<T>::type{ op a.x } );                                                                                                                       \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE T operator op( const T &a )                                                                                                                                       \
	{                                                                                                                                                                                    \
		return T( typename unnormalized_type<T>::type{ op a.x, op a.y } );                                                                                                               \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE T operator op( const T &a )                                                                                                                                       \
	{                                                                                                                                                                                    \
		return T( typename unnormalized_type<T>::type{ op a.x, op a.y, op a.z } );                                                                                                       \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE T operator op( const T &a )                                                                                                                                       \
	{                                                                                                                                                                                    \
		return T( typename unnormalized_type<T>::type{ op a.x, op a.y, op a.z, op a.w } );                                                                                               \
	}

KOISHI_COMPWISE_OP( +, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( -, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( op, ... )                                                                                                                                                    \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type &operator op( T &a, const T &b )                                                                                              \
	{                                                                                                                                                                                    \
		return a.x op b.x, a;                                                                                                                                                            \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type &operator op( T &a, const T &b )                                                                                              \
	{                                                                                                                                                                                    \
		return a.x op b.x, a.y op b.y, a;                                                                                                                                                \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type &operator op( T &a, const T &b )                                                                                              \
	{                                                                                                                                                                                    \
		return a.x op b.x, a.y op b.y, a.z op b.z, a;                                                                                                                                    \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type &operator op( T &a, const T &b )                                                                                              \
	{                                                                                                                                                                                    \
		return a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w, a;                                                                                                                        \
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
#define KOISHI_COMPWISE_OP( op, ... )                                                                                                                                                    \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE bool operator op( const T &a, const T &b )                                                                                                                        \
	{                                                                                                                                                                                    \
		return a.x op b.x;                                                                                                                                                               \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE bool operator op( const T &a, const T &b )                                                                                                                        \
	{                                                                                                                                                                                    \
		return a.x op b.x && a.y op b.y;                                                                                                                                                 \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE bool operator op( const T &a, const T &b )                                                                                                                        \
	{                                                                                                                                                                                    \
		return a.x op b.x && a.y op b.y && a.z op b.z;                                                                                                                                   \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE bool operator op( const T &a, const T &b )                                                                                                                        \
	{                                                                                                                                                                                    \
		return a.x op b.x && a.y op b.y && a.z op b.z && a.w op b.w;                                                                                                                     \
	}

KOISHI_COMPWISE_OP( ==, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( >, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( >=, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( <, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( <=, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( op, ... )                                                                                                                                                    \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const T &a, typename koishi::trait::com<T>::type b )                                                             \
	{                                                                                                                                                                                    \
		return { a.x op b };                                                                                                                                                             \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const T &a, typename koishi::trait::com<T>::type b )                                                             \
	{                                                                                                                                                                                    \
		return { a.x op b, a.y op b };                                                                                                                                                   \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const T &a, typename koishi::trait::com<T>::type b )                                                             \
	{                                                                                                                                                                                    \
		return { a.x op b, a.y op b, a.z op b };                                                                                                                                         \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( const T &a, typename koishi::trait::com<T>::type b )                                                             \
	{                                                                                                                                                                                    \
		return { a.x op b, a.y op b, a.z op b, a.w op b };                                                                                                                               \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( typename koishi::trait::com<T>::type a, const T &b )                                                             \
	{                                                                                                                                                                                    \
		return { a op b.x };                                                                                                                                                             \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( typename koishi::trait::com<T>::type a, const T &b )                                                             \
	{                                                                                                                                                                                    \
		return { a op b.x, a op b.y };                                                                                                                                                   \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( typename koishi::trait::com<T>::type a, const T &b )                                                             \
	{                                                                                                                                                                                    \
		return { a op b.x, a op b.y, a op b.z };                                                                                                                                         \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type operator op( typename koishi::trait::com<T>::type a, const T &b )                                                             \
	{                                                                                                                                                                                    \
		return { a op b.x, a op b.y, a op b.z, a op b.w };                                                                                                                               \
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
#define KOISHI_COMPWISE_OP( op, ... )                                                                                                                                                    \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type &operator op( T &a, typename koishi::trait::com<T>::type b )                                                                  \
	{                                                                                                                                                                                    \
		return a.x op b, a;                                                                                                                                                              \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type &operator op( T &a, typename koishi::trait::com<T>::type b )                                                                  \
	{                                                                                                                                                                                    \
		return a.x op b, a.y op b, a;                                                                                                                                                    \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type &operator op( T &a, typename koishi::trait::com<T>::type b )                                                                  \
	{                                                                                                                                                                                    \
		return a.x op b, a.y op b, a.z op b, a;                                                                                                                                          \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type &operator op( T &a, typename koishi::trait::com<T>::type b )                                                                  \
	{                                                                                                                                                                                    \
		return a.x op b, a.y op b, a.z op b, a.w op b, a;                                                                                                                                \
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
#define KOISHI_COMPWISE_OP( op, ... )                                                                                                                                                    \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE bool operator op( const T &a, typename koishi::trait::com<T>::type b )                                                                                            \
	{                                                                                                                                                                                    \
		return a.x op b;                                                                                                                                                                 \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE bool operator op( const T &a, typename koishi::trait::com<T>::type b )                                                                                            \
	{                                                                                                                                                                                    \
		return a.x op b && a.y op b;                                                                                                                                                     \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE bool operator op( const T &a, typename koishi::trait::com<T>::type b )                                                                                            \
	{                                                                                                                                                                                    \
		return a.x op b && a.y op b && a.z op b;                                                                                                                                         \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE bool operator op( const T &a, typename koishi::trait::com<T>::type b )                                                                                            \
	{                                                                                                                                                                                    \
		return a.x op b && a.y op b && a.z op b && a.w op b;                                                                                                                             \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE bool operator op( typename koishi::trait::com<T>::type a, const T &b )                                                                                            \
	{                                                                                                                                                                                    \
		return b.x op a;                                                                                                                                                                 \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE bool operator op( typename koishi::trait::com<T>::type a, const T &b )                                                                                            \
	{                                                                                                                                                                                    \
		return b.x op a && b.y op a;                                                                                                                                                     \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE bool operator op( typename koishi::trait::com<T>::type a, const T &b )                                                                                            \
	{                                                                                                                                                                                    \
		return b.x op a && b.y op a && b.z op a;                                                                                                                                         \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE bool operator op( typename koishi::trait::com<T>::type a, const T &b )                                                                                            \
	{                                                                                                                                                                                    \
		return b.x op a && b.y op a && b.z op a && b.w op a;                                                                                                                             \
	}

KOISHI_COMPWISE_OP( ==, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( >, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( >=, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( <, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( <=, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( fn, name, ... )                                                                                                                                              \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type name( const T &a, const T &b )                                                                                                \
	{                                                                                                                                                                                    \
		return { fn( a.x, b.x ) };                                                                                                                                                       \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type name( const T &a, const T &b )                                                                                                \
	{                                                                                                                                                                                    \
		return { fn( a.x, b.x ), fn( a.y, b.y ) };                                                                                                                                       \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type name( const T &a, const T &b )                                                                                                \
	{                                                                                                                                                                                    \
		return { fn( a.x, b.x ), fn( a.y, b.y ), fn( a.z, b.z ) };                                                                                                                       \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type name( const T &a, const T &b )                                                                                                \
	{                                                                                                                                                                                    \
		return { fn( a.x, b.x ), fn( a.y, b.y ), fn( a.z, b.z ), fn( a.w, b.w ) };                                                                                                       \
	}

KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::pow, pow, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::atan2, atan2, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( __func::max, max, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( __func::min, min, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( ... )                                                                                                                                                        \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type dot( const T &a, const T &b )                                                                                                \
	{                                                                                                                                                                                    \
		return a.x * b.x;                                                                                                                                                                \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type dot( const T &a, const T &b )                                                                                                \
	{                                                                                                                                                                                    \
		return a.x * b.x + a.y * b.y;                                                                                                                                                    \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type dot( const T &a, const T &b )                                                                                                \
	{                                                                                                                                                                                    \
		return a.x * b.x + a.y * b.y + a.z * b.z;                                                                                                                                        \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type dot( const T &a, const T &b )                                                                                                \
	{                                                                                                                                                                                    \
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;                                                                                                                            \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type dot( const typename unnormalized_type<T>::type &a, const T &b )                                                              \
	{                                                                                                                                                                                    \
		return a.x * b.x;                                                                                                                                                                \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type dot( const typename unnormalized_type<T>::type &a, const T &b )                                                              \
	{                                                                                                                                                                                    \
		return a.x * b.x + a.y * b.y;                                                                                                                                                    \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type dot( const typename unnormalized_type<T>::type &a, const T &b )                                                              \
	{                                                                                                                                                                                    \
		return a.x * b.x + a.y * b.y + a.z * b.z;                                                                                                                                        \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type dot( const typename unnormalized_type<T>::type &a, const T &b )                                                              \
	{                                                                                                                                                                                    \
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;                                                                                                                            \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type dot( const T &a, const typename unnormalized_type<T>::type &b )                                                              \
	{                                                                                                                                                                                    \
		return a.x * b.x;                                                                                                                                                                \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type dot( const T &a, const typename unnormalized_type<T>::type &b )                                                              \
	{                                                                                                                                                                                    \
		return a.x * b.x + a.y * b.y;                                                                                                                                                    \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type dot( const T &a, const typename unnormalized_type<T>::type &b )                                                              \
	{                                                                                                                                                                                    \
		return a.x * b.x + a.y * b.y + a.z * b.z;                                                                                                                                        \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type dot( const T &a, const typename unnormalized_type<T>::type &b )                                                              \
	{                                                                                                                                                                                    \
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;                                                                                                                            \
	}

KOISHI_COMPWISE_OP( KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( ... )                                                                                                                                               \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type> \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type cross( const T &a, const T &b )                                                                                      \
	{                                                                                                                                                                           \
		return { a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y };                                                                                         \
	}                                                                                                                                                                           \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type> \
	KOISHI_HOST_DEVICE T cross( const T &a, const typename normalized_type<T>::type &b )                                                                                        \
	{                                                                                                                                                                           \
		return { a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y };                                                                                         \
	}                                                                                                                                                                           \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type> \
	KOISHI_HOST_DEVICE T cross( const typename normalized_type<T>::type &a, const T &b )                                                                                        \
	{                                                                                                                                                                           \
		return { a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y };                                                                                         \
	}

KOISHI_COMPWISE_OP( KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( ... )                                                                                                                                                        \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE T reflect( const T &I, const typename normalized_type<T>::type &N )                                                                                               \
	{                                                                                                                                                                                    \
		return T( N * dot( static_cast<const T &>( N ), I ) * typename koishi::trait::com<T>::type( 2 ) - I );                                                                           \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE T reflect( const T &I, const typename normalized_type<T>::type &N )                                                                                               \
	{                                                                                                                                                                                    \
		return T( N * dot( static_cast<const T &>( N ), I ) * typename koishi::trait::com<T>::type( 2 ) - I );                                                                           \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE T reflect( const T &I, const typename normalized_type<T>::type &N )                                                                                               \
	{                                                                                                                                                                                    \
		return T( N * dot( static_cast<const T &>( N ), I ) * typename koishi::trait::com<T>::type( 2 ) - I );                                                                           \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE T reflect( const T &I, const typename normalized_type<T>::type &N )                                                                                               \
	{                                                                                                                                                                                    \
		return T( N * dot( static_cast<const T &>( N ), I ) * typename koishi::trait::com<T>::type( 2 ) - I );                                                                           \
	}

KOISHI_COMPWISE_OP( KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( ... )                                                                                                                                                        \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE bool refract( const T &I, typename normalized_type<T>::type &R, const typename normalized_type<T>::type &N, float eta )                                           \
	{                                                                                                                                                                                    \
		float costh = dot( static_cast<const T &>( N ), I );                                                                                                                             \
		float sinth2 = __func::max( 0., 1. - costh * costh );                                                                                                                            \
		float sinphi2 = sinth2 * eta * eta;                                                                                                                                              \
		if ( sinphi2 >= 1. ) return false;                                                                                                                                               \
		float cosphi = sqrt( 1 - sinphi2 );                                                                                                                                              \
		R = normalize( eta * -I + ( eta * costh - cosphi ) * N );                                                                                                                        \
		return true;                                                                                                                                                                     \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE bool refract( const T &I, typename normalized_type<T>::type &R, const typename normalized_type<T>::type &N, float eta )                                           \
	{                                                                                                                                                                                    \
		float costh = dot( static_cast<const T &>( N ), I );                                                                                                                             \
		float sinth2 = __func::max( 0., 1. - costh * costh );                                                                                                                            \
		float sinphi2 = sinth2 * eta * eta;                                                                                                                                              \
		if ( sinphi2 >= 1. ) return false;                                                                                                                                               \
		float cosphi = sqrt( 1 - sinphi2 );                                                                                                                                              \
		R = normalize( eta * -I + ( eta * costh - cosphi ) * N );                                                                                                                        \
		return true;                                                                                                                                                                     \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE bool refract( const T &I, typename normalized_type<T>::type &R, const typename normalized_type<T>::type &N, float eta )                                           \
	{                                                                                                                                                                                    \
		float costh = dot( static_cast<const T &>( N ), I );                                                                                                                             \
		float sinth2 = __func::max( 0., 1. - costh * costh );                                                                                                                            \
		float sinphi2 = sinth2 * eta * eta;                                                                                                                                              \
		if ( sinphi2 >= 1. ) return false;                                                                                                                                               \
		float cosphi = sqrt( 1 - sinphi2 );                                                                                                                                              \
		R = normalize( eta * -I + ( eta * costh - cosphi ) * N );                                                                                                                        \
		return true;                                                                                                                                                                     \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE bool refract( const T &I, typename normalized_type<T>::type &R, const typename normalized_type<T>::type &N, float eta )                                           \
	{                                                                                                                                                                                    \
		float costh = dot( static_cast<const T &>( N ), I );                                                                                                                             \
		float sinth2 = __func::max( 0., 1. - costh * costh );                                                                                                                            \
		float sinphi2 = sinth2 * eta * eta;                                                                                                                                              \
		if ( sinphi2 >= 1. ) return false;                                                                                                                                               \
		float cosphi = sqrt( 1 - sinphi2 );                                                                                                                                              \
		R = normalize( eta * -I + ( eta * costh - cosphi ) * N );                                                                                                                        \
		return true;                                                                                                                                                                     \
	}

KOISHI_COMPWISE_OP( KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( fn, name, ... )                                                                                                                                              \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type name( const T &a )                                                                                                            \
	{                                                                                                                                                                                    \
		return { fn( a.x ) };                                                                                                                                                            \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type name( const T &a )                                                                                                            \
	{                                                                                                                                                                                    \
		return { fn( a.x ), fn( a.y ) };                                                                                                                                                 \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type name( const T &a )                                                                                                            \
	{                                                                                                                                                                                    \
		return { fn( a.x ), fn( a.y ), fn( a.z ) };                                                                                                                                      \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename unnormalized_type<T>::type name( const T &a )                                                                                                            \
	{                                                                                                                                                                                    \
		return { fn( a.x ), fn( a.y ), fn( a.z ), fn( a.w ) };                                                                                                                           \
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

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( sqrt, ... )                                                                                                                                                  \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type squaredLength( const T &a )                                                                                                  \
	{                                                                                                                                                                                    \
		return ( a.x * a.x );                                                                                                                                                            \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type squaredLength( const T &a )                                                                                                  \
	{                                                                                                                                                                                    \
		return ( a.x * a.x + a.y * a.y );                                                                                                                                                \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type squaredLength( const T &a )                                                                                                  \
	{                                                                                                                                                                                    \
		return ( a.x * a.x + a.y * a.y + a.z * a.z );                                                                                                                                    \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type squaredLength( const T &a )                                                                                                  \
	{                                                                                                                                                                                    \
		return ( a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w );                                                                                                                        \
	}

KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::sqrt, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( sqrt, ... )                                                                                                                                                  \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type length( const T &a )                                                                                                         \
	{                                                                                                                                                                                    \
		return sqrt( a.x * a.x );                                                                                                                                                        \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type length( const T &a )                                                                                                         \
	{                                                                                                                                                                                    \
		return sqrt( a.x * a.x + a.y * a.y );                                                                                                                                            \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type length( const T &a )                                                                                                         \
	{                                                                                                                                                                                    \
		return sqrt( a.x * a.x + a.y * a.y + a.z * a.z );                                                                                                                                \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename koishi::trait::com<T>::type length( const T &a )                                                                                                         \
	{                                                                                                                                                                                    \
		return sqrt( a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w );                                                                                                                    \
	}

KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::sqrt, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( rsqrt, ... )                                                                                                                                                 \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	KOISHI_HOST_DEVICE typename normalized_type<T>::type normalize( const T &a )                                                                                                         \
	{                                                                                                                                                                                    \
		return typename normalized_type<T>::type( a * rsqrt( a.x * a.x ) );                                                                                                              \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	KOISHI_HOST_DEVICE typename normalized_type<T>::type normalize( const T &a )                                                                                                         \
	{                                                                                                                                                                                    \
		return typename normalized_type<T>::type( a * rsqrt( a.x * a.x + a.y * a.y ) );                                                                                                  \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	KOISHI_HOST_DEVICE typename normalized_type<T>::type normalize( const T &a )                                                                                                         \
	{                                                                                                                                                                                    \
		return typename normalized_type<T>::type( a * rsqrt( a.x * a.x + a.y * a.y + a.z * a.z ) );                                                                                      \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	KOISHI_HOST_DEVICE typename normalized_type<T>::type normalize( const T &a )                                                                                                         \
	{                                                                                                                                                                                    \
		return typename normalized_type<T>::type( a * rsqrt( a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w ) );                                                                          \
	}

#if defined( KOISHI_USE_CUDA )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::rsqrt, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
#else
KOISHI_COMPWISE_OP( __func::rsqrt, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
#endif

#undef KOISHI_MATH_NAMESP

#undef KOISHI_COMPWISE_OP

#if !defined( KOISHI_USE_CUDA )

}  // namespace vec

}  // namespace koishi

using namespace koishi::vec;
using namespace koishi::vec::__func;

#else

using namespace __func;

#endif
