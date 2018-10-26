#pragma once

#include <type_traits>

#if !defined( CUDA )
#include <cmath>
#endif

#ifdef KOISHI_THINKPAD_X1C  // for intellisense
#include <cuda_runtime.h>
#endif

namespace koishi
{
namespace vm
{
#if defined( CUDA )

#define KOISHI_HOST_DEVICE \
	__host__ __device__
#define KOISHI_MATH_NAMESP

#else

#define KOISHI_HOST_DEVICE \
	inline
/* clang-format off */
	#define KOISHI_DEF_VEC(type) \
		struct type##1 { type x; };\
		struct type##2 { type x, y; };\
		struct type##3 { type x, y, z; };\
		struct type##4 { type x, y, z, w; }
/* clang-format on */
using uint = unsigned int;
KOISHI_DEF_VEC( float );
KOISHI_DEF_VEC( double );
KOISHI_DEF_VEC( int );
KOISHI_DEF_VEC( uint );
#undef KOISHI_DEF_VEC
#define KOISHI_MATH_NAMESP \
	std

#endif

namespace __trait
{
template <typename T, typename X, typename... R>
struct is_in
{
	constexpr static bool value = std::is_same<T, X>::value || is_in<T, R...>::value;
};
template <typename T, typename X>
struct is_in<T, X>
{
	constexpr static bool value = std::is_same<T, X>::value;
};

template <typename... T>
struct make_void
{
	using type = void;
};
template <typename... T>
using void_t = typename make_void<T...>::type;

#define KOISHI_CHECK_HAS_COMP( comp )                                           \
	template <typename T, typename = void>                                      \
	struct has_##comp                                                           \
	{                                                                           \
		static constexpr bool value = false;                                    \
	};                                                                          \
	template <typename T>                                                       \
	struct has_##comp<T, __trait::void_t<decltype( std::declval<T &>().comp )>> \
	{                                                                           \
		static constexpr bool value = true;                                     \
	}
KOISHI_CHECK_HAS_COMP( x );
KOISHI_CHECK_HAS_COMP( y );
KOISHI_CHECK_HAS_COMP( z );
KOISHI_CHECK_HAS_COMP( w );

#undef KOISHI_CHECK_HAS_COMP

template <typename T>
struct is_vec1
{
	static constexpr bool value = has_x<T>::value && !has_y<T>::value;
};

template <typename T>
struct is_vec2
{
	static constexpr bool value = has_y<T>::value && !has_z<T>::value;
};

template <typename T>
struct is_vec3
{
	static constexpr bool value = has_z<T>::value && !has_w<T>::value;
};

template <typename T>
struct is_vec4
{
	static constexpr bool value = has_w<T>::value;
};

template <typename T>
struct com
{
	using type = decltype( std::declval<T &>().x );
};

}  // namespace __trait

#define KOISHI_VEC_FLOAT float1, float2, float3, float4
#define KOISHI_VEC_DOUBLE double1, double2, double3, double4
#define KOISHI_VEC_INT int1, int2, int3, int4
#define KOISHI_VEC_UINT uint1, uint2, uint3, uint4

namespace __func
{
template <typename T>
KOISHI_HOST_DEVICE T min( T a, T b )
{
	return a < b ? a : b;
}

template <typename T>
KOISHI_HOST_DEVICE T max( T a, T b )
{
	return a > b ? a : b;
}

#if !defined( CUDA )
template <typename T>
KOISHI_HOST_DEVICE T rsqrt( T a )
{
	return T( 1.0 ) / KOISHI_MATH_NAMESP::sqrt( a );
}

#endif

}  // namespace __func

#define KOISHI_COMPWISE_OP( op, ... )                                                                 \
	namespace vec1                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a, const T &b )                                        \
	{                                                                                                 \
		return { a.x op b.x };                                                                        \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec2                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a, const T &b )                                        \
	{                                                                                                 \
		return { a.x op b.x, a.y op b.y };                                                            \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec3                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a, const T &b )                                        \
	{                                                                                                 \
		return { a.x op b.x, a.y op b.y, a.z op b.z };                                                \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec4                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T operator op( const T &a, const T &b )                                        \
	{                                                                                                 \
		return { a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w };                                    \
	}                                                                                                 \
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
#define KOISHI_COMPWISE_OP( op, ... )                                                                 \
	namespace vec1                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T &operator op( T &a, const T &b )                                             \
	{                                                                                                 \
		return a.x op b.x, a;                                                                         \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec2                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T &operator op( T &a, const T &b )                                             \
	{                                                                                                 \
		return a.x op b.x, a.y op b.y, a;                                                             \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec3                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T &operator op( T &a, const T &b )                                             \
	{                                                                                                 \
		return a.x op b.x, a.y op b.y, a.z op b.z, a;                                                 \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec4                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T &operator op( T &a, const T &b )                                             \
	{                                                                                                 \
		return a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w, a;                                     \
	}                                                                                                 \
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
#define KOISHI_COMPWISE_OP( fn, name, ... )                                                           \
	namespace vec1                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a, const T &b )                                               \
	{                                                                                                 \
		return { fn( a.x, b.x ) };                                                                    \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec2                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a, const T &b )                                               \
	{                                                                                                 \
		return { fn( a.x, b.x ), fn( a.y, b.y ) };                                                    \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec3                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a, const T &b )                                               \
	{                                                                                                 \
		return { fn( a.x, b.x ), fn( a.y, b.y ), fn( a.z, b.z ) };                                    \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec4                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a, const T &b )                                               \
	{                                                                                                 \
		return { fn( a.x, b.x ), fn( a.y, b.y ), fn( a.z, b.z ), fn( a.w, b.w ) };                    \
	}                                                                                                 \
	}

KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::pow, pow, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::atan2, atan2, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( __func::max, max, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )
KOISHI_COMPWISE_OP( __func::min, min, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( ... )                                                                     \
	namespace vec1                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE typename __trait::com<T>::type dot( const T &a, const T &b )                   \
	{                                                                                                 \
		return a.x * b.x;                                                                             \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec2                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE typename __trait::com<T>::type dot( const T &a, const T &b )                   \
	{                                                                                                 \
		return a.x * b.x + a.y * b.y;                                                                 \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec3                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE typename __trait::com<T>::type dot( const T &a, const T &b )                   \
	{                                                                                                 \
		return a.x * b.x + a.y * b.y + a.z * b.z;                                                     \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec4                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE typename __trait::com<T>::type dot( const T &a, const T &b )                   \
	{                                                                                                 \
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;                                         \
	}                                                                                                 \
	}

KOISHI_COMPWISE_OP( KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( ... )                                                                     \
	namespace vec3                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T cross( const T &a, const T &b )                                              \
	{                                                                                                 \
		return { a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y };               \
	}                                                                                                 \
	}

KOISHI_COMPWISE_OP( KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

#undef KOISHI_COMPWISE_OP
#define KOISHI_COMPWISE_OP( fn, name, ... )                                                           \
	namespace vec1                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec1<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a )                                                           \
	{                                                                                                 \
		return { fn( a.x ) };                                                                         \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec2                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec2<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a )                                                           \
	{                                                                                                 \
		return { fn( a.x ), fn( a.y ) };                                                              \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec3                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec3<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a )                                                           \
	{                                                                                                 \
		return { fn( a.x ), fn( a.y ), fn( a.z ) };                                                   \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec4                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec4<T>::value>::type>       \
	KOISHI_HOST_DEVICE T name( const T &a )                                                           \
	{                                                                                                 \
		return { fn( a.x ), fn( a.y ), fn( a.z ), fn( a.w ) };                                        \
	}                                                                                                 \
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
#if defined( CUDA )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::rsqrt, rsqrt, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
#else
KOISHI_COMPWISE_OP( __func::rsqrt, rsqrt, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
#endif
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::abs, abs, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::floor, floor, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )
KOISHI_COMPWISE_OP( KOISHI_MATH_NAMESP::ceil, ceil, KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE )

using namespace vec1;
using namespace vec2;
using namespace vec3;
using namespace vec4;

#undef KOISHI_MATH_NAMESP

#undef KOISHI_COMPWISE_OP
#undef KOISHI_HOST_DEVICE

#undef KOISHI_VEC_FLOAT
#undef KOISHI_VEC_DOUBLE
#undef KOISHI_VEC_INT
#undef KOISHI_VEC_UINT

}  // namespace vm

}  // namespace koishi

using namespace koishi::vm;