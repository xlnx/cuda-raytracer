#pragma once

#include <type_traits>

#ifdef KOISHI_THINKPAD_X1C  // for intellisense
#include <cuda_runtime.h>
#endif

namespace koishi
{
namespace cuda
{
namespace __func
{
template <typename T>
__host__ __device__ T min( T a, T b )
{
	return a < b ? a : b;
}

template <typename T>
__host__ __device__ T max( T a, T b )
{
	return a > b ? a : b;
}

}  // namespace __func

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

#define KOISHI_CUDA_CHECK_HAS_COMP( comp )                                      \
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
KOISHI_CUDA_CHECK_HAS_COMP( x );
KOISHI_CUDA_CHECK_HAS_COMP( y );
KOISHI_CUDA_CHECK_HAS_COMP( z );
KOISHI_CUDA_CHECK_HAS_COMP( w );

#undef KOISHI_CUDA_CHECK_HAS_COMP

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

#define KOISHI_CUDA_VEC_FLOAT float1, float2, float3, float4
#define KOISHI_CUDA_VEC_DOUBLE double1, double2, double3, double4
#define KOISHI_CUDA_VEC_INT int1, int2, int3, int4
#define KOISHI_CUDA_VEC_UINT uint1, uint2, uint3, uint4

#define KOISHI_CUDA_COMPWISE_OP( op, ... )                                                            \
	namespace vec1                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec1<T>::value>::type>       \
	__host__ __device__ T operator op( const T &a, const T &b )                                       \
	{                                                                                                 \
		return { a.x op b.x };                                                                        \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec2                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec2<T>::value>::type>       \
	__host__ __device__ T operator op( const T &a, const T &b )                                       \
	{                                                                                                 \
		return { a.x op b.x, a.y op b.y };                                                            \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec3                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec3<T>::value>::type>       \
	__host__ __device__ T operator op( const T &a, const T &b )                                       \
	{                                                                                                 \
		return { a.x op b.x, a.y op b.y, a.z op b.z };                                                \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec4                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec4<T>::value>::type>       \
	__host__ __device__ T operator op( const T &a, const T &b )                                       \
	{                                                                                                 \
		return { a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w };                                    \
	}                                                                                                 \
	}

KOISHI_CUDA_COMPWISE_OP( +, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( -, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( *, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( /, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( %, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( &, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( |, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( <<, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( >>, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )

#undef KOISHI_CUDA_COMPWISE_OP
#define KOISHI_CUDA_COMPWISE_OP( op, ... )                                                            \
	namespace vec1                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec1<T>::value>::type>       \
	__host__ __device__ T &operator op( T &a, const T &b )                                            \
	{                                                                                                 \
		return a.x op b.x, a;                                                                         \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec2                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec2<T>::value>::type>       \
	__host__ __device__ T &operator op( T &a, const T &b )                                            \
	{                                                                                                 \
		return a.x op b.x, a.y op b.y, a;                                                             \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec3                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec3<T>::value>::type>       \
	__host__ __device__ T &operator op( T &a, const T &b )                                            \
	{                                                                                                 \
		return a.x op b.x, a.y op b.y, a.z op b.z, a;                                                 \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec4                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec4<T>::value>::type>       \
	__host__ __device__ T &operator op( T &a, const T &b )                                            \
	{                                                                                                 \
		return a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w, a;                                     \
	}                                                                                                 \
	}

KOISHI_CUDA_COMPWISE_OP( +=, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( -=, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( *=, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( /=, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( %=, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( &=, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( |=, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( <<=, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( >>=, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )

#undef KOISHI_CUDA_COMPWISE_OP
#define KOISHI_CUDA_COMPWISE_OP( namesp, fn, ... )                                                                 \
	namespace vec1                                                                                                 \
	{                                                                                                              \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value &&              \
															 __trait::is_vec1<T>::value>::type>                    \
	__host__ __device__ T fn( const T &a, const T &b )                                                             \
	{                                                                                                              \
		return { namesp::fn( a.x, b.x ) };                                                                         \
	}                                                                                                              \
	}                                                                                                              \
	namespace vec2                                                                                                 \
	{                                                                                                              \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value &&              \
															 __trait::is_vec2<T>::value>::type>                    \
	__host__ __device__ T fn( const T &a, const T &b )                                                             \
	{                                                                                                              \
		return { namesp::fn( a.x, b.x ), namesp::fn( a.y, b.y ) };                                                 \
	}                                                                                                              \
	}                                                                                                              \
	namespace vec3                                                                                                 \
	{                                                                                                              \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value &&              \
															 __trait::is_vec3<T>::value>::type>                    \
	__host__ __device__ T fn( const T &a, const T &b )                                                             \
	{                                                                                                              \
		return { namesp::fn( a.x, b.x ), namesp::fn( a.y, b.y ), namesp::fn( a.z, b.z ) };                         \
	}                                                                                                              \
	}                                                                                                              \
	namespace vec4                                                                                                 \
	{                                                                                                              \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value &&              \
															 __trait::is_vec4<T>::value>::type>                    \
	__host__ __device__ T fn( const T &a, const T &b )                                                             \
	{                                                                                                              \
		return { namesp::fn( a.x, b.x ), namesp::fn( a.y, b.y ), namesp::fn( a.z, b.z ), namesp::fn( a.w, b.w ) }; \
	}                                                                                                              \
	}

KOISHI_CUDA_COMPWISE_OP(, pow, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, atan2, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP( __func, max, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )
KOISHI_CUDA_COMPWISE_OP( __func, min, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE, KOISHI_CUDA_VEC_INT, KOISHI_CUDA_VEC_UINT )

#undef KOISHI_CUDA_COMPWISE_OP
#define KOISHI_CUDA_COMPWISE_OP( ... )                                                                \
	namespace vec1                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec1<T>::value>::type>       \
	__host__ __device__ typename __trait::com<T>::type dot( const T &a, const T &b )                  \
	{                                                                                                 \
		return a.x * b.x;                                                                             \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec2                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec2<T>::value>::type>       \
	__host__ __device__ typename __trait::com<T>::type dot( const T &a, const T &b )                  \
	{                                                                                                 \
		return a.x * b.x + a.y * b.y;                                                                 \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec3                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec3<T>::value>::type>       \
	__host__ __device__ typename __trait::com<T>::type dot( const T &a, const T &b )                  \
	{                                                                                                 \
		return a.x * b.x + a.y * b.y + a.z * b.z;                                                     \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec4                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec4<T>::value>::type>       \
	__host__ __device__ typename __trait::com<T>::type dot( const T &a, const T &b )                  \
	{                                                                                                 \
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;                                         \
	}                                                                                                 \
	}

KOISHI_CUDA_COMPWISE_OP( KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )

#undef KOISHI_CUDA_COMPWISE_OP
#define KOISHI_CUDA_COMPWISE_OP( ... )                                                                \
	namespace vec3                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec3<T>::value>::type>       \
	__host__ __device__ T cross( const T &a, const T &b )                                             \
	{                                                                                                 \
		return { a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y };               \
	}                                                                                                 \
	}

KOISHI_CUDA_COMPWISE_OP( KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )

#undef KOISHI_CUDA_COMPWISE_OP
#define KOISHI_CUDA_COMPWISE_OP( namesp, fn, ... )                                                    \
	namespace vec1                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec1<T>::value>::type>       \
	__host__ __device__ T fn( const T &a )                                                            \
	{                                                                                                 \
		return { namesp::fn( a.x ) };                                                                 \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec2                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec2<T>::value>::type>       \
	__host__ __device__ T fn( const T &a )                                                            \
	{                                                                                                 \
		return { namesp::fn( a.x ), namesp::fn( a.y ) };                                              \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec3                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec3<T>::value>::type>       \
	__host__ __device__ T fn( const T &a )                                                            \
	{                                                                                                 \
		return { namesp::fn( a.x ), namesp::fn( a.y ), namesp::fn( a.z ) };                           \
	}                                                                                                 \
	}                                                                                                 \
	namespace vec4                                                                                    \
	{                                                                                                 \
	template <typename T, typename = typename std::enable_if<__trait::is_in<T, __VA_ARGS__>::value && \
															 __trait::is_vec4<T>::value>::type>       \
	__host__ __device__ T fn( const T &a )                                                            \
	{                                                                                                 \
		return { namesp::fn( a.x ), namesp::fn( a.y ), namesp::fn( a.z ), namesp::fn( a.w ) };        \
	}                                                                                                 \
	}

KOISHI_CUDA_COMPWISE_OP(, sin, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, cos, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, tan, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, asin, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, acos, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, atan, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, exp, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, log, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, exp2, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, log2, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, sqrt, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, rsqrt, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, abs, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, floor, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )
KOISHI_CUDA_COMPWISE_OP(, ceil, KOISHI_CUDA_VEC_FLOAT, KOISHI_CUDA_VEC_DOUBLE )

using namespace vec1;
using namespace vec2;
using namespace vec3;
using namespace vec4;

#undef KOISHI_CUDA_COMPWISE_OP

#undef KOISHI_CUDA_VEC_FLOAT
#undef KOISHI_CUDA_VEC_DOUBLE
#undef KOISHI_CUDA_VEC_INT
#undef KOISHI_CUDA_VEC_UINT

}  // namespace cuda

}  // namespace koishi