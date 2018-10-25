#pragma once

#ifdef KOISHI_THINKPAD_X1C  // for intellisense
#include <cuda_runtime.h>
#endif

namespace koishi
{
namespace cuda
{
#define KOISHI_CUDA_COMPWISE1( T, op )                          \
	__host__ __device__ T operator op( const T &a, const T &b ) \
	{                                                           \
		return T{ a.x op b.x };                                 \
	}
#define KOISHI_CUDA_COMPWISE_UNAFN1( T, namesp, fn ) \
	__host__ __device__ T fn( const T &a )           \
	{                                                \
		return T{ namesp::fn( a.x ) };               \
	}
#define KOISHI_CUDA_COMPWISE_BINFN1( T, namesp, fn )   \
	__host__ __device__ T fn( const T &a, const T &b ) \
	{                                                  \
		return T{ namesp::fn( a.x, b.x ) };            \
	}
#define KOISHI_CUDA_COMPWISEASS1( T, op )                  \
	__host__ __device__ T &operator op( T &a, const T &b ) \
	{                                                      \
		return a.x op b.x, a;                              \
	}
#define KOISHI_CUDA_COMPWISE2( T, op )                          \
	__host__ __device__ T operator op( const T &a, const T &b ) \
	{                                                           \
		return T{ a.x op b.x, a.y op b.y };                     \
	}
#define KOISHI_CUDA_COMPWISE_UNAFN2( T, namesp, fn )      \
	__host__ __device__ T fn( const T &a )                \
	{                                                     \
		return T{ namesp::fn( a.x ), namesp::fn( a.y ) }; \
	}
#define KOISHI_CUDA_COMPWISE_BINFN2( T, namesp, fn )                \
	__host__ __device__ T fn( const T &a, const T &b )              \
	{                                                               \
		return T{ namesp::fn( a.x, b.x ), namesp::fn( a.y, b.y ) }; \
	}
#define KOISHI_CUDA_COMPWISEASS2( T, op )                  \
	__host__ __device__ T &operator op( T &a, const T &b ) \
	{                                                      \
		return a.x op b.x, a.y op b.y, a;                  \
	}
#define KOISHI_CUDA_COMPWISE3( T, op )                          \
	__host__ __device__ T operator op( const T &a, const T &b ) \
	{                                                           \
		return T{ a.x op b.x, a.y op b.y, a.z op b.z };         \
	}
#define KOISHI_CUDA_COMPWISE_UNAFN3( T, namesp, fn )                         \
	__host__ __device__ T fn( const T &a )                                   \
	{                                                                        \
		return T{ namesp::fn( a.x ), namesp::fn( a.y ), namesp::fn( a.z ) }; \
	}
#define KOISHI_CUDA_COMPWISE_BINFN3( T, namesp, fn )                                        \
	__host__ __device__ T fn( const T &a, const T &b )                                      \
	{                                                                                       \
		return T{ namesp::fn( a.x, b.x ), namesp::fn( a.y, b.y ), namesp::fn( a.z, b.z ) }; \
	}
#define KOISHI_CUDA_COMPWISEASS3( T, op )                  \
	__host__ __device__ T &operator op( T &a, const T &b ) \
	{                                                      \
		return a.x op b.x, a.y op b.y, a.z op b.z, a;      \
	}
#define KOISHI_CUDA_COMPWISE4( T, op )                              \
	__host__ __device__ T operator op( const T &a, const T &b )     \
	{                                                               \
		return T{ a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w }; \
	}
#define KOISHI_CUDA_COMPWISE_UNAFN4( T, namesp, fn )                                            \
	__host__ __device__ T fn( const T &a, const T &b )                                          \
	{                                                                                           \
		return T{ namesp::fn( a.x ), namesp::fn( a.y ), namesp::fn( a.z ), namesp::fn( a.w ) }; \
	}
#define KOISHI_CUDA_COMPWISE_BINFN4( T, namesp, fn )                                                                \
	__host__ __device__ T fn( const T &a, const T &b )                                                              \
	{                                                                                                               \
		return T{ namesp::fn( a.x, b.x ), namesp::fn( a.y, b.y ), namesp::fn( a.z, b.z ), namesp::fn( a.w, b.w ) }; \
	}
#define KOISHI_CUDA_COMPWISEASS4( T, op )                         \
	__host__ __device__ T &operator op( T &a, const T &b )        \
	{                                                             \
		return a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w, a; \
	}

namespace funcs
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

}  // namespace funcs

KOISHI_CUDA_COMPWISE1( float1, +)
KOISHI_CUDA_COMPWISE1( float1, -)
KOISHI_CUDA_COMPWISE1( float1, * )
KOISHI_CUDA_COMPWISE1( float1, / )

KOISHI_CUDA_COMPWISE2( float2, +)
KOISHI_CUDA_COMPWISE2( float2, -)
KOISHI_CUDA_COMPWISE2( float2, * )
KOISHI_CUDA_COMPWISE2( float2, / )

KOISHI_CUDA_COMPWISE3( float3, +)
KOISHI_CUDA_COMPWISE3( float3, -)
KOISHI_CUDA_COMPWISE3( float3, * )
KOISHI_CUDA_COMPWISE3( float3, / )

KOISHI_CUDA_COMPWISE4( float4, +)
KOISHI_CUDA_COMPWISE4( float4, -)
KOISHI_CUDA_COMPWISE4( float4, * )
KOISHI_CUDA_COMPWISE4( float4, / )

KOISHI_CUDA_COMPWISEASS1( float1, += )
KOISHI_CUDA_COMPWISEASS1( float1, -= )
KOISHI_CUDA_COMPWISEASS1( float1, *= )
KOISHI_CUDA_COMPWISEASS1( float1, /= )

KOISHI_CUDA_COMPWISEASS2( float2, += )
KOISHI_CUDA_COMPWISEASS2( float2, -= )
KOISHI_CUDA_COMPWISEASS2( float2, *= )
KOISHI_CUDA_COMPWISEASS2( float2, /= )

KOISHI_CUDA_COMPWISEASS3( float3, += )
KOISHI_CUDA_COMPWISEASS3( float3, -= )
KOISHI_CUDA_COMPWISEASS3( float3, *= )
KOISHI_CUDA_COMPWISEASS3( float3, /= )

KOISHI_CUDA_COMPWISEASS4( float4, += )
KOISHI_CUDA_COMPWISEASS4( float4, -= )
KOISHI_CUDA_COMPWISEASS4( float4, *= )
KOISHI_CUDA_COMPWISEASS4( float4, /= )

KOISHI_CUDA_COMPWISE_BINFN1( float1, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN1( float1, funcs, min )

KOISHI_CUDA_COMPWISE_BINFN2( float2, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN2( float2, funcs, min )

KOISHI_CUDA_COMPWISE_BINFN3( float3, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN3( float3, funcs, min )

KOISHI_CUDA_COMPWISE_BINFN4( float4, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN4( float4, funcs, min )

KOISHI_CUDA_COMPWISE_UNAFN1( float1, , sin )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , cos )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , tan )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , asin )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , acos )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , atan )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , exp )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , log )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , exp2 )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , log2 )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , sqrt )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , rsqrt )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , abs )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , floor )
KOISHI_CUDA_COMPWISE_UNAFN1( float1, , ceil )
KOISHI_CUDA_COMPWISE_BINFN1( float1, , pow )
KOISHI_CUDA_COMPWISE_BINFN1( float1, , atan2 )

KOISHI_CUDA_COMPWISE_UNAFN2( float2, , sin )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , cos )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , tan )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , asin )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , acos )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , atan )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , exp )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , log )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , exp2 )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , log2 )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , sqrt )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , rsqrt )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , abs )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , floor )
KOISHI_CUDA_COMPWISE_UNAFN2( float2, , ceil )
KOISHI_CUDA_COMPWISE_BINFN2( float2, , pow )
KOISHI_CUDA_COMPWISE_BINFN2( float2, , atan2 )

KOISHI_CUDA_COMPWISE_UNAFN3( float3, , sin )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , cos )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , tan )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , asin )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , acos )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , atan )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , exp )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , log )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , exp2 )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , log2 )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , sqrt )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , rsqrt )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , abs )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , floor )
KOISHI_CUDA_COMPWISE_UNAFN3( float3, , ceil )
KOISHI_CUDA_COMPWISE_BINFN3( float3, , pow )
KOISHI_CUDA_COMPWISE_BINFN3( float3, , atan2 )

KOISHI_CUDA_COMPWISE_UNAFN4( float4, , sin )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , cos )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , tan )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , asin )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , acos )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , atan )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , exp )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , log )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , exp2 )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , log2 )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , sqrt )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , rsqrt )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , abs )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , floor )
KOISHI_CUDA_COMPWISE_UNAFN4( float4, , ceil )
KOISHI_CUDA_COMPWISE_BINFN4( float4, , pow )
KOISHI_CUDA_COMPWISE_BINFN4( float4, , atan2 )

__host__ __device__ float dot( const float1 &a, const float1 &b )
{
	return a.x * b.x;
}
__host__ __device__ float dot( const float2 &a, const float2 &b )
{
	return a.x * b.x + a.y * b.y;
}
__host__ __device__ float dot( const float3 &a, const float3 &b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ float dot( const float4 &a, const float4 &b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
__host__ __device__ float3 cross( const float3 &a, const float4 &b )
{
	return float3{ a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y };
}

KOISHI_CUDA_COMPWISE1( double1, +)
KOISHI_CUDA_COMPWISE1( double1, -)
KOISHI_CUDA_COMPWISE1( double1, * )
KOISHI_CUDA_COMPWISE1( double1, / )

KOISHI_CUDA_COMPWISE2( double2, +)
KOISHI_CUDA_COMPWISE2( double2, -)
KOISHI_CUDA_COMPWISE2( double2, * )
KOISHI_CUDA_COMPWISE2( double2, / )

KOISHI_CUDA_COMPWISE3( double3, +)
KOISHI_CUDA_COMPWISE3( double3, -)
KOISHI_CUDA_COMPWISE3( double3, * )
KOISHI_CUDA_COMPWISE3( double3, / )

KOISHI_CUDA_COMPWISE4( double4, +)
KOISHI_CUDA_COMPWISE4( double4, -)
KOISHI_CUDA_COMPWISE4( double4, * )
KOISHI_CUDA_COMPWISE4( double4, / )

KOISHI_CUDA_COMPWISEASS1( double1, += )
KOISHI_CUDA_COMPWISEASS1( double1, -= )
KOISHI_CUDA_COMPWISEASS1( double1, *= )
KOISHI_CUDA_COMPWISEASS1( double1, /= )

KOISHI_CUDA_COMPWISEASS2( double2, += )
KOISHI_CUDA_COMPWISEASS2( double2, -= )
KOISHI_CUDA_COMPWISEASS2( double2, *= )
KOISHI_CUDA_COMPWISEASS2( double2, /= )

KOISHI_CUDA_COMPWISEASS3( double3, += )
KOISHI_CUDA_COMPWISEASS3( double3, -= )
KOISHI_CUDA_COMPWISEASS3( double3, *= )
KOISHI_CUDA_COMPWISEASS3( double3, /= )

KOISHI_CUDA_COMPWISEASS4( double4, += )
KOISHI_CUDA_COMPWISEASS4( double4, -= )
KOISHI_CUDA_COMPWISEASS4( double4, *= )
KOISHI_CUDA_COMPWISEASS4( double4, /= )

KOISHI_CUDA_COMPWISE_BINFN1( double1, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN1( double1, funcs, min )

KOISHI_CUDA_COMPWISE_BINFN2( double2, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN2( double2, funcs, min )

KOISHI_CUDA_COMPWISE_BINFN3( double3, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN3( double3, funcs, min )

KOISHI_CUDA_COMPWISE_BINFN4( double4, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN4( double4, funcs, min )

KOISHI_CUDA_COMPWISE_UNAFN1( double1, , sin )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , cos )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , tan )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , asin )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , acos )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , atan )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , exp )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , log )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , exp2 )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , log2 )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , sqrt )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , rsqrt )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , abs )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , floor )
KOISHI_CUDA_COMPWISE_UNAFN1( double1, , ceil )
KOISHI_CUDA_COMPWISE_BINFN1( double1, , pow )
KOISHI_CUDA_COMPWISE_BINFN1( double1, , atan2 )

KOISHI_CUDA_COMPWISE_UNAFN2( double2, , sin )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , cos )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , tan )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , asin )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , acos )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , atan )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , exp )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , log )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , exp2 )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , log2 )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , sqrt )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , rsqrt )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , abs )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , floor )
KOISHI_CUDA_COMPWISE_UNAFN2( double2, , ceil )
KOISHI_CUDA_COMPWISE_BINFN2( double2, , pow )
KOISHI_CUDA_COMPWISE_BINFN2( double2, , atan2 )

KOISHI_CUDA_COMPWISE_UNAFN3( double3, , sin )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , cos )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , tan )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , asin )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , acos )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , atan )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , exp )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , log )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , exp2 )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , log2 )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , sqrt )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , rsqrt )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , abs )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , floor )
KOISHI_CUDA_COMPWISE_UNAFN3( double3, , ceil )
KOISHI_CUDA_COMPWISE_BINFN3( double3, , pow )
KOISHI_CUDA_COMPWISE_BINFN3( double3, , atan2 )

KOISHI_CUDA_COMPWISE_UNAFN4( double4, , sin )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , cos )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , tan )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , asin )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , acos )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , atan )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , exp )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , log )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , exp2 )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , log2 )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , sqrt )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , rsqrt )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , abs )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , floor )
KOISHI_CUDA_COMPWISE_UNAFN4( double4, , ceil )
KOISHI_CUDA_COMPWISE_BINFN4( double4, , pow )
KOISHI_CUDA_COMPWISE_BINFN4( double4, , atan2 )

__host__ __device__ double dot( const double1 &a, const double1 &b )
{
	return a.x * b.x;
}
__host__ __device__ double dot( const double2 &a, const double2 &b )
{
	return a.x * b.x + a.y * b.y;
}
__host__ __device__ double dot( const double3 &a, const double3 &b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ double dot( const double4 &a, const double4 &b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
__host__ __device__ double3 cross( const double3 &a, const double4 &b )
{
	return double3{ a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y };
}

KOISHI_CUDA_COMPWISE1( int1, +)
KOISHI_CUDA_COMPWISE1( int1, -)
KOISHI_CUDA_COMPWISE1( int1, * )
KOISHI_CUDA_COMPWISE1( int1, / )
KOISHI_CUDA_COMPWISE1( int1, % )
KOISHI_CUDA_COMPWISE1( int1, & )
KOISHI_CUDA_COMPWISE1( int1, | )
KOISHI_CUDA_COMPWISE1( int1, << )
KOISHI_CUDA_COMPWISE1( int1, >> )

KOISHI_CUDA_COMPWISE2( int2, +)
KOISHI_CUDA_COMPWISE2( int2, -)
KOISHI_CUDA_COMPWISE2( int2, * )
KOISHI_CUDA_COMPWISE2( int2, / )
KOISHI_CUDA_COMPWISE2( int2, % )
KOISHI_CUDA_COMPWISE2( int2, & )
KOISHI_CUDA_COMPWISE2( int2, | )
KOISHI_CUDA_COMPWISE2( int2, << )
KOISHI_CUDA_COMPWISE2( int2, >> )

KOISHI_CUDA_COMPWISE3( int3, +)
KOISHI_CUDA_COMPWISE3( int3, -)
KOISHI_CUDA_COMPWISE3( int3, * )
KOISHI_CUDA_COMPWISE3( int3, / )
KOISHI_CUDA_COMPWISE3( int3, % )
KOISHI_CUDA_COMPWISE3( int3, & )
KOISHI_CUDA_COMPWISE3( int3, | )
KOISHI_CUDA_COMPWISE3( int3, << )
KOISHI_CUDA_COMPWISE3( int3, >> )

KOISHI_CUDA_COMPWISE4( int4, +)
KOISHI_CUDA_COMPWISE4( int4, -)
KOISHI_CUDA_COMPWISE4( int4, * )
KOISHI_CUDA_COMPWISE4( int4, / )
KOISHI_CUDA_COMPWISE4( int4, % )
KOISHI_CUDA_COMPWISE4( int4, & )
KOISHI_CUDA_COMPWISE4( int4, | )
KOISHI_CUDA_COMPWISE4( int4, << )
KOISHI_CUDA_COMPWISE4( int4, >> )

KOISHI_CUDA_COMPWISEASS1( int1, += )
KOISHI_CUDA_COMPWISEASS1( int1, -= )
KOISHI_CUDA_COMPWISEASS1( int1, *= )
KOISHI_CUDA_COMPWISEASS1( int1, /= )
KOISHI_CUDA_COMPWISEASS1( int1, %= )
KOISHI_CUDA_COMPWISEASS1( int1, &= )
KOISHI_CUDA_COMPWISEASS1( int1, |= )
KOISHI_CUDA_COMPWISEASS1( int1, <<= )
KOISHI_CUDA_COMPWISEASS1( int1, >>= )

KOISHI_CUDA_COMPWISEASS2( int2, += )
KOISHI_CUDA_COMPWISEASS2( int2, -= )
KOISHI_CUDA_COMPWISEASS2( int2, *= )
KOISHI_CUDA_COMPWISEASS2( int2, /= )
KOISHI_CUDA_COMPWISEASS2( int2, %= )
KOISHI_CUDA_COMPWISEASS2( int2, &= )
KOISHI_CUDA_COMPWISEASS2( int2, |= )
KOISHI_CUDA_COMPWISEASS2( int2, <<= )
KOISHI_CUDA_COMPWISEASS2( int2, >>= )

KOISHI_CUDA_COMPWISEASS3( int3, += )
KOISHI_CUDA_COMPWISEASS3( int3, -= )
KOISHI_CUDA_COMPWISEASS3( int3, *= )
KOISHI_CUDA_COMPWISEASS3( int3, /= )
KOISHI_CUDA_COMPWISEASS3( int3, %= )
KOISHI_CUDA_COMPWISEASS3( int3, &= )
KOISHI_CUDA_COMPWISEASS3( int3, |= )
KOISHI_CUDA_COMPWISEASS3( int3, <<= )
KOISHI_CUDA_COMPWISEASS3( int3, >>= )

KOISHI_CUDA_COMPWISEASS4( int4, += )
KOISHI_CUDA_COMPWISEASS4( int4, -= )
KOISHI_CUDA_COMPWISEASS4( int4, *= )
KOISHI_CUDA_COMPWISEASS4( int4, /= )
KOISHI_CUDA_COMPWISEASS4( int4, %= )
KOISHI_CUDA_COMPWISEASS4( int4, &= )
KOISHI_CUDA_COMPWISEASS4( int4, |= )
KOISHI_CUDA_COMPWISEASS4( int4, <<= )
KOISHI_CUDA_COMPWISEASS4( int4, >>= )

KOISHI_CUDA_COMPWISE_BINFN1( int1, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN1( int1, funcs, min )

KOISHI_CUDA_COMPWISE_BINFN2( int2, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN2( int2, funcs, min )

KOISHI_CUDA_COMPWISE_BINFN3( int3, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN3( int3, funcs, min )

KOISHI_CUDA_COMPWISE_BINFN4( int4, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN4( int4, funcs, min )

KOISHI_CUDA_COMPWISE1( uint1, +)
KOISHI_CUDA_COMPWISE1( uint1, -)
KOISHI_CUDA_COMPWISE1( uint1, * )
KOISHI_CUDA_COMPWISE1( uint1, / )
KOISHI_CUDA_COMPWISE1( uint1, % )
KOISHI_CUDA_COMPWISE1( uint1, & )
KOISHI_CUDA_COMPWISE1( uint1, | )
KOISHI_CUDA_COMPWISE1( uint1, << )
KOISHI_CUDA_COMPWISE1( uint1, >> )

KOISHI_CUDA_COMPWISE2( uint2, +)
KOISHI_CUDA_COMPWISE2( uint2, -)
KOISHI_CUDA_COMPWISE2( uint2, * )
KOISHI_CUDA_COMPWISE2( uint2, / )
KOISHI_CUDA_COMPWISE2( uint2, % )
KOISHI_CUDA_COMPWISE2( uint2, & )
KOISHI_CUDA_COMPWISE2( uint2, | )
KOISHI_CUDA_COMPWISE2( uint2, << )
KOISHI_CUDA_COMPWISE2( uint2, >> )

KOISHI_CUDA_COMPWISE3( uint3, +)
KOISHI_CUDA_COMPWISE3( uint3, -)
KOISHI_CUDA_COMPWISE3( uint3, * )
KOISHI_CUDA_COMPWISE3( uint3, / )
KOISHI_CUDA_COMPWISE3( uint3, % )
KOISHI_CUDA_COMPWISE3( uint3, & )
KOISHI_CUDA_COMPWISE3( uint3, | )
KOISHI_CUDA_COMPWISE3( uint3, << )
KOISHI_CUDA_COMPWISE3( uint3, >> )

KOISHI_CUDA_COMPWISE4( uint4, +)
KOISHI_CUDA_COMPWISE4( uint4, -)
KOISHI_CUDA_COMPWISE4( uint4, * )
KOISHI_CUDA_COMPWISE4( uint4, / )
KOISHI_CUDA_COMPWISE4( uint4, % )
KOISHI_CUDA_COMPWISE4( uint4, & )
KOISHI_CUDA_COMPWISE4( uint4, | )
KOISHI_CUDA_COMPWISE4( uint4, << )
KOISHI_CUDA_COMPWISE4( uint4, >> )

KOISHI_CUDA_COMPWISEASS1( uint1, += )
KOISHI_CUDA_COMPWISEASS1( uint1, -= )
KOISHI_CUDA_COMPWISEASS1( uint1, *= )
KOISHI_CUDA_COMPWISEASS1( uint1, /= )
KOISHI_CUDA_COMPWISEASS1( uint1, %= )
KOISHI_CUDA_COMPWISEASS1( uint1, &= )
KOISHI_CUDA_COMPWISEASS1( uint1, |= )
KOISHI_CUDA_COMPWISEASS1( uint1, <<= )
KOISHI_CUDA_COMPWISEASS1( uint1, >>= )

KOISHI_CUDA_COMPWISEASS2( uint2, += )
KOISHI_CUDA_COMPWISEASS2( uint2, -= )
KOISHI_CUDA_COMPWISEASS2( uint2, *= )
KOISHI_CUDA_COMPWISEASS2( uint2, /= )
KOISHI_CUDA_COMPWISEASS2( uint2, %= )
KOISHI_CUDA_COMPWISEASS2( uint2, &= )
KOISHI_CUDA_COMPWISEASS2( uint2, |= )
KOISHI_CUDA_COMPWISEASS2( uint2, <<= )
KOISHI_CUDA_COMPWISEASS2( uint2, >>= )

KOISHI_CUDA_COMPWISEASS3( uint3, += )
KOISHI_CUDA_COMPWISEASS3( uint3, -= )
KOISHI_CUDA_COMPWISEASS3( uint3, *= )
KOISHI_CUDA_COMPWISEASS3( uint3, /= )
KOISHI_CUDA_COMPWISEASS3( uint3, %= )
KOISHI_CUDA_COMPWISEASS3( uint3, &= )
KOISHI_CUDA_COMPWISEASS3( uint3, |= )
KOISHI_CUDA_COMPWISEASS3( uint3, <<= )
KOISHI_CUDA_COMPWISEASS3( uint3, >>= )

KOISHI_CUDA_COMPWISEASS4( uint4, += )
KOISHI_CUDA_COMPWISEASS4( uint4, -= )
KOISHI_CUDA_COMPWISEASS4( uint4, *= )
KOISHI_CUDA_COMPWISEASS4( uint4, /= )
KOISHI_CUDA_COMPWISEASS4( uint4, %= )
KOISHI_CUDA_COMPWISEASS4( uint4, &= )
KOISHI_CUDA_COMPWISEASS4( uint4, |= )
KOISHI_CUDA_COMPWISEASS4( uint4, <<= )
KOISHI_CUDA_COMPWISEASS4( uint4, >>= )

KOISHI_CUDA_COMPWISE_BINFN1( uint1, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN1( uint1, funcs, min )

KOISHI_CUDA_COMPWISE_BINFN2( uint2, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN2( uint2, funcs, min )

KOISHI_CUDA_COMPWISE_BINFN3( uint3, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN3( uint3, funcs, min )

KOISHI_CUDA_COMPWISE_BINFN4( uint4, funcs, max )
KOISHI_CUDA_COMPWISE_BINFN4( uint4, funcs, min )

#undef KOISHI_CUDA_COMPWISE1
#undef KOISHI_CUDA_COMPWISE2
#undef KOISHI_CUDA_COMPWISE3
#undef KOISHI_CUDA_COMPWISE4

#undef KOISHI_CUDA_COMPWISEASS1
#undef KOISHI_CUDA_COMPWISEASS2
#undef KOISHI_CUDA_COMPWISEASS3
#undef KOISHI_CUDA_COMPWISEASS4

#undef KOISHI_CUDA_COMPWISE_BINFN1
#undef KOISHI_CUDA_COMPWISE_BINFN2
#undef KOISHI_CUDA_COMPWISE_BINFN3
#undef KOISHI_CUDA_COMPWISE_BINFN4

}  // namespace cuda

}  // namespace koishi