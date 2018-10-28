#pragma once

namespace koishi
{
namespace vec
{
#if !defined( KOISHI_USE_CUDA )

#define KOISHI_DEF_VEC( type )        \
	struct type##1 { type x; };       \
	struct type##2 { type x, y; };    \
	struct type##3 { type x, y, z; }; \
	struct type##4 { type x, y, z, w; }

using uint = unsigned int;
KOISHI_DEF_VEC( float );
KOISHI_DEF_VEC( double );
KOISHI_DEF_VEC( int );
KOISHI_DEF_VEC( uint );

#undef KOISHI_DEF_VEC

#endif

#define KOISHI_VEC_FLOAT float1, float2, float3, float4
#define KOISHI_VEC_DOUBLE double1, double2, double3, double4
#define KOISHI_VEC_INT int1, int2, int3, int4
#define KOISHI_VEC_UINT uint1, uint2, uint3, uint4
#define KOISHI_VEC KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT

}  // namespace vec

}  // namespace koishi

using namespace koishi::vec;