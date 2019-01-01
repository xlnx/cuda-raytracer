#pragma once

#include <iostream>
#include "trait.hpp"

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

namespace koishi
{
namespace vec
{
using uint = unsigned int;

#if !defined( KOISHI_USE_CUDA )

#define KOISHI_DEF_VEC( type )        \
	struct type##1 { type x; };       \
	struct type##2 { type x, y; };    \
	struct type##3 { type x, y, z; }; \
	struct type##4 { type x, y, z, w; }

KOISHI_DEF_VEC( float );
KOISHI_DEF_VEC( double );
KOISHI_DEF_VEC( int );
KOISHI_DEF_VEC( uint );

#undef KOISHI_DEF_VEC

#endif

#define KOISHI_DEF_VEC( T )                                                            \
	struct normalized_##T##1 : T##1                                                    \
	{                                                                                  \
		KOISHI_HOST_DEVICE normalized_##T##1() = default;                              \
		KOISHI_HOST_DEVICE explicit normalized_##T##1( const T##1 & v ) : T##1( v ) {} \
	};                                                                                 \
	struct normalized_##T##2 : T##2                                                    \
	{                                                                                  \
		KOISHI_HOST_DEVICE normalized_##T##2() = default;                              \
		KOISHI_HOST_DEVICE explicit normalized_##T##2( const T##2 & v ) : T##2( v ) {} \
	};                                                                                 \
	struct normalized_##T##3 : T##3                                                    \
	{                                                                                  \
		KOISHI_HOST_DEVICE normalized_##T##3() = default;                              \
		KOISHI_HOST_DEVICE explicit normalized_##T##3( const T##3 & v ) : T##3( v ) {} \
	};                                                                                 \
	struct normalized_##T##4 : T##4                                                    \
	{                                                                                  \
		KOISHI_HOST_DEVICE normalized_##T##4() = default;                              \
		KOISHI_HOST_DEVICE explicit normalized_##T##4( const T##4 & v ) : T##4( v ) {} \
	}

KOISHI_DEF_VEC( float );
KOISHI_DEF_VEC( double );
KOISHI_DEF_VEC( int );
KOISHI_DEF_VEC( uint );

#undef KOISHI_DEF_VEC

template <typename T>
struct normalized_type;

template <typename T>
struct unnormalized_type;

#define KOISHI_DEF_VEC( T )                     \
	template <uint N>                           \
	struct vec##T;                              \
	template <>                                 \
	struct vec##T<1>                            \
	{                                           \
		using type = T##1;                      \
	};                                          \
	template <>                                 \
	struct normalized_type<T##1>                \
	{                                           \
		using type = normalized_##T##1;         \
	};                                          \
	template <>                                 \
	struct normalized_type<normalized_##T##1>   \
	{                                           \
		using type = normalized_##T##1;         \
	};                                          \
	template <>                                 \
	struct unnormalized_type<T##1>              \
	{                                           \
		using type = T##1;                      \
	};                                          \
	template <>                                 \
	struct unnormalized_type<normalized_##T##1> \
	{                                           \
		using type = T##1;                      \
	};                                          \
	template <>                                 \
	struct vec##T<2>                            \
	{                                           \
		using type = T##2;                      \
	};                                          \
	template <>                                 \
	struct normalized_type<T##2>                \
	{                                           \
		using type = normalized_##T##2;         \
	};                                          \
	template <>                                 \
	struct normalized_type<normalized_##T##2>   \
	{                                           \
		using type = normalized_##T##2;         \
	};                                          \
	template <>                                 \
	struct unnormalized_type<T##2>              \
	{                                           \
		using type = T##2;                      \
	};                                          \
	template <>                                 \
	struct unnormalized_type<normalized_##T##2> \
	{                                           \
		using type = T##2;                      \
	};                                          \
	template <>                                 \
	struct vec##T<3>                            \
	{                                           \
		using type = T##3;                      \
	};                                          \
	template <>                                 \
	struct normalized_type<T##3>                \
	{                                           \
		using type = normalized_##T##3;         \
	};                                          \
	template <>                                 \
	struct normalized_type<normalized_##T##3>   \
	{                                           \
		using type = normalized_##T##3;         \
	};                                          \
	template <>                                 \
	struct unnormalized_type<T##3>              \
	{                                           \
		using type = T##3;                      \
	};                                          \
	template <>                                 \
	struct unnormalized_type<normalized_##T##3> \
	{                                           \
		using type = T##3;                      \
	};                                          \
	template <>                                 \
	struct vec##T<4>                            \
	{                                           \
		using type = T##4;                      \
	};                                          \
	template <>                                 \
	struct normalized_type<T##4>                \
	{                                           \
		using type = normalized_##T##4;         \
	};                                          \
	template <>                                 \
	struct normalized_type<normalized_##T##4>   \
	{                                           \
		using type = normalized_##T##4;         \
	};                                          \
	template <>                                 \
	struct unnormalized_type<T##4>              \
	{                                           \
		using type = T##4;                      \
	};                                          \
	template <>                                 \
	struct unnormalized_type<normalized_##T##4> \
	{                                           \
		using type = T##4;                      \
	}

KOISHI_DEF_VEC( float );
KOISHI_DEF_VEC( double );
KOISHI_DEF_VEC( int );
KOISHI_DEF_VEC( uint );

#define KOISHI_VEC_FLOAT float1, float2, float3, float4, normalized_float1, normalized_float2, normalized_float3, normalized_float4
#define KOISHI_VEC_DOUBLE double1, double2, double3, double4, normalized_double1, normalized_double2, normalized_double3, normalized_double4
#define KOISHI_VEC_INT int1, int2, int3, int4
#define KOISHI_VEC_UINT uint1, uint2, uint3, uint4
#define KOISHI_VEC KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT

}  // namespace vec

}  // namespace koishi

using namespace koishi::vec;

#if !defined( KOISHI_USE_CUDA )

namespace koishi
{
namespace vec
{
#endif

#define KOISHI_VEC_PRINT( ... )                                                                                                                                                          \
	template <typename T, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec1<T>::value>::type>                            \
	std::ostream &operator<<( std::ostream &os, const T &t )                                                                                                                             \
	{                                                                                                                                                                                    \
		os << "[ " << t.x << " ]";                                                                                                                                                       \
		return os;                                                                                                                                                                       \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec2<T>::value>::type>                   \
	std::ostream &operator<<( std::ostream &os, const T &t )                                                                                                                             \
	{                                                                                                                                                                                    \
		os << "[ " << t.x << ", " << t.y << " ]";                                                                                                                                        \
		return os;                                                                                                                                                                       \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec3<T>::value>::type>          \
	std::ostream &operator<<( std::ostream &os, const T &t )                                                                                                                             \
	{                                                                                                                                                                                    \
		os << "[ " << t.x << ", " << t.y << ", " << t.z << " ]";                                                                                                                         \
		return os;                                                                                                                                                                       \
	}                                                                                                                                                                                    \
	template <typename T, int = 0, int = 0, int = 0, int = 0, typename = typename std::enable_if<koishi::trait::is_in<T, __VA_ARGS__>::value && koishi::trait::is_vec4<T>::value>::type> \
	std::ostream &operator<<( std::ostream &os, const T &t )                                                                                                                             \
	{                                                                                                                                                                                    \
		os << "[ " << t.x << ", " << t.y << ", " << t.z << ", " << t.w << " ]";                                                                                                          \
		return os;                                                                                                                                                                       \
	}

KOISHI_VEC_PRINT( KOISHI_VEC_FLOAT, KOISHI_VEC_DOUBLE, KOISHI_VEC_INT, KOISHI_VEC_UINT )

#undef KOISHI_VEC_PRINT

#if !defined( KOISHI_USE_CUDA )
}
}
#endif
