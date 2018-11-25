#pragma once

#include <type_traits>
#include <vec/vec.hpp>
#include <vec/vmath.hpp>

namespace koishi
{
namespace core
{
struct Hit
{
	uint id;
	float t = INFINITY;
	float2 uv;

	KOISHI_HOST_DEVICE operator bool() const
	{
		return t != INFINITY;
	}
};

struct Ray
{
	float3 o, d;

	KOISHI_HOST_DEVICE bool intersect_bbox( const float3 &vmin, const float3 &vmax ) const
	{
		auto invv = 1.0 / d;
		auto a = ( vmin - o ) * invv, b = ( vmax - o ) * invv;
		auto tmin = min( a, b ), tmax = max( a, b );
		auto t0 = max( tmin.x, max( tmin.y, tmin.z ) ), t1 = min( tmax.x, min( tmax.y, tmax.z ) );
		return t0 <= t1 && t1 >= 0;
	}
	KOISHI_HOST_DEVICE bool intersect_triangle( const float3 &v0, const float3 &d1, const float3 &d2, Hit &hit ) const
	{
		auto e1 = d1, e2 = d2, v = v0;
		auto P = cross( d, e2 );
		auto det = dot( e1, P );
		float3 T;
		if ( det > 0 )
		{
			T = o - v;
		}
		else
		{
			T = v - o;
			det = -det;
		}
		//	if ( det < .0001f )
		//	{
		//		return false;
		//	}
		hit.uv.x = dot( T, P );
		auto Q = cross( T, e1 );
		hit.uv.y = dot( d, Q );
		hit.t = dot( e2, Q );
		float invdet = 1.f / det;
		hit.t *= invdet;
		hit.uv *= invdet;

		return hit.t > 0. && hit.uv.x >= 0.f && hit.uv.y >= 0.f && hit.uv.x + hit.uv.y <= 1.f;
	}
};

KOISHI_HOST_DEVICE inline float3 interplot( const float3 &v0, const float3 &v1,
											const float3 &v2, const float2 &uv )
{
	return v0 * ( 1 - uv.x - uv.y ) + v1 * uv.x + v2 * uv.y;
}

}  // namespace core

}  // namespace koishi
