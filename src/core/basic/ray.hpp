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
	float3 data;

	KOISHI_HOST_DEVICE operator bool() const
	{
		return t != INFINITY;
	}
};

struct Ray
{
	float3 o;
	normalized_float3 d;

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
		auto T = o - v;
		auto P = cross( d, e2 );
		auto Q = cross( T, e1 );
		auto det = dot( e1, P );
		hit.uv.x = dot( T, P );
		hit.uv.y = dot( d, Q );
		hit.t = dot( e2, Q );
		float invdet = 1.f / det;
		hit.t *= invdet;
		hit.uv *= invdet;

		return hit.t > 0. && hit.uv.x >= 0.f && hit.uv.y >= 0.f && hit.uv.x + hit.uv.y <= 1.f;
	}
};

struct Seg : Ray
{
	float t;

	KOISHI_HOST_DEVICE bool intersect_bbox( const float3 &vmin, const float3 &vmax ) const
	{
		auto c = ( vmin + vmax ) * .5f;
		auto l = vmax - c;
		auto p0 = o - c, p1 = p0 + d * t;
		auto m = ( p0 + p1 ) * .5f;
		auto w = m - p0;
		auto W = abs( w );
		auto x = abs( m ) - W - l;
		if ( x.x > 0 || x.y > 0 || x.z > 0 )
		{
			return false;
		}
		else if ( abs( -w.z * m.y + w.y * m.z ) > l.y * W.z + l.z * W.y ||
				  abs( -w.z * m.x + w.x * m.z ) > l.x * W.z + l.z * W.x ||
				  abs( -w.y * m.x + w.x * m.y ) > l.x * W.y + l.y * W.x )
		{
			return false;
		}
		return true;
	}

	KOISHI_HOST_DEVICE bool intersect_triangle( const float3 &v0, const float3 &d1, const float3 &d2 ) const
	{
		Hit hit;
		auto n = cross( d1, d2 );
		if ( dot( o - v0, n ) * dot( o + d * t - v0, n ) > 0 )
		{
			return false;
		}
		return Ray::intersect_triangle( v0, d1, d2, hit );
	}
};

KOISHI_HOST_DEVICE inline float3
  interplot( const float3 &v0, const float3 &v1,
			 const float3 &v2, const float2 &uv )
{
	return v0 * ( 1 - uv.x - uv.y ) + v1 * uv.x + v2 * uv.y;
}

}  // namespace core

}  // namespace koishi
