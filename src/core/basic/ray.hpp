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
	double t = INFINITY;
	double2 uv;

	KOISHI_HOST_DEVICE operator bool() const
	{
		return t != INFINITY;
	}
};

struct Ray
{
	double3 o, d;

	KOISHI_HOST_DEVICE bool intersect_bbox( const double3 &vmin, const double3 &vmax ) const
	{
		auto invv = 1.0 / d;
		auto a = ( vmin - o ) * invv, b = ( vmax - o ) * invv;
		auto tmin = min( a, b ), tmax = max( a, b );
		auto t0 = max( tmin.x, max( tmin.y, tmin.z ) ), t1 = min( tmax.x, min( tmax.y, tmax.z ) );
		return t0 <= t1 && t1 >= 0;
	}
	KOISHI_HOST_DEVICE bool intersect_triangle( const double3 &v0, const double3 &v1, const double3 &v2, Hit &hit ) const
	{
		auto e1 = v1 - v0, e2 = v2 - v0;
		auto P = cross( d, e2 );
		auto det = dot( e1, P );
		double3 T;
		if ( det > 0 )
		{
			T = o - v0;
		}
		else
		{
			T = v0 - o;
			det = -det;
		}
		if ( det < .0001f )
		{
			return false;
		}
		hit.uv.x = dot( T, P );
		if ( hit.uv.x < 0.f || hit.uv.x > det )
		{
			return false;
		}
		auto Q = cross( T, e1 );
		hit.uv.y = dot( d, Q );
		if ( hit.uv.y < 0.f || hit.uv.x + hit.uv.y > det )
		{
			return false;
		}
		hit.t = dot( e2, Q );
		double invdet = 1.f / det;
		hit.t *= invdet;
		hit.uv *= invdet;

		return hit.t > 0.;
	}
};

KOISHI_HOST_DEVICE inline double3 interplot( const double3 &v0, const double3 &v1,
											 const double3 &v2, const double2 &uv )
{
	return v0 * ( 1 - uv.x - uv.y ) + v1 * uv.x + v2 * uv.y;
}

}  // namespace core

}  // namespace koishi
