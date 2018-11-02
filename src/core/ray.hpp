#pragma once

#include <type_traits>
#include <vec/vec.hpp>
#include <vec/vmath.hpp>
#include "mesh.hpp"

namespace koishi
{
namespace core
{
struct Hit
{
	uint id;
	double t = INFINITY;
	float2 uv;

	operator bool() const
	{
		return t != INFINITY;
	}
};

struct Ray
{
	double3 o, v;

	KOISHI_HOST_DEVICE bool intersect_bbox( const double3 &vmin, const double3 &vmax ) const
	{
		auto invv = 1.0 / v;
		auto a = ( vmin - o ) * invv, b = ( vmax - o ) * invv;
		auto tmin = min( a, b ), tmax = max( a, b );
		auto t0 = max( tmin.x, max( tmin.y, tmin.z ) ), t1 = min( tmax.x, min( tmax.y, tmax.z ) );
		return t0 <= t1 && t1 >= 0;
	}
	KOISHI_HOST_DEVICE bool intersect_triangle( const double3 &v0, const double3 &v1, const double3 &v2, Hit &hit ) const
	{
		auto e1 = v1 - v0, e2 = v2 - v0;
		auto P = cross( v, e2 );
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
		hit.uv.y = dot( v, Q );
		if ( hit.uv.y < 0.f || hit.uv.x + hit.uv.y > det )
		{
			return false;
		}
		hit.t = dot( e2, Q );
		double invdet = 1.f / det;
		hit.t *= invdet;
		hit.uv *= invdet;

		constexpr double eps = 0.001f;
		return hit.t > eps;
	}

	template <typename Mesh, typename = typename std::enable_if<
							   std::is_same<Mesh, core::SubMesh>::value
#if defined( KOISHI_USE_CUDA )
							   || std::is_same<Mesh, core::dev::SubMesh>::value
#endif
							   >::type>
	KOISHI_HOST_DEVICE bool intersect( const Mesh &mesh, uint root, Hit &hit ) const
	{
		uint i = root;
		while ( !mesh.bvh[ i ].isleaf )
		{
			auto left = intersect_bbox( mesh.bvh[ i << 1 ].vmin, mesh.bvh[ i << 1 ].vmax );
			auto right = intersect_bbox( mesh.bvh[ ( i << 1 ) + 1 ].vmin, mesh.bvh[ ( i << 1 ) + 1 ].vmax );
			if ( !left && !right ) return false;
			if ( left && right )
			{
				Hit hit1;
				auto b0 = intersect( mesh, root << 1, hit );
				auto b1 = intersect( mesh, ( root << 1 ) | 1, hit1 );
				if ( !b0 && !b1 )
				{
					return false;
				}
				if ( !b0 || b1 && hit1.t < hit.t )
				{
					hit = hit1;
				}
				return true;
			}
			i <<= 1;
			if ( right ) i |= 1;
		}
		// return true;
		hit.t = INFINITY;
		for ( uint j = mesh.bvh[ i ].begin; j < mesh.bvh[ i ].end; j += 3 )
		{
			core::Hit hit1;
			if ( intersect_triangle( mesh.vertices[ mesh.indices[ j ] ],
									 mesh.vertices[ mesh.indices[ j + 1 ] ],
									 mesh.vertices[ mesh.indices[ j + 2 ] ], hit1 ) &&
				 hit1.t < hit.t )
			{
				hit = hit1;
				hit.id = j;
			}
		}
		return hit.t != INFINITY;
	}
	KOISHI_HOST_DEVICE Ray reflect( double t, const double3 &N ) const
	{
		return Ray{ o + v * t, normalize( vm::reflect( v, N ) ) };
	}
};

KOISHI_HOST_DEVICE double3 interplot( const double3 &v0, const double3 &v1,
									  const double3 &v2, const float2 &uv )
{
	return v0 * ( 1 - uv.x - uv.y ) + v1 * uv.x + v2 * uv.y;
}

}  // namespace core

}  // namespace koishi
