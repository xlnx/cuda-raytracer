#pragma once

#include <vec/vec.hpp>
#include <vec/vmath.hpp>
#include "mesh.hpp"

namespace koishi
{
namespace util
{
struct Ray
{
	float3 o, v;

	KOISHI_HOST_DEVICE bool intersect_bbox( const float3 &vmin, const float3 &vmax ) const
	{
		auto a = ( vmin - o ) / v, b = ( vmax - o ) / v;
		auto tmin = min( a, b ), tmax = max( a, b );
		auto t0 = max( tmin.x, max( tmin.y, tmin.z ) ), t1 = min( tmax.x, min( tmax.y, tmax.z ) );
		return t0 <= t1 && t1 >= 0;
	}
	KOISHI_HOST_DEVICE bool intersect_triangle( const float3 &v0, const float3 &v1, const float3 &v2,
												float &t, float2 &uv ) const
	{
		auto e1 = v1 - v0, e2 = v2 - v0;
		auto P = cross( v, e2 );
		auto det = dot( e1, P );
		float3 T;
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
		uv.x = dot( T, P );
		if ( uv.x < 0.f || uv.x > det )
		{
			return false;
		}
		auto Q = cross( T, e1 );
		uv.y = dot( v, Q );
		if ( uv.y < 0.f || uv.x + uv.y > det )
		{
			return false;
		}
		t = dot( e2, Q );
		float invdet = 1.f / det;
		t *= invdet;
		uv *= invdet;

		return true;
	}

	KOISHI_HOST_DEVICE bool intersect( const SubMesh &mesh, uint root, float &t, float2 &uv ) const
	{
		uint i = root;
		while ( !mesh.bvh[ i ].isleaf )
		{
			auto left = intersect_bbox( mesh.bvh[ i << 1 ].vmin, mesh.bvh[ i << 1 ].vmax );
			auto right = intersect_bbox( mesh.bvh[ ( i << 1 ) + 1 ].vmin, mesh.bvh[ ( i << 1 ) + 1 ].vmax );
			if ( !left && !right ) return false;
			if ( left && right )
			{
				float t1;
				float2 uv1;
				auto b0 = intersect( mesh, root << 1, t, uv );
				auto b1 = intersect( mesh, ( root << 1 ) | 1, t1, uv1 );
				if ( !b0 && !b1 )
				{
					return false;
				}
				if ( !b0 || b1 && t1 < t )
				{
					t = t1;
					uv = uv1;
				}
				return true;
			}
			i <<= 1;
			if ( right ) i |= 1;
		}
		// return true;
		for ( uint j = mesh.bvh[ i ].begin; j < mesh.bvh[ i ].end; j += 3 )
		{
			float t;
			float2 uv;
			if ( intersect_triangle( mesh.vertices[ mesh.indices[ j ] ],
									 mesh.vertices[ mesh.indices[ j + 1 ] ],
									 mesh.vertices[ mesh.indices[ j + 2 ] ], t, uv ) &&
				 t > 0.f )
			{
				return true;
			}
		}
		return false;
	}
};

}  // namespace util

using namespace util;

}  // namespace koishi
