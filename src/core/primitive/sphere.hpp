#pragma once

#include "primitive.hpp"

namespace koishi
{
namespace core
{
struct Sphere : Primitive
{
	Sphere( float3 o, float r, uint shaderId ) :
	  o( o ),
	  r2( r * r )
	{
		this->shaderId = shaderId;
	}

	KOISHI_HOST_DEVICE normalized_float3 normal( const Hit &hit ) const override
	{
		return normalized_float3( hit.data );
	}
	KOISHI_HOST_DEVICE LocalVaryings sample( const float2 &u, float &pdf ) const override
	{
		LocalVaryings varyings;
		auto w = H::fromEular( u.x * PI, u.y * PI * 2 );
		varyings.p = o + sqrt( r2 ) * w;
		varyings.n = w;
		pdf = 1. / ( 4 * PI * r2 );
		return varyings;
	}
	KOISHI_HOST_DEVICE bool intersect( const Ray &ray, Hit &hit, Allocator &pool ) const override
	{
		auto oc = o - ray.o;
		// KLOG( "oc=", oc );
		auto poc = dot( ray.d, oc );
		// KLOG( "poc=", poc );
		auto oc2 = dot( oc, oc );
		auto d2 = oc2 - poc * poc;
		auto det = r2 - d2;
		// KLOG( "det=", det );
		if ( det < 0 ) return false;
		constexpr auto eps = 1e-3;
		// if ( det < eps )
		// else
		{
			det = sqrt( det );
			if ( ( hit.t = poc - det ) < 0 )
			{
				hit.t = poc + det;
			}
		}
		hit.data = normalize( ray.o + hit.t * ray.d - o );
		return hit.t >= 0;
	}

private:
	float3 o;
	float r2;
};

}  // namespace core

}  // namespace koishi