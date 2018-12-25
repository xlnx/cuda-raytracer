#pragma once

#include <core/basic/basic.hpp>
#include <core/meta/scene.hpp>
#include "light.hpp"

namespace koishi
{
namespace core
{
struct AreaLight : Light
{
	AreaLight( const float3 &emission, const poly::object<Primitive> &obj ) :
	  obj( obj )
	{
	}

	KOISHI_HOST_DEVICE solid sample( const Scene &scene,
									 const LocalVaryings &res,
									 const float2 &u, float3 &li,
									 Allocator &pool ) const override
	{
		float pdf;
		auto varyings = obj->sample( res.p, u, pdf );
		auto seg = res.emitSeg( varyings.p );
		// li = float3{ 1, 1, 1 } / pdf;
		li = ( scene.intersect( seg, pool ) ? float3{ 0, 0, 0 } : emission ) / pdf;
		// li = varyings.n;
		return res.local( normalize( varyings.p - res.p ) );
	}

private:
	poly::ref<Primitive> obj;
	float3 emission;
};

}  // namespace core

}  // namespace koishi