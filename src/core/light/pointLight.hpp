#pragma once

#include <core/light/light.hpp>
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
struct PointLight : Light
{
	PointLight( const float3 &p ) :
	  p( p )
	{
	}

	KOISHI_HOST_DEVICE float3 sample( const Scene &scene,
									  const Interreact &res,
									  const float2 &u, float3 &li,
									  Allocator &pool ) const override
	{
		auto seg = res.emitSeg( p );
		li = scene.intersect( seg, pool ) ? float3{ 0, 0, 0 } : float3{ 1, 1, 1 };
		return res.local( normalize( p - res.p ) );
	}

private:
	float3 p;
};

}  // namespace core

}  // namespace koishi