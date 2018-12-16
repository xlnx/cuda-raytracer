#pragma once

#include <core/light/light.hpp>
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
struct PointLight : Light
{
	PointLight( const Config &config ) :
	  p( get<float3>( config.props, "position" ) ),
	  emissive( get<float3>( config.props, "emissive", float3{ 1, 1, 1 } ) )
	{
	}

	KOISHI_HOST_DEVICE normalized_float3 sample( const Scene &scene,
												 const Interreact &res,
												 const float2 &u, float3 &li,
												 Allocator &pool ) const override
	{
		auto seg = res.emitSeg( p );
		li = scene.intersect( seg, pool ) ? float3{ 0, 0, 0 } : emissive;
		return res.local( normalize( p - res.p ) );
	}

private:
	float3 p;
	float3 emissive;
};

}  // namespace core

}  // namespace koishi