#pragma once

#include "light.hpp"
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
struct PointLight : Light
{
	PointLight( const Config &config ) :
	  p( get<float3>( config.props, "position" ) ),
	  emissive( get<float3>( config.props, "color", float3{ 1, 1, 1 } ) *
				get<float>( config.props, "strength", 2.f ) )
	{
	}

	KOISHI_HOST_DEVICE solid sample( const Scene &scene,
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