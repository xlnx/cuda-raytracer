#pragma once

#include "light.hpp"
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
struct AreaLight : Light
{
	AreaLight( const Config &config, const poly::object<Primitive> &obj ) :
	  obj( obj ),
	  emissive( get<float3>( config.props, "emissive", float3{ 1, 1, 1 } ) )
	{
	}

	KOISHI_HOST_DEVICE solid sample( const Scene &scene,
												 const Interreact &res,
												 const float2 &u, float3 &li,
												 Allocator &pool ) const override
	{
	}

private:
	poly::ref<Primitive> obj;
	float3 emissive;
};

}  // namespace core

}  // namespace koishi