#pragma once

#include "light.hpp"
#include <ext/material/luz.hpp>
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
struct AreaLight : Light
{
	AreaLight( const poly::object<Material> &mat, const poly::object<Primitive> &obj ) :
	  obj( obj ),
	  mat( mat )
	{
	}

	KOISHI_HOST_DEVICE solid sample( const Scene &scene,
									 const LocalInput &res,
									 const float2 &u, float3 &li,
									 Allocator &pool ) const override
	{
		float pdf;
		auto input = obj->sample( res.p, u, pdf );
		auto seg = res.emitSeg( input.p );
		// li = float3{ 1, 1, 1 } / pdf;
		auto &luz = static_cast<const ext::LuzMaterial &>( *mat );
		li = ( scene.intersect( seg, pool ) ? float3{ 0, 0, 0 } : luz.emissive ) / pdf;
		// li = input.n;
		return res.local( normalize( input.p - res.p ) );
	}

private:
	poly::ref<Primitive> obj;
	poly::ref<Material> mat;
};

}  // namespace core

}  // namespace koishi