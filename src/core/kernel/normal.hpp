#pragma once

#include <core/basic/basic.hpp>
#include <core/misc/sampler.hpp>
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
PolyFunction( Normal, Host, Device )(
  ( const Ray &r, const Scene &scene, Allocator &pool, Sampler &rng )
	->float3 {
		float3 L = { 0, 0, 0 };
		Varyings varyings;

		if ( scene.intersect( r, varyings, pool ) )
		{
			L = varyings.n;
		}
		pool.clear();

		return L;
	} );

}  // namespace core

}  // namespace koishi
