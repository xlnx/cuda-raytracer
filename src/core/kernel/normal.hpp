#pragma once

#include <vec/vec.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/poly.hpp>
#include <core/basic/allocator.hpp>
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
		Input input;

		if ( scene.intersect( r, input, pool ) )
		{
			L = input.n;
		}
		pool.clear();

		return L;
	} );

}  // namespace core

}  // namespace koishi
