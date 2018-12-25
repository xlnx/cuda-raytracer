#pragma once

#include <core/basic/basic.hpp>
#include <core/misc/sampler.hpp>
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
PolyFunction( BruteForce, Host, Device )(
  ( const Ray &r, const Scene &scene, Allocator &pool, Sampler &rng )
	->float3 {
		auto ray = r;
		Varyings varyings;
		float3 L = { 0, 0, 0 }, beta = { 1, 1, 1 };  // brdf can contain 3 components
		constexpr auto maxBounce = 8;

		for ( auto bounce = 0; scene.intersect( ray, varyings, pool ) &&
							   bounce != maxBounce;
			  ++bounce )
		{
			auto &shader = scene.shaders[ varyings.shaderId ];

			// evaluate direct lighting
			L += beta * varyings.emission;
			// L += beta * varyings.global( wi );
			// L += beta * 0.5;
			// emit new light for indirect lighting, according to BSDF
			{
				shader->execute( varyings, rng, pool, sample_wi_f_by_wo );
				beta *= varyings.f * abs( dot( varyings.wi, float3{ 0, 0, 1 } ) );
				ray = varyings.emitRay( varyings.global( varyings.wi ) );
			}
			pool.clear();
		}

		return L;
	} );

}  // namespace core

}  // namespace koishi
