#pragma once

#include <vec/vec.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/poly.hpp>
#include <core/basic/allocator.hpp>
#include <core/misc/sampler.hpp>
#include <core/meta/mesh.hpp>
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
PolyFunction( BruteForce, Host, Device )(
  ( const Ray &r, const Scene &scene, Allocator &pool, Sampler &rng )
	->float3 {
		auto ray = r;
		Input input;
		input.sampler = &rng;
		float3 L = { 0, 0, 0 }, beta = { 1, 1, 1 };  // brdf can contain 3 components
		constexpr auto maxBounce = 8;

		for ( auto bounce = 0; scene.intersect( ray, input, pool ) &&
							   bounce != maxBounce;
			  ++bounce )
		{
			// evaluate direct lighting
			L += beta * input.emissive;
			// L += beta * input.global( wi );
			// L += beta * 0.5;
			// emit new light for indirect lighting, according to BSDF
			{
				float3 f;
				auto wi = input.bxdf->sample( input.wo, rng.sample2(), f );
				beta *= f * input.color * abs( dot( wi, float3{ 0, 0, 1 } ) );
				ray = input.emitRay( input.global( wi ) );
			}
			pool.clear();
		}

		return L;
	} );

}  // namespace core

}  // namespace koishi
