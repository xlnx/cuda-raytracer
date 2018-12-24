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
PolyFunction( Radiance, Host, Device )(
  ( const Ray &r, const Scene &scene, Allocator &pool, Sampler &rng )
	->float3 {
		auto ray = r;
		Input input;
		input.sampler = &rng;
		float3 L = { 0, 0, 0 }, beta = { 1, 1, 1 };  // brdf can contain 3 components
		constexpr auto maxBounce =					 //1;
		  1024;
		// 1024;

		for ( auto bounce = 0; scene.intersect( ray, input, pool ) &&
							   bounce != maxBounce;
			  ++bounce )
		{
			// evaluate direct lighting
			if ( scene.lights.size() )
			{
				float3 li;
				uint idx = min( (uint)floor( rng.sample() * scene.lights.size() ),
								( uint )( scene.lights.size() - 1 ) );
				float lpdf = 1.f / scene.lights.size();
				auto wi = scene.lights[ idx ]->sample( scene, input, rng.sample2(), li, pool );
				auto f = input.color * input.bxdf->f( input.wo, wi );
				L += beta * f * li * abs( dot( wi, float3{ 0, 0, 1 } ) ) / lpdf;
			}
			L += beta * input.emissive;
			// emit new light for indirect lighting, according to BSDF
			{
				float3 f;
				auto wi = input.bxdf->sample( input.wo, rng.sample2(), f );
				beta *= f * input.color * abs( dot( wi, float3{ 0, 0, 1 } ) );
				ray = input.emitRay( input.global( wi ) );
			}
			pool.clear();

			auto rr = max( beta.x, max( beta.y, beta.z ) );
			if ( rr < 1. && bounce > 3 )
			{
				auto q = max( .05f, 1 - rr );
				if ( rng.sample() < q ) break;
				beta /= 1 - q;
			}
		}

		return L;
	} );

}  // namespace core

}  // namespace koishi
