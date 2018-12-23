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
PolyFunction( Custom, Host, Device )(
  ( const Ray &r, const Scene &scene, Allocator &pool, Sampler &rng )
	->float3 {
		float3 L = { 0, 0, 0 };
		auto ray = r;

		if ( auto isect = scene.intersect( r, pool ) )
		{
			auto wo = isect.local( -ray.d );
			auto bxdf = isect.bsdf->sampleBxDF( rng.sample() );

			float3 li;
			uint idx = 0;
			// min( (uint)floor( rng.sample() * scene.lights.size() ),
			// 				( uint )( scene.lights.size() - 1 ) );
			auto wi = scene.lights[ idx ]->sample( scene, isect, rng.sample2(), li, pool );
			auto f = isect.color * bxdf->f( wo, wi ) * abs( dot( wi, float3{ 0, 0, 1 } ) );

			L = li;
			// L = li;
		}
		pool.clear();

		return L;
	} );

}  // namespace core

}  // namespace koishi
