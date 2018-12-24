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

		Input input;
		input.sampler = &rng;

		if ( scene.intersect( r, input, pool ) )
		{
			float3 li;
			uint idx = 0;
			// min( (uint)floor( rng.sample() * scene.lights.size() ),
			// 				( uint )( scene.lights.size() - 1 ) );
			auto wi = scene.lights[ idx ]->sample( scene, input, rng.sample2(), li, pool );
			auto f0 = input.bxdf->f( input.wo, wi );
			auto ndl = abs( dot( wi, float3{ 0, 0, 1 } ) );
			auto f = input.color * f0 * ndl;

			// L = float3{f0.x, f0.y * ndl, ndl};
			L = f;
			// L = float3{ f.x, li.y * f.y, li.z };
			//   normalize( input.wo + wi );
			// input.bxdf->f( input.wo, wi );
			// float3{ 1, 1, 1 } * abs( dot( wi, float3{ 0, 0, 1 } ) );
			// li;
			// L = li;
		}
		pool.clear();

		return L;
	} );

}  // namespace core

}  // namespace koishi
