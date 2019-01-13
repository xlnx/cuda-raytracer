#pragma once

#include <core/basic/basic.hpp>
#include "kernel.hpp"

namespace koishi
{
namespace core
{
struct VisibilityKernel : Kernel
{
	VisibilityKernel( const Properties &props ) :
	  Kernel( props ) {}

	KOISHI_HOST_DEVICE float3 execute( Ray ray, const Scene &scene,
									   Allocator &pool, Sampler &rng,
									   ProfileSlice *prof ) override
	{
		float3 L = { 0, 0, 0 };

		Varyings varyings;

		if ( scene.intersect( ray, varyings, pool ) )
		{
			auto &shader = scene.shaders[ varyings.shaderId ];

			float3 li;
			uint idx = 0;
			varyings.wi = scene.lights[ idx ]->sample( scene, varyings,
													   rng.sample2(), li, pool );

			L = li == float3{ 0, 0, 0 } ? float3{ 0, 0, 0 } : float3{ 1, 1, 1 };
		}
		pool.clear();

		return L;
	}
};

}  // namespace core

}  // namespace koishi
