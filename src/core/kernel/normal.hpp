#pragma once

#include <core/basic/basic.hpp>
#include "kernel.hpp"

namespace koishi
{
namespace core
{
struct NormalKernel : Kernel
{
	NormalKernel( const Properties &props ) :
	  Kernel( props )
	{
	}

	float3 execute( Ray r, const Scene &scene, Allocator &pool,
					Sampler &rng, ProfileSlice *prof ) override
	{
		float3 L = { 0, 0, 0 };
		Varyings varyings;

		if ( scene.intersect( r, varyings, pool ) )
		{
			L = varyings.n;
		}
		pool.clear();

		return L;
	}
};

}  // namespace core

}  // namespace koishi
