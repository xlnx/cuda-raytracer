#pragma once

#include <core/basic/basic.hpp>
#include <core/misc/sampler.hpp>
#include <core/meta/scene.hpp>
#include <core/meta/profiler.hpp>

namespace koishi
{
namespace core
{
struct Kernel : emittable
{
	Kernel( const Properties &props )
	{
	}

	virtual float3 execute( Ray ray, const Scene &scene,
							Allocator &pool, Sampler &rng, const ProfileSlice &prof ) = 0;
};

}  // namespace core

}  // namespace koishi