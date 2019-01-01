#pragma once

#include <core/kernel/kernel.hpp>
#include <core/misc/lens.hpp>
#include <core/misc/sampler.hpp>
#include <util/image.hpp>

namespace koishi
{
namespace core
{
struct Tracer : emittable
{
	Tracer( const Properties &props )
	{
	}

	virtual void execute( util::Image<3> &image, poly::object<Lens> &lens, SamplerGenerator &rng_gen,
						  Scene &scene, uint spp, Profiler &profiler ) = 0;
};

}  // namespace core

}  // namespace koishi