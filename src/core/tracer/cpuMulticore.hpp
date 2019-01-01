#pragma once

#include <vector>
#include <thread>
#include <atomic>
#include <iostream>
#include <core/basic/basic.hpp>
#include "tracer.hpp"

namespace koishi
{
namespace core
{
struct CPUMulticoreTracer : Tracer
{
	CPUMulticoreTracer( const Properties &props );

	void execute( util::Image<3> &image, poly::object<Lens> &lens, SamplerGenerator &rng_gen,
				  Scene &scene, uint spp, Profiler &profiler ) override;

private:
	poly::object<Kernel> kernel;
};

}  // namespace core

}  // namespace koishi
