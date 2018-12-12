#pragma once

#include <vector>
#include <thread>
#include <iostream>
#include <vec/vec.hpp>
#include <util/image.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/poly.hpp>
#include <core/basic/allocator.hpp>
#include <core/misc/lens.hpp>
#include <core/misc/sampler.hpp>
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
template <typename Radiance, typename Alloc = HybridAllocator>
PolyFunction( CPUMultiCoreTracer, Require<Host, Radiance, HybridAllocator> )(
  ( util::Image<3> & image, Lens &lens, SamplerGenerator &rng_gen, Scene &scene, uint spp )
	->void {
		uint w = image.width();
		uint h = image.height();

		static constexpr uint MaxThreads =
#ifndef DEBUG
		  -1u;
#else
		  1u;
#endif

		auto ncores = std::thread::hardware_concurrency();
		if ( MaxThreads < ncores ) ncores = MaxThreads;
		std::cout << "using " << ncores << " threads:" << std::endl;
		std::vector<std::thread> ts;
		KINFO( tracer, "Tracing start" );
		util::tick();
		auto rng = rng_gen.create();
		auto tracer_thread = [ncores, spp, h, w, &scene, &image, &lens, &rng]( uint id ) {
			static constexpr uint b = 1, kb = 1024 * b, mb = 1024 * kb;
			static constexpr uint block_size = 480 * b;

			char block[ block_size ];

			Alloc pool( block, block_size );

			for ( uint j = id; j < h; j += ncores )
			{
				for ( uint i = 0; i != w; ++i )
				{
					float3 rad = { 0, 0, 0 };
					for ( uint k = 0; k != spp; ++k )
					{
						rad += Self::template call<Radiance>( lens.sample( i, j, k ), scene, pool, rng );
						pool.clear();
					}
					image.at( i, j ) = rad / spp;
				}
			}
		};
		for ( auto id = 0u; id != ncores - 1; ++id )
		{
			ts.emplace_back( tracer_thread, id );
		}
		tracer_thread( ncores - 1 );

		for ( auto &th : ts )
		{
			th.join();
		}
		KINFO( tracer, "Tracer joint in", util::tick(), "seconds" );
	} );

}  // namespace core

}  // namespace koishi
