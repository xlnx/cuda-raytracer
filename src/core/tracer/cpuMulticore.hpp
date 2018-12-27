#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <util/image.hpp>
#include <core/basic/basic.hpp>
#include <core/misc/lens.hpp>
#include <core/misc/sampler.hpp>
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
template <typename Radiance, typename Alloc = HybridAllocator>
PolyFunction( CPUMultiCoreTracer, Require<Host, Radiance, HybridAllocator> )(
  ( util::Image<3> & image, poly::object<Lens> &lens, SamplerGenerator &rng_gen, Scene &scene, uint spp )
	->void {
		uint w = image.width();
		uint h = image.height();

		static constexpr uint MaxThreads =
#ifndef DEBUG
		  -1u;
#else
		  1u;
#endif

		auto nthreads = std::thread::hardware_concurrency() - 1;
		if ( MaxThreads < nthreads ) nthreads = MaxThreads;
		KLOG( "Using", nthreads, "threads:" );
		std::vector<std::thread> ts;
		KINFO( tracer, "Tracing start" );
		util::tick();
		auto rng = rng_gen.create();
		std::vector<std::pair<int, int>> status( nthreads );
		std::atomic<int> g_nthreads( nthreads );
		auto tracer_thread = [nthreads, spp, h, w, &scene,
							  &image, &lens, &rng, &status, &g_nthreads]( uint id ) {
			static constexpr uint b = 1, kb = 1024 * b, mb = 1024 * kb;
			static constexpr uint block_size = 480 * b;

			char block[ block_size ];

			Alloc pool( block, block_size );

			{
				static std::mutex m;
				std::lock_guard<std::mutex> _( m );
				status[ id ].second = 0;
				status[ id ].first = 0;
				for ( uint j = id; j < h; j += nthreads )
				{
					status[ id ].second += w;
				}
				g_nthreads--;
			}

			for ( uint j = id; j < h; j += nthreads )
			{
				for ( uint i = 0; i != w; ++i )
				{
					float3 rad = { 0, 0, 0 };
					for ( uint k = 0; k != spp; ++k )
					{
						rad += Self::template call<Radiance>( lens->sample( i, j, k ), scene, pool, rng );
					}
					image.at( i, j ) = rad / spp;
					status[ id ].first++;
				}
			}
		};
		for ( auto id = 0u; id != nthreads; ++id )
		{
			ts.emplace_back( tracer_thread, id );
		}

		static constexpr int nsteps = 64;
		uint total = 0, step = 0;
		while ( g_nthreads > 0 )
			;
		for ( auto id = 0u; id != nthreads; ++id )
		{
			total += status[ id ].second;
			KLOG( "- Thread", id, "issuing", status[ id ].second, "samples" );
		}
		KLOG( total, "samples in total" );
		std::cout << "[ ";
		for ( int i = 0; i != nsteps; ++i )
		{
			std::cout << "-";
		}
		std::cout << " ]" << std::endl
				  << "[ ";
		while ( true )
		{
			bool finished = true;
			uint rendered = 0;
			for ( auto i = 0u; i != nthreads; ++i )
			{
				if ( status[ i ].first < status[ i ].second )
				{
					finished = false;
				}
				rendered += status[ i ].first;
			}
			int new_step = rendered * nsteps / total;
			if ( new_step > step )
			{
				for ( auto i = step; i != new_step; ++i )
				{
					std::cout << "~";
				}
				step = new_step;
			}
			if ( finished ) break;
		}
		std::cout << " ]" << std::endl;

		for ( auto &th : ts )
		{
			th.join();
		}

		KINFO( tracer, "Tracer joint in", util::tick(), "seconds" );
	} );

}  // namespace core

}  // namespace koishi
