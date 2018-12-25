#pragma once

#include <util/image.hpp>
#include <core/basic/basic.hpp>
#include <core/misc/lens.hpp>
#include <core/misc/sampler.hpp>
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
#if defined( KOISHI_USE_CUDA )

constexpr int b = 1, kb = 1024 * b, mb = 1024 * kb;
constexpr int stackPoolSize = 1 * kb;  // keep 4 bytes per indice, dfs based bvh intersection queue won't exceed 32 ints due to the indice space limit of 2^32

template <typename Radiance, typename Alloc>
PolyFunction( DoIntegrate, Require<Radiance, Alloc, Device> )(

  ( poly::vector<float3> & buffer, const poly::object<Lens> &lens, SamplerGenerator &rng_gen, const Scene &scene, uint spp, uint unroll )
	->void {
		char stackPool[ stackPoolSize ];
		Alloc pool( stackPool, stackPoolSize );
		auto rng = rng_gen.create();

		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		uint y = blockIdx.y * blockDim.y + threadIdx.y;
		int line = blockDim.x * gridDim.x;
		int index = x + y * line;

		float invSpp = 1.0 / spp;

		float3 sum{ 0, 0, 0 };

		for ( uint i = 0; i < spp; ++i )
		{
			sum += call<Radiance>( lens->sample( x, y, i ), scene, pool, rng );
		}

		buffer[ index ] = sum * invSpp;
	} );

template <typename Radiance, typename Alloc>
__global__ void integrate( poly::vector<float3> &buffer, const poly::object<Lens> &lens, SamplerGenerator &rng_gen, const Scene &scene, uint spp, uint unroll )
{
	Device::call<DoIntegrate<Radiance, Alloc>>( buffer, lens, rng_gen, scene, spp, unroll );
}

template <typename Radiance, typename Alloc = HybridAllocator>
PolyFunction( CudaSingleGPUTracer, Require<Host, On<Radiance, Device>, On<Alloc, Device>> )(

  ( util::Image<3> & image, poly::object<Lens> &lens, SamplerGenerator &rng_gen, Scene &scene, uint spp )
	->void {
		uint w = image.width();
		uint h = image.height();

		poly::vector<float3> buffer( w * h );

		int nDevices;
		cudaDeviceProp prop;
		cudaGetDeviceCount( &nDevices );
		cudaGetDeviceProperties( &prop, 0 );

		int threadPerBlock = prop.maxThreadsPerBlock;
		int threadPerSM = prop.maxThreadsPerMultiProcessor;
		int sharedMemPerBlock = 0;

		int blockPerSM = 8;

		int blockDim = 16;

		KASSERT( w % blockDim == 0 && h % blockDim == 0 &&
				 blockDim * blockDim * blockPerSM <= threadPerSM );

		KLOG( "Using", blockDim * blockDim, "threads" );
		// calculate how many steps the integrater need to unrool the loop here
		uint unroll = spp & -spp;  // use the lowbit of spp as unrool stride, thus spp % unroll = 0
		KLOG( "Unrolling stride", unroll );

		poly::kernel( integrate<Radiance, Alloc>,
					  dim3( w / blockDim, h / blockDim ),
					  dim3( blockDim, blockDim ),
					  sharedMemPerBlock )( buffer, lens, rng_gen, scene, spp, unroll );

		for ( uint j = 0; j != h; ++j )
		{
			for ( uint i = 0; i != w; ++i )
			{
				image.at( i, j ) = buffer[ i + j * w ];
			}
		}
	} );

#endif

}  // namespace core

}  // namespace koishi
