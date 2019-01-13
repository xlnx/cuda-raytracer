#pragma once

#include <core/basic/basic.hpp>
#include "tracer.hpp"

namespace koishi
{
namespace core
{
#if defined( KOISHI_USE_CUDA )

constexpr int b = 1, kb = 1024 * b, mb = 1024 * kb;
constexpr int stackPoolSize = 1 * kb;  // keep 4 bytes per indice, dfs based bvh intersection queue won't exceed 32 ints due to the indice space limit of 2^32


__global__ void integrate( poly::object<Kernel> &kern, poly::vector<float3> &buffer, const poly::object<Lens> &lens, SamplerGenerator &rng_gen,
						   const Scene &scene, uint spp, uint unroll )
{
    char stackPool[ stackPoolSize ];
	HybridAllocator pool( stackPool, stackPoolSize );
	auto rng = rng_gen.create();

	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	int line = blockDim.x * gridDim.x;
	int index = x + y * line;

    float invSpp = 1.0 / spp;

	float3 sum{ 0, 0, 0 };

    for ( uint i = 0; i < spp; ++i )
	{
		sum += kern->execute( lens->sample( x, y, i ), scene, pool, rng, nullptr );
	}

	buffer[ index ] = sum * invSpp;
}

struct CudaSingleGPUTracer : Tracer
{
    struct Configuration : serializable<Configuration>
    {
      //  Property();
    };

    CudaSingleGPUTracer( const Properties &props ):
      Tracer( props )
    {
    }

    void execute( util::Image<3> & image, poly::object<Lens> &lens, SamplerGenerator &rng_gen,
	              Scene &scene, uint spp, poly::object<Profiler> &profiler )
	{
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

		poly::kernel( integrate,
					  dim3( w / blockDim, h / blockDim ),
					  dim3( blockDim, blockDim ),
					  sharedMemPerBlock )( this->kern, buffer, lens, rng_gen, scene, spp, unroll );

		for ( uint j = 0; j != h; ++j )
		{
			for ( uint i = 0; i != w; ++i )
			{
				image.at( i, j ) = buffer[ i + j * w ];
			}
		}
	}
};

#endif

}  // namespace core

}  // namespace koishi
