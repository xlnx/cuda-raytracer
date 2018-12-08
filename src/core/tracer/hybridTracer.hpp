#pragma once

#include <vec/vec.hpp>
#include <util/debug.hpp>
#include <util/image.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/poly.hpp>
#include <core/basic/allocator.hpp>
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
#if defined( KOISHI_USE_CUDA )

namespace hybrid
{
constexpr int b = 1, kb = 1024 * b, mb = 1024 * kb;
constexpr int stackPoolSize = 1 * kb;  // keep 4 bytes per indice, dfs based bvh intersection queue won't exceed 32 ints due to the indice space limit of 2^32

template <typename Radiance, typename Alloc>
PolyFunction( DoHybridIntegrate, Require<Radiance, Alloc, Device> )

  ( poly::vector<float3> &buffer, const poly::vector<Ray> &rays, const Scene &scene, uint spp, uint unroll )
	->void
{
	char stackPool[ stackPoolSize ];
	Alloc pool( stackPool, stackPoolSize );

	int line = blockDim.x * gridDim.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x +
				( blockIdx.y * blockDim.y + threadIdx.y ) * line;

	float invSpp = 1.0 / spp;

	int rayIndex = index * spp;
	float3 sum{ 0, 0, 0 };

#define KOISHI_CUDA_INTEGRATER_MAIN_LOOP               \
	for ( uint i = rayIndex; i < rayIndex + spp; ++i ) \
	{                                                  \
		auto ray = rays[ i ];                          \
		sum += call<Radiance>( ray, scene, pool );     \
		pool.clear();                                  \
	}                                                  \
	break;

	// do loop unrolling work
	switch ( unroll )  // no divergence yah
	{
	case 1:  // no unroll
		KOISHI_CUDA_INTEGRATER_MAIN_LOOP
	case 2:
#pragma unroll( 2 )
		KOISHI_CUDA_INTEGRATER_MAIN_LOOP
	case 4:
#pragma unroll( 4 )
		KOISHI_CUDA_INTEGRATER_MAIN_LOOP
	default:  // no larger unrolls due to code size
#pragma unroll( 8 )
		KOISHI_CUDA_INTEGRATER_MAIN_LOOP
	}

#undef KOISHI_CUDA_INTEGRATER_MAIN_LOOP

	buffer[ index ] = sum * invSpp;
}

EndPolyFunction();

template <typename Radiance, typename Alloc>
__global__ void hybridIntegrate( poly::vector<float3> &buffer, const poly::vector<Ray> &rays, const Scene &scene, uint spp, uint unroll )
{
	Device::call<DoHybridIntegrate<Radiance, Alloc>>( buffer, rays, scene, spp, unroll );
}

template <typename Radiance, typename Alloc = HybridAllocator>
PolyFunction( HybridTracer, Require<Host, On<Radiance, Device>, On<Alloc, Device>> )

  ( util::Image<3> &image, poly::vector<Ray> &rays, Scene &scene, uint spp )
	->void
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

	poly::kernel( hybridIntegrate<Radiance, Alloc>,
				  dim3( w / blockDim, h / blockDim ),
				  dim3( blockDim, blockDim ),
				  sharedMemPerBlock )( buffer, rays, scene, spp, unroll );

	for ( uint j = 0; j != h; ++j )
	{
		for ( uint i = 0; i != w; ++i )
		{
			image.at( i, j ) = buffer[ i + j * w ];
		}
	}
}

EndPolyFunction();

}  // namespace hybrid

using hybrid::HybridTracer;

#endif

}  // namespace core

}  // namespace koishi
