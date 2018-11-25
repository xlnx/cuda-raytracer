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

constexpr int b = 1, kb = 1024 * b, mb = 1024 * kb;
constexpr int stackPoolSize = 1 * kb;  // keep 4 bytes per indice, dfs based bvh intersection queue won't exceed 32 ints due to the indice space limit of 2^32

template <typename Radiance, typename Alloc>
__global__ void intergrate( PolyVector<float3> &buffer, const PolyVector<Ray> &rays, const Scene &scene, uint spp )
{
	char stackPool[ stackPoolSize ];
	Alloc pool( stackPool, stackPoolSize );

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	float invSpp = 1.0 / spp;

	for ( uint k = index; k < buffer.size(); k += stride )  // k is the k-th pixel
	{
		int rayIndex = k * spp;
		float3 sum{ 0, 0, 0 };

		for ( uint i = rayIndex; i < rayIndex + spp; ++i )  // i is the i-th sample
		{
			auto ray = rays[ i ];
			sum += Device::call<Radiance>( ray, scene, pool );
			//	clear( pool );
		}

		buffer[ k ] = sum * invSpp;
	}
}

template <typename Radiance, typename Alloc = HybridAllocator>
PolyFunction( CudaSingleGPUTracer, Require<Host, On<Radiance, Device>, On<Alloc, Device>> )(
  ( util::Image<3> & image, PolyVector<Ray> &rays, Scene &scene, uint spp )->void {
	  uint w = image.width();
	  uint h = image.height();

	  PolyVector<float3> buffer( w * h );

	  int nDevices;
	  cudaDeviceProp prop;
	  cudaGetDeviceCount( &nDevices );
	  cudaGetDeviceProperties( &prop, 0 );

	  int threadPerBlock = prop.maxThreadsPerBlock;
	  int threadPerSM = prop.maxThreadsPerMultiProcessor;
	  int sharedMemPerBlock = 0;

	  int blockPerSM = 8;

	  KLOG( "Using", threadPerBlock, "threads" );

	  kernel( intergrate<Radiance, Alloc>,
			  prop.multiProcessorCount * blockPerSM * 8,
			  threadPerSM / blockPerSM,
			  sharedMemPerBlock )( buffer, rays, scene, spp );

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
