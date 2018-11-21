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
constexpr int sharedMemPerThread = 128 * b;	// keep 4 bytes per indice, dfs based bvh intersection queue won't exceed 32 ints due to the indice space limit of 2^32

namespace cuda
{
template <typename Radiance, typename Alloc>
__global__ void intergrate( PolyVector<double3> &buffer, const PolyVector<Ray> &rays, const Scene &scene, uint spp )
{
	extern __shared__ char sharedMem[];

	char *threadMem = sharedMem + sharedMemPerThread * threadIdx.x;
	Alloc pool( threadMem, sharedMemPerThread );

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double invSpp = 1.0 / spp;

	for ( uint k = index; k < buffer.size(); k += stride )  // k is the k-th pixel
	{
		int rayIndex = k * spp;
		double3 sum{ 0, 0, 0 };

		for ( uint i = rayIndex; i < rayIndex + spp; ++i )  // i is the i-th sample
		{
			//sum += rays[i].d;
			sum += Device::call<Radiance>( rays[ i ], scene, pool );
			//	clear( pool );
		}

		buffer[ k ] = sum * invSpp;
	}
}

template <typename Radiance, typename Alloc = HybridAllocator>
PolyFunction( Tracer, Require<Host> )(
  ( util::Image<3> & image, PolyVector<Ray> &rays, Scene &scene, uint spp )->void {
	  static_assert( std::is_base_of<Device, Radiance>::value, "Radiance must be host callable" );
	  static_assert( std::is_base_of<Device, Alloc>::value, "Alloc must be host callable" );

	  uint w = image.width();
	  uint h = image.height();

	  PolyVector<double3> buffer( w * h );

	  int nDevices;
	  cudaDeviceProp prop;
	  cudaGetDeviceCount( &nDevices );
	  cudaGetDeviceProperties( &prop, 0 );

	  int threadPerBlock = prop.maxThreadsPerBlock;
	  int threadShmLimit = prop.sharedMemPerBlock / sharedMemPerThread;
	  if ( threadShmLimit < threadPerBlock )
	  {
		  KLOG( "using", threadShmLimit, "of", threadPerBlock, "threads due to shared memory limit" );
		  threadPerBlock = threadShmLimit;
	  }
	  else
	  {
		  KLOG( "using", threadPerBlock, "threads" );
	  }
	  int sharedMemPerBlock = threadPerBlock * sharedMemPerThread;

	  KLOG1( "start intergrating" );

	  kernel( intergrate<Radiance, Alloc>,
			  prop.multiProcessorCount,
			  threadPerBlock,
			  sharedMemPerBlock )( buffer, rays, scene, spp );

	  KLOG2( "finished intergrating" );

	  for ( uint j = 0; j != h; ++j )
	  {
		  for ( uint i = 0; i != w; ++i )
		  {
			  image.at( i, j ) = buffer[ i + j * w ];
		  }
	  }
  } );

}  // namespace cuda

#endif

}  // namespace core

}  // namespace koishi
