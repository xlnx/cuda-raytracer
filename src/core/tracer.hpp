#pragma once

#include <vector>
#include <thread>
#include <iostream>
#include <vec/vec.hpp>
#include <util/image.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/poly.hpp>
#include <core/basic/allocator.hpp>
#include <core/meta/mesh.hpp>
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
template <typename Radiance, typename Alloc = HostAllocator, uint MaxThreads = -1u>
PolyFunction( Tracer, Require<Host, Radiance, Alloc> )(
  ( util::Image<3> & image, PolyVector<Ray> &rays, Scene &scene, uint spp )->void {
	  Alloc pool;

	  uint w = image.width();
	  uint h = image.height();

	  auto ncores = std::thread::hardware_concurrency();
	  if ( MaxThreads < ncores ) ncores = MaxThreads;
	  std::cout << "using " << ncores << " threads:" << std::endl;
	  std::vector<std::thread> ts;
	  auto tracer_thread = [ncores, spp, h, w, &scene, &image, &rays, &pool]( uint id ) {
		  for ( uint j = id; j < h; j += ncores )
		  {
			  for ( uint i = 0; i != w; ++i )
			  {
				  double3 rad = { 0, 0, 0 };
				  for ( uint k = 0; k != spp; ++k )
				  {
					  rad += Self::template call<Radiance>( rays[ ( j * w + i ) * spp + k ], scene, pool );
					  clear( pool );
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
  } );

#if defined( KOISHI_USE_CUDA )

constexpr int b = 1, kb = 1024 * b, mb = 1024 * kb;
constexpr int sharedMemPerThread = 48 * b;

namespace cuda
{
template <typename Radiance, typename Alloc>
__global__ void intergrate( PolyVector<double3> &buffer, const PolyVector<Ray> &rays, const Scene &scene, uint spp )
{
	Alloc pool;

	extern __shared__ char sharedMem[];

	char *threadMem = sharedMem + sharedMemPerThread * threadIdx.x;

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

template <typename Radiance, typename Alloc = DeviceAllocator>
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
	  if ( threadShmLimit > threadPerBlock )
	  {
		  LOG( "using", threadShmLimit, "of", threadPerBlock, "threads due to shared memory limit" );
		  threadPerBlock = threadShmLimit;
	  }
	  else
	  {
		  LOG( "using", threadPerBlock, "threads" );
	  }
	  int sharedMemPerBlock = threadPerBlock * sharedMemPerThread;

	  LOG1( "start intergrating" );

	  kernel( intergrate<Radiance, Alloc>,
			  prop.multiProcessorCount,
			  threadPerBlock,
			  sharedMemPerBlock )( buffer, rays, scene, spp );

	  LOG2( "finished intergrating" );

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
