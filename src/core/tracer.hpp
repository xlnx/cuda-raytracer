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
  ( util::Image<3> & image, PolyVectorView<Ray> &rays, Scene &scene, uint spp )->void {
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

namespace cuda
{
template <typename Radiance, typename Alloc>
__global__ void intergrate( PolyVectorView<double3> &buffer, const PolyVectorView<Ray> &rays, const Scene &scene, uint h )
{
	Alloc pool;

	extern __shared__ double3 rad[];

	auto index = threadIdx.x + blockIdx.x * blockDim.x;
	auto stride = gridDim.x * blockDim.x;
	for ( uint i = index, j = 0; i < rays.size(); i += stride, ++j )
	{
		rad[ j ] += Device::call<Radiance>( rays[ i ], scene, pool );
		clear( pool );
	}

	__syncthreads();

	for ( uint i = threadIdx.x; i < h; i += blockDim.x )
	{
		buffer[ i * gridDim.x + blockIdx.x ] = rad[ i ];
	}
}

template <typename Radiance, typename Alloc = DeviceAllocator>
PolyFunction( Tracer, Require<Host> )(
  ( util::Image<3> & image, PolyVectorView<Ray> &rays, Scene &scene, uint spp )->void {
	  static_assert( std::is_base_of<Device, Radiance>::value, "Radiance must be host callable" );
	  static_assert( std::is_base_of<Device, Alloc>::value, "Alloc must be host callable" );

	  uint w = image.width();
	  uint h = image.height();

	  float gridDim = w;
	  float blockDim = spp;

	  //rays.emitAndReplace();
	  //scene.emitAndReplace();
	  PolyVectorView<double3> buffer( w * h );
	  //buffer.emitAndReplace();

	  kernel(intergrate<Radiance, Alloc>, gridDim, blockDim, h * sizeof(double3)) 
		  (buffer, rays, scene, h);

	  //buffer.fetchAndReplace();

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
