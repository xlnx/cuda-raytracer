#pragma once

#include <vector>
#include <vec/vec.hpp>
#include <util/image.hpp>
#include "mesh.hpp"
#include "ray.hpp"
#include "dev.hpp"

namespace koishi
{
namespace core
{
template <typename Radiance>
struct Tracer
{
	void trace( util::Image<3> &image, const std::vector<Ray> &rays, const std::vector<SubMesh> &meshs, uint spp )
	{
		uint w = image.width();
		uint h = image.height();

		for ( uint j = 0; j != h; ++j )
		{
			for ( uint i = 0; i != w; ++i )
			{
				double3 rad = { 0, 0, 0 };
				for ( uint k = 0; k != spp; ++k )
				{
					rad += Radiance::radiance( rays[ ( j * w + i ) * spp + k ], meshs );
				}
				image.at( i, j ) = rad / spp;
			}
		}
	}
};

#if defined( KOISHI_USE_CUDA )

namespace cuda
{
template <typename Radiance>
struct Tracer
{
	void trace( util::Image<3> &image, const std::vector<Ray> &rays, const std::vector<SubMesh> &meshs, uint spp )
	{
		uint w = image.width();
		uint h = image.height();

		float gridDim = w;
		float blockDim = spp;

		dev::vector<Ray> devRays = rays;
		dev::vector<double3> devBuffer( w * h, 0 );
		dev::vector<dev::SubMesh> devMeshs( meshs.begin(), meshs.end() );

		intergrate<Radiance><<<gridDim, blockDim, h * sizeof( double3 )>>>( devRays, devBuffer, devMeshs, rays.size(), h );

		dev::host_vector<double3> buffer( devBuffer );

		for ( uint j = 0; j != h; ++j )
		{
			for ( uint i = 0; i != w; ++i )
			{
				image.at( i, j ) = buffer[ i + j * w ];
			}
		}
	}
};

template <typename Radiance>
	__global__ static void intergrate( const dev::vector<Ray> &rays,
									   const dev::vector<double3> &buffer,
									   const dev::vector<dev::SubMesh> &meshs,
									   uint N, uint h )
	{
		// __shared__ double3 rad = { 0, 0, 0 };
		extern __shared__ double3 rad[];

		auto index = threadIdx.x + blockIdx.x * blockDim.x;
		auto stride = gridDim.x * blockDim.x;
		for ( uint i = index, j = 0; i < N; i += stride, ++j )
		{
			rad[ j ] += Radiance::radiance( rays[ i ], meshs );
		}

		__syncthreads();

		for ( uint i = threadIdx; i < h; i += blockDim.x )
		{
			buffer[ i * gridDim.x + blockIdx.x ] = rad[ i ];
		}
	}

}  // namespace cuda

#endif

}  // namespace core

}  // namespace koishi
