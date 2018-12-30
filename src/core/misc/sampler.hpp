#pragma once

#include <cstdlib>
#include <ctime>
#include <vec/vmath.hpp>
#include <core/basic/basic.hpp>
#ifdef KOISHI_USE_CUDA
#include <cuda.h>
#include <curand.h>
#endif

namespace koishi
{
namespace core
{
struct SamplerGenerator;

struct Sampler
{
	friend struct SamplerGenerator;

	KOISHI_HOST_DEVICE float sample()
	{
#ifdef __CUDA_ARCH__
		return this->nums->operator[]( id += det );
#else
		static unsigned long long seed = ( ( (long long int)time( nullptr ) ) << 16 ) | ::rand();

		constexpr auto m = 0x100000000LL;
		constexpr auto c = 0xB16;
		constexpr auto a = 0x5DEECE66DLL;
		seed = ( a * seed + c ) & 0xFFFFFFFFFFFFLL;
		unsigned int x = seed >> 16;
		return ( (float)x / (float)m );
#endif
	}

	KOISHI_HOST_DEVICE float2 sample2()
	{
		return float2{ sample(), sample() };
	}

	KOISHI_HOST_DEVICE float3 sample3()
	{
		return float3{ sample(), sample(), sample() };
	}

	KOISHI_HOST_DEVICE float4 sample4()
	{
		return float4{ sample(), sample(), sample(), sample() };
	}

	KOISHI_HOST Sampler() = default;

private:
	using ull = unsigned long long;
	KOISHI_HOST_DEVICE Sampler( const poly::vector<float> &nums )
	{
		this->nums = &nums;
#ifdef __CUDA_ARCH__
		ull bd_x = blockDim.x;
		ull bd_xy = bd_x * blockDim.y;
		ull bd_xyz = bd_xy * blockDim.z;
		ull gd_x = gridDim.x;
		ull gd_xy = gd_x * gridDim.y;
		ull gd_xyz = gd_xy * gridDim.z;
		ull th_idx = threadIdx.x +
					 threadIdx.y * bd_x +
					 threadIdx.z * bd_xy;
		ull bl_idx = blockIdx.x +
					 blockIdx.y * gd_x +
					 blockIdx.z * gd_xy;
		ull idx = th_idx +
				  bl_idx * bd_xyz;

		auto m = 0x100000000LL;
		auto c = 0xB16;
		auto a = 0x5DEECE66DLL;
		auto seed = ( a * idx + c ) & 0xFFFFFFFFFFFFLL;
		unsigned int x = seed >> 8;
		float r = (float)x / (float)m;
		this->id = uint( this->nums->size() * r ) % this->nums->size();
		seed = ( a * seed + c ) & 0xFFFFFFFFFFFFLL;
		x = seed >> 8;
		r = (float)x / (float)m;
		this->det = uint( this->nums->size() * r ) % ( this->nums->size() / 4 ) + 1u;
#endif
	}

private:
	uint16_t id = 0, det = 1;
	const poly::vector<float> *nums;
};

struct SamplerGenerator : emittable
{
#ifdef KOISHI_USE_CUDA
	SamplerGenerator() :
	  nums( int( uint16_t( -1u ) ) + 1 )
	{
		float *value = nullptr, *dev_ptr;
		std::size_t count = 0;
		curandGenerator_t gen;

		nums.swap( value, count );

		cudaMalloc( &dev_ptr, count * sizeof( float ) );
		curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_MTGP32 );
		curandSetPseudoRandomGeneratorSeed( gen, (long long int)time( nullptr ) );
		curandGenerateUniform( gen, dev_ptr, count );
		cudaMemcpy( value, dev_ptr, count * sizeof( float ), cudaMemcpyDeviceToHost );
		curandDestroyGenerator( gen );
		cudaFree( dev_ptr );

		nums.swap( value, count );
	}
#else
	SamplerGenerator() :
	  nums( 0 )
	{
	}
#endif
	KOISHI_HOST_DEVICE Sampler create() const
	{
		return Sampler( nums );
	}

private:
	poly::vector<float> nums;
};

}  // namespace core

}  // namespace koishi
