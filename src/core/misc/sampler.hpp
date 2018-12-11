#pragma once

#include <cstdlib>
#include <ctime>
#include <vec/vmath.hpp>
#include <core/basic/poly.hpp>
#ifdef KOISHI_USE_CUDA
#include <cuda.h>
#include <curand.h>
#endif

namespace koishi
{
namespace core
{
struct Sampler : emittable
{
#ifdef KOISHI_USE_CUDA
	Sampler() :
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
		cudaMemcpy( valuem, dev_ptr, count * sizeof( float ), cudaMemcpyDeviceToHost );
		curandDestroyGenerator( gen );
		cudaFree( dev_ptr );

		nums.swap( value, count );
	}
#else
	Sampler() :
	  nums( 0 )
	{
	}
#endif

	KOISHI_HOST_DEVICE float sample()
	{
#ifdef __CUDA_ARCH__
		return nums[ id++ ];
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

private:
	uint16_t id = 0;
	poly::vector<float> nums;
};

}  // namespace core

}  // namespace koishi
