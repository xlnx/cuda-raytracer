#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct CosDistribution : IsotropicSphericalDistribution
{
	CosDistribution( const Properties &props )
	{
	}

	KOISHI_HOST_DEVICE float3 f( const float3 &w ) const override
	{
		return abs( hemisphere::cosTheta( w ) ) / invPI * float3{ 1, 1, 1 };
	}
	KOISHI_HOST_DEVICE normalized_float3 sample( const float3 &u, float &pdf ) const override
	{
		auto w = hemisphere::sampleCos( float2{ u.x, u.y } );
		pdf = abs( hemisphere::cosTheta( w ) ) / invPI;
		return w;
	}
};

}  // namespace ext

}  // namespace koishi