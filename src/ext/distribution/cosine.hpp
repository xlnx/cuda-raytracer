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
	KOISHI_HOST_DEVICE normalized_float3 sample( const float3 &u, float &pdf ) const override
	{
		auto w = hemisphere::sampleCos( float2{ u.x, u.y } );
		pdf = hemisphere::h( w ) / invPI;
		return w;
	}
};

}  // namespace ext

}  // namespace koishi