#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct CosDistribution : IsotropicSphericalDistribution
{
	CosDistribution() = default;

	CosDistribution( const Properties &props )
	{
	}

	KOISHI_HOST_DEVICE float3 f( const solid &w ) const override
	{
		return w.z >= 0 ? invPI * float3{ 1, 1, 1 } : float3{ 0, 0, 0 };
	}
	KOISHI_HOST_DEVICE solid sample( const float3 &u, float &pdf ) const override
	{
		auto w = H::sampleCos( float2{ u.x, u.y } );
		pdf = invPI;
		return w;
	}
};

}  // namespace ext

}  // namespace koishi