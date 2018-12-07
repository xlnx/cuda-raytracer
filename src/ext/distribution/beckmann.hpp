#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct BeckmannDistribution : IsotropicSphericalDistribution
{
	KOISHI_HOST_DEVICE BeckmannDistribution( const Properties &props ) :
	  alpha( get( props, "alpha", .5f ) )
	{
	}

	KOISHI_HOST_DEVICE float3 sample( const float3 &u, float &pdf ) const override
	{
		auto logu = log( u.x );
		auto alpha2 = alpha * alpha;
		auto tg2th = -alpha2 * logu;
		auto phi = u.y * 2 * PI;
		float costh = 1.f / sqrt( 1 + tg2th );
		auto sinth = sqrt( max( 0.f, 1.f - costh * costh ) );
		pdf = exp( -tg2th / alpha2 ) / ( PI * alpha2 ) * ( 1 + tg2th ) * ( 1 + tg2th );
		return hemisphere::fromEular( sinth, costh, phi );
	}

private:
	float alpha;
};

}  // namespace ext

}  // namespace koishi