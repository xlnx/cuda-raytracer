#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct BeckmannDistribution : IsotropicSphericalDistribution
{
	KOISHI_HOST_DEVICE BeckmannDistribution( const Properties &props ) :
	  alpha2( get( props, "roughness", .5f ) * get( props, "roughness", .5f ) )
	{
	}

	KOISHI_HOST_DEVICE float3 f( const float3 &w ) const override
	{
		auto tg2th = hemisphere::tan2Theta( w );
		auto cos4th = hemisphere::cos2Theta( w ) * hemisphere::cos2Theta( w );
		return float3{ 1, 1, 1 } * exp( -tg2th / alpha2 ) / ( PI * alpha2 * cos4th );
	}

	KOISHI_HOST_DEVICE normalized_float3 sample( const float3 &u, float &pdf ) const override
	{
		auto logu = log( u.x );
		auto tg2th = -alpha2 * logu;
		auto phi = u.y * 2 * PI;
		float costh = 1.f / sqrt( 1 + tg2th );
		float invcos2th = 1 + tg2th;
		auto sinth = sqrt( max( 0.f, 1.f - costh * costh ) );
		auto w = hemisphere::fromEular( sinth, costh, phi );
		pdf = exp( -tg2th / alpha2 ) / ( PI * alpha2 ) *
			  invcos2th * invcos2th * abs( hemisphere::cosTheta( w ) );
		return w;
	}

private:
	float alpha2;
};

}  // namespace ext

}  // namespace koishi