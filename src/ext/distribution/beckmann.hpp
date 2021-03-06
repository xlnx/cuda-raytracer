#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct BeckmannDistribution : IsotropicSphericalDistribution
{
	BeckmannDistribution( const Properties &props ) :
	  IsotropicSphericalDistribution( props )
	{
		auto roughness = get( props, "roughness", .5f );
		roughness = roughness * roughness;
		alpha2 = roughness;
	}

	KOISHI_HOST_DEVICE float3 f( const solid &w ) const override
	{
		auto tg2th = H::tan2Theta( w );
		auto cos4th = H::cos2Theta( w ) * H::cos2Theta( w );
		return float3{ 1, 1, 1 } * exp( -tg2th / alpha2 ) / ( PI * alpha2 * cos4th ) * fabs( H::cosTheta( w ) );
	}

	KOISHI_HOST_DEVICE solid sample( const float3 &u, float &pdf ) const override
	{
		auto logu = log( u.x );
		auto tg2th = -alpha2 * logu;
		auto phi = u.y * 2 * PI;
		float costh = 1.f / sqrt( 1 + tg2th );
		float invcos2th = 1 + tg2th;
		auto sinth = sqrt( max( 0.f, 1.f - costh * costh ) );
		auto w = H::fromEular( sinth, costh, phi );
		pdf = exp( -tg2th / alpha2 ) / ( PI * alpha2 ) *
			  invcos2th * invcos2th * fabs( H::cosTheta( w ) );
		return w;
	}

	void writeNode( nlohmann::json &j ) const override
	{
		j[ "Beckmann" ] = sqrt( alpha2 );
	}

private:
	float alpha2;
};

}  // namespace ext

}  // namespace koishi