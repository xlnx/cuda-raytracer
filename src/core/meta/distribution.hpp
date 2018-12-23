#pragma once

#include <vec/vmath.hpp>
#include <util/hemisphere.hpp>
#include <util/config.hpp>
#include <core/basic/factory.hpp>

namespace koishi
{
namespace core
{
struct SphericalDistribution : emittable
{
	SphericalDistribution() = default;
	SphericalDistribution( const Properties &config ) {}

	KOISHI_HOST_DEVICE virtual float3 f( const solid &w ) const = 0;
	KOISHI_HOST_DEVICE virtual solid sample( const float3 &u, float &pdf ) const = 0;
};

struct IsotropicSphericalDistribution : SphericalDistribution
{
	IsotropicSphericalDistribution() = default;
	IsotropicSphericalDistribution( const Properties &config ) :
	  SphericalDistribution( config )
	{
	}
	KOISHI_HOST_DEVICE solid sample( const float2 &u, float &pdf ) const
	{
		return SphericalDistribution::sample( float3{ u.x, u.y, 0.f }, pdf );
	}
};

}  // namespace core

}  // namespace koishi