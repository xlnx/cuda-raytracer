#pragma once

#include "node.hpp"

namespace koishi
{
namespace core
{
struct SphericalDistribution : Node
{
	SphericalDistribution( const Properties &props ) :
	  Node( props )
	{
	}

	KOISHI_HOST_DEVICE virtual float3 f( const solid &w ) const = 0;
	KOISHI_HOST_DEVICE virtual solid sample( const float3 &u, float &pdf ) const = 0;
	KOISHI_HOST_DEVICE solid sample( const float3 &u ) const
	{
		float pdf;
		return sample( u, pdf );
	}
};

struct IsotropicSphericalDistribution : SphericalDistribution
{
	IsotropicSphericalDistribution( const Properties &props ) :
	  SphericalDistribution( props )
	{
	}
	KOISHI_HOST_DEVICE solid sample( const float2 &u, float &pdf ) const
	{
		return SphericalDistribution::sample( float3{ u.x, u.y, 0.f }, pdf );
	}
};

}  // namespace core

}  // namespace koishi