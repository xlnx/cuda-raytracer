#pragma once

#include <vec/vmath.hpp>

namespace koishi
{
constexpr float invPI = 1. / M_PI;
constexpr float PI = M_PI;

namespace hemisphere
{
KOISHI_HOST_DEVICE inline bool isSame( const float3 &w0, const float3 &w1 )
{
	return w0.z * w1.z > 0.;
}

KOISHI_HOST_DEVICE inline float h( const float3 &w )
{
	return abs( w.z );
}

KOISHI_HOST_DEVICE inline float3 fromEular( float sinth, float costh, float phi )
{
	return float3{ sinth * sin( phi ), sinth * cos( phi ), costh };
}

KOISHI_HOST_DEVICE inline float3 fromEular( float th, float phi )
{
	return fromEular( sin( th ), cos( th ), phi );
}

KOISHI_HOST_DEVICE inline float3 sampleCos( const float2 &rn )
{
	auto uv = 2. * rn - 1;  // map uniform rn to [-1,1]
	if ( uv.x == 0 && uv.y == 0 )
	{
		return float3{ 0, 0, 1 };
	}

	float r, phi;
	if ( uv.x * uv.x > uv.y * uv.y )
	{
		r = uv.x, phi = ( PI / 4 ) * ( uv.y / uv.x );
	}
	else
	{
		r = uv.y, phi = ( PI / 2 ) - ( ( PI / 4 ) * ( uv.x / uv.y ) );
	}

	return float3{ r * cos( phi ), r * sin( phi ), sqrt( max( 0., 1. - r * r ) ) };
}

}  // namespace hemisphere

}  // namespace koishi