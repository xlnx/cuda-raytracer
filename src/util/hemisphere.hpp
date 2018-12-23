#pragma once

#include <vec/vmath.hpp>

namespace koishi
{
constexpr float invPI = 1. / M_PI;
constexpr float PI = M_PI;

using solid = normalized_float3;

namespace hemisphere
{
KOISHI_HOST_DEVICE inline bool isSame( const solid &w0, const solid &w1 )
{
	return w0.z * w1.z > 0.;
}

KOISHI_HOST_DEVICE inline float cosTheta( const solid &w )
{
	return w.z;
}

KOISHI_HOST_DEVICE inline float cos2Theta( const solid &w )
{
	return w.z * w.z;
}

KOISHI_HOST_DEVICE inline float sin2Theta( const solid &w )
{
	return max( 0.f, 1 - cos2Theta( w ) );
}

KOISHI_HOST_DEVICE inline float sinTheta( const solid &w )
{
	return sqrt( sin2Theta( w ) );
}

KOISHI_HOST_DEVICE inline float tan2Theta( const solid &w )
{
	return sin2Theta( w ) / cos2Theta( w );
}

KOISHI_HOST_DEVICE inline float tanTheta( const solid &w )
{
	return sqrt( tan2Theta( w ) );
}

KOISHI_HOST_DEVICE inline float sinPhi( const solid &w )
{
	auto sinth = sinTheta( w );
	return sinth == 0 ? 1 : max( -1.f, min( 1.f, w.y / sinth ) );
}

KOISHI_HOST_DEVICE inline float cosPhi( const solid &w )
{
	auto sinth = sinTheta( w );
	return sinth == 0 ? 1 : max( -1.f, min( 1.f, w.x / sinth ) );
}

KOISHI_HOST_DEVICE inline float tanPhi( const solid &w )
{
	return sinPhi( w ) / cosPhi( w );
}

KOISHI_HOST_DEVICE inline solid fromEular( float sinth, float costh, float phi )
{
	return solid( float3{ sinth * sin( phi ), sinth * cos( phi ), costh } );
}

KOISHI_HOST_DEVICE inline solid fromEular( float th, float phi )
{
	return fromEular( sin( th ), cos( th ), phi );
}

KOISHI_HOST_DEVICE inline solid sampleCos( const float2 &rn )
{
	auto uv = 2. * rn - 1;  // map uniform rn to [-1,1]
	if ( uv.x == 0 && uv.y == 0 )
	{
		return solid( float3{ 0, 0, 1 } );
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

	return solid( float3{ r * cos( phi ), r * sin( phi ), sqrt( max( 0., 1. - r * r ) ) } );
}

}  // namespace hemisphere

}  // namespace koishi