#pragma once

#include <vec/vmath.hpp>

namespace koishi
{
constexpr double invPI = 1. / M_PI;
constexpr double PI = M_PI;

namespace hemisphere
{
KOISHI_HOST_DEVICE bool isSame( const double3 &w0, const double3 &w1 )
{
	return w0.z * w1.z > 0.;
}

KOISHI_HOST_DEVICE double h( const double3 &w )
{
	return abs( w.z );
}

KOISHI_HOST_DEVICE double3 sampleCos( const double2 &rn )
{
	auto uv = 2. * rn - 1;  // map uniform rn to [-1,1]
	if ( uv.x == 0 && uv.y == 0 )
	{
		return double3{ 0, 0, 1 };
	}

	double r, phi;
	if ( uv.x * uv.x > uv.y * uv.y )
	{
		r = uv.x, phi = ( PI / 4 ) * ( uv.y / uv.x );
	}
	else
	{
		r = uv.y, phi = ( PI / 2 ) - ( ( PI / 4 ) * ( uv.x / uv.y ) );
	}

	return double3{ r * cos( phi ), r * sin( phi ), sqrt( max( 0., 1. - r * r ) ) };
}

}  // namespace hemisphere

}  // namespace koishi