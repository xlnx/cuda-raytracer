#pragma once

#include <vec/vec.hpp>
#include <vec/vmath.hpp>
#include <util/config.hpp>
#include <util/debug.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/poly.hpp>

namespace koishi
{
namespace core
{
struct Lens : emittable
{
	Lens( const CameraConfig &camera, uint w, uint h, uint spp ) :
	  w( w ), h( h ), spp( spp ),
	  o( camera.position ),
	  n( normalize( camera.target ) * float( w ) / ( 2 * tan( radians( camera.fovx * .5 ) ) ) ),
	  u( normalize( cross( n, camera.upaxis ) ) ),
	  v( normalize( cross( n, reinterpret_cast<const float3 &>( u ) ) ) )
	{
	}

	KOISHI_HOST_DEVICE virtual Ray sample( uint x, uint y, uint k ) const = 0;

protected:
	KOISHI_HOST_DEVICE float3 origin( uint x, uint y ) const
	{
		return n + u * ( float( x ) - w * .5 ) + v * ( float( y ) - h * .5 );
	}

	KOISHI_HOST_DEVICE float3 center( uint x, uint y ) const
	{
		return n + u * ( float( x ) - w * .5 + .5 ) + v * ( float( y ) - h * .5 + .5 );
	}

protected:
	uint w, h, spp;

	float3 o, n;
	normalized_float3 u, v;
};

struct CircleLens : Lens
{
	CircleLens( const CameraConfig &config, uint w, uint h, uint spp ) :
	  Lens( config, w, h, spp )
	{
	}

	KOISHI_HOST_DEVICE Ray sample( uint x, uint y, uint k ) const override
	{
		auto t = center( x, y );
		return Ray{ o, normalize( t + .8 * ( u * sin( radians( 30. + k * 360. / spp ) ) ) +
								  ( v * cos( radians( 30. + k * 360. / spp ) ) ) ) };
	}
};

struct SquareLens : Lens
{
	SquareLens( const CameraConfig &config, uint w, uint h, uint spp ) :
	  Lens( config, w, h, spp ),
	  pw( ceil( sqrt( spp ) ) ),
	  du( u / float( pw ) ), dv( v / float( pw ) )
	{
	}

	KOISHI_HOST_DEVICE Ray sample( uint x, uint y, uint k ) const override
	{
		auto t = origin( x, y );
		return Ray{ o, normalize( t + du * ( k % pw ) + dv * ( k / pw ) ) };
	}

private:
	uint pw;

	float3 du, dv;
};

}  // namespace core

}  // namespace koishi
