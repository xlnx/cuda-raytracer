#pragma once

#include <vec/vec.hpp>
#include <vec/vmath.hpp>
#include <util/config.hpp>
#include <util/debug.hpp>
#include <core/basic/basic.hpp>

namespace koishi
{
namespace core
{
struct Lens : emittable
{
	Lens( const CameraConfig &camera, uint w, uint h, uint spp ) :
	  w( w ),
	  h( h ),
	  spp( spp ),
	  pw( ceil( sqrt( spp ) ) ),
	  o( camera.position ),
	  n( normalize( camera.target ) * float( w ) / ( 2 * tan( radians( camera.fovx * .5 ) ) ) ),
	  u( normalize( cross( n, camera.upaxis ) ) ),
	  v( normalize( cross( n, reinterpret_cast<const float3 &>( u ) ) ) ),
	  du( u / float( pw ) ),
	  dv( v / float( pw ) )
	{
	}

	KOISHI_HOST_DEVICE virtual Ray sample( uint x, uint y, uint k ) const = 0;

protected:
	uint w, h, spp, pw;

	float3 o;
	normalized_float3 n, u, v;

	float3 du, dv;
};

struct OrthographicLens : Lens
{
	OrthographicLens( const CameraConfig &config, uint w, uint h, uint spp ) :
	  Lens( config, w, h, spp )
	{
	}

	KOISHI_HOST_DEVICE Ray sample( uint x, uint y, uint k ) const override
	{
		auto d = u * ( float( x ) - w * .5 ) + v * ( float( y ) - h * .5 );
		d += du * ( k % pw ) + dv * ( k / pw );
		return Ray{ o + d * 4.f / w, n };
	}
};

struct PinholeLens : Lens
{
	PinholeLens( const CameraConfig &config, uint w, uint h, uint spp ) :
	  Lens( config, w, h, spp )
	{
	}

	KOISHI_HOST_DEVICE Ray sample( uint x, uint y, uint k ) const override
	{
		auto t = n + u * ( float( x ) - w * .5 ) + v * ( float( y ) - h * .5 );
		return Ray{ o, normalize( t + du * ( k % pw ) + dv * ( k / pw ) ) };
	}
};

}  // namespace core

}  // namespace koishi
