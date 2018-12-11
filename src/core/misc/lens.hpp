#pragma once

#include <vec/vec.hpp>
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
	  v( normalize( cross( reinterpret_cast<const float3 &>( u ), n ) ) )
	{
	}

	KOISHI_HOST_DEVICE Ray sample( uint x, uint y, uint k ) const
	{
		auto t = n + u * ( float( x ) - w * .5 + .5 ) + v * ( h * .5 - float( y ) - .5 );
		return Ray{ o, normalize( t + .8 * ( u * sin( radians( 30. + k * 360. / spp ) ) ) +
								  ( v * cos( radians( 30. + k * 360. / spp ) ) ) ) };
	}

private:
	uint w, h, spp;

	float3 o, n;
	normalized_float3 u, v;
};

}  // namespace core

}  // namespace koishi
