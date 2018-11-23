#pragma once

#include <vector>
#include <vec/vec.hpp>
#include <util/config.hpp>
#include <util/debug.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/poly.hpp>

namespace koishi
{
namespace core
{
struct Sampler
{
	Sampler( uint w, uint h ) :
	  w( w ), h( h )
	{
	}

	PolyVector<Ray> sample( const jsel::Camera &camera, uint spp ) const
	{
		KINFO( sampler, "Sampling rays" );
		util::tick();
		PolyVector<Ray> rays( spp * w * h );
		float3 target = normalize( camera.target ) * float( w ) / ( 2 * tan( radians( camera.fovx * .5 ) ) );
		float3 U = normalize( cross( target, camera.upaxis ) );
		float3 V = normalize( cross( U, target ) );
		for ( uint j = 0; j != h; ++j )
		{
			for ( uint i = 0; i != w; ++i )
			{
				auto *ray = &rays[ ( j * w + i ) * spp ];
				auto N = target + U * ( float( i ) - w * .5 + .5 ) + V * ( h * .5 - float( j ) - .5 );
				// float3 diff = sin( radians( 30. ) ) * U + cos( radians( 30. ) ) * V;
				for ( uint k = 0; k != spp; ++k )
				{
					ray[ k ].o = camera.position;
					ray[ k ].d = normalize( N + .8 * ( U * sin( radians( 30. + k * 360. / spp ) ) +
													   V * cos( radians( 30. + k * 360. / spp ) ) ) );
				}
			}
		}
		KINFO( sampler, "Sampled", rays.size() , "rays in", util::tick(), "seconds" );
		return std::move( rays );
	}

private:
	uint w, h;
};

}  // namespace core

}  // namespace koishi
