#pragma once

#include <vector>
#include <vec/vec.hpp>
#include <util/ray.hpp>
#include <util/config.hpp>
#include <util/upaxis.hpp>

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

	std::vector<util::Ray> sample( const jsel::Camera &camera, uint spp ) const
	{
		std::vector<util::Ray> rays( spp * w * h );
		double3 target = normalize( camera.target ) * double( h ) / ( 2 * tan( radians( camera.fovy * .5 ) ) );
		double3 U = normalize( cross( target, util::upaxis ) );
		double3 V = normalize( cross( U, target ) );
		for ( uint j = 0; j != h; ++j )
		{
			for ( uint i = 0; i != w; ++i )
			{
				auto *ray = &rays[ ( j * w + i ) * spp ];
				auto N = target + U * ( double( i ) - w * .5 + .5 ) + V * ( h * .5 - double( j ) - .5 );
				// double3 diff = sin( radians( 30. ) ) * U + cos( radians( 30. ) ) * V;
				for ( uint k = 0; k != spp; ++k )
				{
					ray[ k ].o = camera.position;
					ray[ k ].v = normalize( N + .8 * ( U * sin( radians( 30. + k * 360. / spp ) ) +
													   V * cos( radians( 30. + k * 360. / spp ) ) ) );
				}
			}
		}
		return std::move( rays );
	}

private:
	uint w, h;
};

}  // namespace core

}  // namespace koishi