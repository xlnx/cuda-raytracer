#include <fstream>
#include "renderer.hpp"
#include <util/config.hpp>

namespace koishi
{
namespace fallback
{
Renderer::Renderer( uint w, uint h ) :
  buffer( w * h ), w( w ), h( h )
{
}

void Renderer::render( const std::string &path, uint spp )
{
	jsel::Scene scene;
	std::ifstream( path ) >> scene;
	if ( !scene.camera.size() )
	{
		throw "no camera in the scene.";
	}
	auto &camera = scene.camera[ 0 ];

	rays.resize( spp * buffer.size() );
	float3 origin;
	float3 target{ 0, 0, -float( h ) / ( 2 * tan( radians( camera.fovy * .5 ) ) ) };
	for ( uint j = 0; j != h; ++j )
	{
		for ( uint i = 0; i != w; ++i )
		{
			auto *ray = &rays[ j * spp * w + i ];
			target.x = float( i ) - w * .5 + .5;  // make sure target at grid center
			target.y = h * .5 - float( j ) - .5;
			float3 diff{ sin( radians( 30. ) ), cos( radians( 30. ) ), 0 };
			for ( uint k = 0; k != spp; ++k )
			{
				ray[ k ].v = normalize( target + .8 * float3{
														sin( radians( 30. + k * 360. / spp ) ),
														cos( radians( 30. + k * 360. / spp ) ), 0 } );
			}
		}
	}
}

}  // namespace fallback

}  // namespace koishi