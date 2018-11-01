#include <fstream>
#include "renderer.hpp"
#include <util/config.hpp>
#include <util/mesh.hpp>
#include <util/image.hpp>

namespace koishi
{
namespace fallback
{
Renderer::Renderer( uint w, uint h ) :
  buffer( w * h ), w( w ), h( h )
{
}

// static bool intersect( const Ray &ray, const util::SubMesh &mesh )
// {
// 	return ray.intersect_branch( mesh, 1 );
// }

void Renderer::render( const std::string &path, const std::string &dest, uint spp )
{
	jsel::Scene scene;
	std::ifstream( path ) >> scene;
	if ( !scene.camera.size() )
	{
		throw "no camera in the scene.";
	}
	auto &camera = scene.camera[ 0 ];
	std::vector<util::SubMesh> mesh;
	for ( auto &m : scene.mesh )
	{
		for ( auto &e : util::PolyMesh( m ).mesh )
		{
			mesh.emplace_back( std::move( e ) );
		}
	}

	rays.resize( spp * buffer.size() );
	float3 target = normalize( camera.target ) * float( h ) / ( 2 * tan( radians( camera.fovy * .5 ) ) );
	float3 U = normalize( cross( target, float3{ 0, 1, 0 } ) );
	float3 V = normalize( cross( U, target ) );
	for ( uint j = 0; j != h; ++j )
	{
		for ( uint i = 0; i != w; ++i )
		{
			auto *ray = &rays[ j * spp * w + i ];
			auto N = target + U * ( float( i ) - w * .5 + .5 ) + V * ( h * .5 - float( j ) - .5 );
			float3 diff{ sin( radians( 30. ) ), cos( radians( 30. ) ), 0 };
			for ( uint k = 0; k != spp; ++k )
			{
				ray[ k ].o = camera.position;
				ray[ k ].v = normalize( N );  // + .8 * float3{
											  //	   sin( radians( 30. + k * 360. / spp ) ),
											  //	   cos( radians( 30. + k * 360. / spp ) ), 0 } );
			}
		}
	}
	util::Image<3> image( 1024, 768 );
	uint p = 0;
	for ( auto &r : rays )
	{
		for ( auto &m : mesh )
		{
			// image.at( p % 1024, p / 1024 ) = { r.v.x, r.v.y, r.v.z };
			float t;
			float2 uv;
			unsigned char val = 255 * r.intersect( m, 1, t, uv );
			image.at( p % 1024, p / 1024 ) = { val, val, val };
		}
		p++;
	}
	image.dump( dest );
}

}  // namespace fallback

}  // namespace koishi