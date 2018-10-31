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

static bool intersect_bbox( const Ray &ray, const float3 &vmin, const float3 &vmax )
{
	auto a = ( vmin - ray.o ) / ray.v, b = ( vmax - ray.o ) / ray.v;
	auto tmin = min( a, b ), tmax = max( a, b );
	auto t0 = max( tmin.x, max( tmin.y, tmin.z ) ), t1 = min( tmax.x, min( tmax.y, tmax.z ) );
	return t0 <= t1 && t1 >= 0;
}

static bool intersect_branch( const Ray &ray, const util::SubMesh &mesh, uint root )
{
	uint i = root;
	while ( !mesh.bvh[ i ].isleaf )
	{
		auto left = intersect_bbox( ray, mesh.bvh[ i << 1 ].vmin, mesh.bvh[ i << 1 ].vmax );
		auto right = intersect_bbox( ray, mesh.bvh[ ( i << 1 ) + 1 ].vmin, mesh.bvh[ ( i << 1 ) + 1 ].vmax );
		if ( !left && !right ) return false;
		if ( left && right )
		{
			//
			return false;  ///
		}
		i <<= 1;
		if ( right ) i += 1;
	}
	// return true;
	for ( uint j = mesh.bvh[ i ].begin; j != mesh.bvh[ i ].end; ++j )
	{
		// if (intersect_triangle(mesh.indices))
	}
}

static bool intersect( const Ray &ray, const util::SubMesh &mesh )
{
	return intersect_branch( ray, mesh, 1 );
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
			unsigned char val = 255 * intersect_bbox( r, m.bvh[ 1 ].vmin, m.bvh[ 1 ].vmax );
			image.at( p % 1024, p / 1024 ) = { val, val, val };
		}
		p++;
	}
	image.dump( "./a.png" );
}

}  // namespace fallback

}  // namespace koishi