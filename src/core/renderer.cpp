#include <fstream>
#include <cstdlib>
#include <ctime>
#include "renderer.hpp"
#include <util/config.hpp>
#include <util/mesh.hpp>
#include <util/image.hpp>
#include "sampler.hpp"

namespace koishi
{
namespace core
{
Renderer::Renderer( uint w, uint h ) :
  buffer( w * h ), w( w ), h( h )
{
}

static unsigned long long seed = 1;

static inline double drand48()
{
	constexpr auto m = 0x100000000LL;
	constexpr auto c = 0xB16;
	constexpr auto a = 0x5DEECE66DLL;
	seed = ( a * seed + c ) & 0xFFFFFFFFFFFFLL;
	unsigned int x = seed >> 16;
	return ( (double)x / (double)m );
}

static inline void srand48( uint i )
{
	seed = ( ( (long long int)i ) << 16 ) | rand();
}

static inline double3 radiance( const util::Ray &r, const std::vector<util::SubMesh> &mesh, uint depth = 0 )
{
	const util::SubMesh *pm;
	util::Hit hit;
	for ( auto &m : mesh )
	{
		util::Hit hit1;
		if ( r.intersect( m, 1, hit1 ) && hit1.t < hit.t )
		{
			hit = hit1;
			pm = &m;
		}
	}
	if ( hit )
	{
		auto n = util::interplot( pm->normals[ pm->indices[ hit.id ] ],
								  pm->normals[ pm->indices[ hit.id + 1 ] ],
								  pm->normals[ pm->indices[ hit.id + 2 ] ],
								  hit.uv );
		// if ( dot( n, r.v ) > 0 ) n = -n;
		auto nr = r.reflect( hit.t, n );
		// return normalize( nr.v );
		auto r1 = 2 * M_PI * drand48(), r2 = drand48(), r2s = sqrt( r2 );
		auto u = normalize( cross( abs( nr.v.x ) > abs( nr.v.y ) ? double3{ 0, 1, 0 } : double3{ 1, 0, 0 }, nr.v ) );
		auto v = cross( nr.v, u );
		auto dr = nr;
		dr.v = normalize( ( u * cos( r1 ) + v * sin( r1 ) ) * r2s + nr.v * sqrt( 1 - r2s ) );
		// return nr.v;  //( r.o + r.v * hit.t ) / 100.;
		if ( depth < 100 )
			if ( drand48() < .9 )
				return pm->emissive +
					   pm->color *
						 radiance( dr, mesh, depth + 1 );
			else
				return pm->emissive +
					   pm->color *
						 radiance( nr, mesh, depth + 1 );
		// return  //pm->emissive +
		// 		//pm->color *
		//   radiance( nr, mesh, depth + 1 );
		else
			// return ( r.o + r.v * hit.t ) / 100.;
			return pm->emissive;
		// radiance( nr, mesh );
	}
	else
	{
		return double3{ 0.f, 0.f, 0.f };
	}
}

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

	srand48( time( nullptr ) );

	rays = core::Sampler( w, h ).sample( camera, spp );

	util::Image<3> image( w, h );
	for ( uint j = 0; j != h; ++j )
	{
		for ( uint i = 0; i != w; ++i )
		{
			double3 rad = { 0, 0, 0 };
			for ( uint k = 0; k != spp; ++k )
			{
				rad += radiance( rays[ ( j * w + i ) * spp + k ], mesh );
			}
			image.at( i, j ) = rad / spp;
		}
	}
	image.dump( dest );
}

}  // namespace core

}  // namespace koishi