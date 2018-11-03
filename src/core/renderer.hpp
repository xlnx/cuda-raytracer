#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <vec/vmath.hpp>
#include <vec/vios.hpp>
#include <util/jsel.hpp>
#include <util/image.hpp>
#include "sampler.hpp"
#include "ray.hpp"
#include "mesh.hpp"
#include "radiance.hpp"
#include "dev.hpp"

namespace koishi
{
namespace core
{
template <template <typename T> class Tracer, typename Random, template <typename T> class Radiance = core::Radiance>
class Renderer
{
	using call_type = Host;

public:
	Renderer( uint w, uint h ) :
	  w( w ), h( h )
	{
	}

	void render( const std::string &path, const std::string &dest, uint spp )
	{
		jsel::Scene scene;
		std::ifstream is( path );
		is >> scene;
		if ( !scene.camera.size() )
		{
			throw "no camera in the scene.";
		}
		auto &camera = scene.camera[ 0 ];
		std::vector<core::SubMesh> mesh;
		for ( auto &m : scene.mesh )
		{
			for ( auto &e : core::PolyMesh( m ).mesh )
			{
				mesh.emplace_back( std::move( e ) );
			}
		}

		rays = core::Sampler( w, h ).sample( camera, spp );

		util::Image<3> image( w, h );
		call<Tracer<Radiance<Random>>>( image, rays, mesh, spp );

		image.dump( dest );
	}

private:
	using uchar = unsigned char;
	std::vector<double3> buffer;
	std::vector<core::Ray> rays;
	uint w, h;
};

}  // namespace core

}  // namespace koishi