#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <vec/vmath.hpp>
#include <vec/vios.hpp>
#include <util/image.hpp>
#include "scene.hpp"
#include "sampler.hpp"
#include "ray.hpp"
#include "mesh.hpp"
#include "radiance.hpp"
#include "random.hpp"
#include "dev.hpp"

namespace koishi
{
namespace core
{
template <typename Tracer = core::Radiance<core::DRand48>>
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
		core::Scene scene( path );
		if ( !scene.camera.size() )
		{
			throw "no camera in the scene.";
		}
		auto &camera = scene.camera[ 0 ];

		rays = core::Sampler( w, h ).sample( camera, spp );

		util::Image<3> image( w, h );
		Host::call<Tracer>( image, rays, scene.mesh, spp );

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
