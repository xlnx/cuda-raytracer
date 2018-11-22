#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <vec/vmath.hpp>
#include <vec/vios.hpp>
#include <util/image.hpp>
#include <util/debug.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/poly.hpp>
#include <core/basic/allocator.hpp>
#include <core/meta/mesh.hpp>
#include <core/meta/scene.hpp>
#include <core/misc/sampler.hpp>

namespace koishi
{
namespace core
{
struct RendererBase
{
	virtual ~RendererBase() = default;

	virtual void render( const std::string &path, const std::string &dest, uint spp ) = 0;

protected:
	using call_type = Host;
};

template <typename Tracer>
struct Renderer : RendererBase
{
	Renderer( uint w, uint h ) :
	  w( w ), h( h )
	{
	}

	void render( const std::string &path, const std::string &dest, uint spp ) override
	{
		KLOG( "Loading scene config file" );
		if ( core::Scene scene = path )
		{
			if ( !scene.camera.size() )
			{
				KTHROW( No camera in the scene );
			}
			auto &camera = scene.camera[ 0 ];

			rays = core::Sampler( w, h ).sample( camera, spp );

			util::Image<3> image( w, h );
			KLOG( "Target resolution:", w, "x", h );
			
			KLOG( "Start intergrating" );
			Host::call<Tracer>( image, rays, scene, spp );
			KLOG( "Finished intergrating" );

			KLOG( "Writting image to file" );
			image.dump( dest );
		}
	}

private:
	using uchar = unsigned char;
	std::vector<double3> buffer;
	PolyVector<Ray> rays;
	uint w, h;
};

}  // namespace core

}  // namespace koishi
