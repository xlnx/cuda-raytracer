#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <util/image.hpp>
#include <util/config.hpp>
#include <core/basic/basic.hpp>
#include <core/meta/scene.hpp>
#include <core/misc/lens.hpp>
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
	  w( w ),
	  h( h )
	{
	}

	void render( const std::string &path, const std::string &dest, uint spp ) override
	{
		KINFO( renderer, "Render start" );
		KINFO( renderer, "Loading config '" + path + "'" );
		RendererConfig config;
		std::ifstream( path ) >> config;
		if ( core::Scene scene = config.scene )
		{
			if ( !scene.camera.size() )
			{
				KTHROW( "no camera in the scene" );
			}
			auto &camera = scene.camera[ 0 ];

			util::Image<3> image( w, h );
			KLOG( "Target resolution:", w, "x", h );

			poly::object<Lens> lens;
			if ( camera.lens == "pinhole" )
			{
				lens = poly::make_object<PinholeLens>( camera, w, h, spp );
			}
			else if ( camera.lens == "orthographic" )
			{
				lens = poly::make_object<OrthographicLens>( camera, w, h, spp );
			}
			else
			{
				KTHROW( "invalid lens type: " + camera.lens );
			}
			SamplerGenerator rng_gen;

			KLOG( "Start intergrating" );
			Host::call<Tracer>( image, lens, rng_gen, scene, spp );
			KLOG( "Finished intergrating" );

			KINFO( renderer, "Writting image to file" );
			image.dump( dest );
			KINFO( renderer, "Written to '" + dest + "'" );
		}
		KINFO( renderer, "Render finished successfully" );
	}

private:
	using uchar = unsigned char;
	std::vector<float3> buffer;
	poly::vector<Ray> rays;
	uint w, h;
};

}  // namespace core

}  // namespace koishi
