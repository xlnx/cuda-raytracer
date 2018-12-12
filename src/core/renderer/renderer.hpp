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
	  w( w ), h( h )
	{
	}

	void render( const std::string &path, const std::string &dest, uint spp ) override
	{
		KINFO( renderer, "Render start" );
		if ( core::Scene scene = path )
		{
			if ( !scene.camera.size() )
			{
				KTHROW( "no camera in the scene" );
			}
			auto &camera = scene.camera[ 0 ];

			util::Image<3> image( w, h );
			KLOG( "Target resolution:", w, "x", h );

			Lens lens( camera, w, h, spp );
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
