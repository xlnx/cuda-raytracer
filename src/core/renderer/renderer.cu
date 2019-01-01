#include "renderer.hpp"

namespace koishi
{
namespace core
{
Renderer::Renderer( uint w, uint h ) :
  w( w ),
  h( h )
{
}

void Renderer::render( const std::string &path, const std::string &dest, uint spp )
{
	KINFO( renderer, "Render start" );
	KINFO( renderer, "Loading config '" + path + "'" );
	Configuration config;
	std::ifstream( path ) >> config;
	if ( core::Scene scene = config.scene )
	{
		if ( !scene.camera.size() )
		{
			KTHROW( "no camera in the scene" );
		}
		auto &camera = scene.camera[ 0 ];

		util::Image<3> image( w, h );
		KLOG( "Target resolution:", w, "x", h, ",", spp, "spp" );

		poly::object<Lens> lens;
		switch ( camera.lens )
		{
		case 0:
			lens = poly::make_object<PinholeLens>( camera, w, h, spp );
			break;
		case 1:
			lens = poly::make_object<OrthographicLens>( camera, w, h, spp );
		}
		SamplerGenerator rng_gen;
		auto tracer = Factory<Tracer>::create( config.tracer );
		Profiler profiler( config.profiler, tracer->getKernel(), w, h, spp );

		KLOG( "Start intergrating" );
		tracer->execute( image, lens, rng_gen, scene, spp, profiler );
		KLOG( "Finished intergrating" );

		KINFO( renderer, "Writting image to file" );
		image.dump( dest );
		KINFO( renderer, "Written to '" + dest + "'" );

		if ( profiler.enabled() )
		{
			auto area = profiler.getArea();
			for ( int j = area.y; j != area.w; ++j )
			{
				image.at( area.x, j ) = image.at( area.z, j ) = float3{ 1, 0, 0 };
			}
			for ( int i = area.x; i != area.z; ++i )
			{
				image.at( i, area.y ) = image.at( i, area.w ) = float3{ 1, 0, 0 };
			}
			KINFO( renderer, "Writting image to file" );
			image.dump( dest + "prof.png" );
			KINFO( renderer, "Written to '" + dest + ".prof.png'" );
		}
	}
	KINFO( renderer, "Render finished successfully" );
}

}  // namespace core

}  // namespace koishi