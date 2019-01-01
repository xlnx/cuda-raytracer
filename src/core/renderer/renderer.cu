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
		KLOG( "Target resolution:", w, "x", h );

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
		Profiler profiler( config.profiler, w, h, spp );
		profiler

		  KLOG( "Start intergrating" );
		tracer->execute( image, lens, rng_gen, scene, spp, profiler );
		KLOG( "Finished intergrating" );

		KINFO( renderer, "Writting image to file" );
		image.dump( dest );
		KINFO( renderer, "Written to '" + dest + "'" );
	}
	KINFO( renderer, "Render finished successfully" );
}

}  // namespace core

}  // namespace koishi