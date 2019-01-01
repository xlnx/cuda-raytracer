#include <cstdio>
#include <core/kernel/radiance.hpp>
#include <core/kernel/normal.hpp>
#include <core/kernel/custom.hpp>
// #include <core/kernel/bruteForce.hpp>
#include <core/tracer/cpuMulticore.hpp>
//#include <core/tracer/cudaSingleGPU.hpp>
// #include <core/renderer/factory.hpp>
#include <core/renderer/renderer.hpp>
#include <vis/renderer.hpp>
#include <cxxopts/cxxopts.hpp>

using namespace koishi;
using namespace core;

int main( int argc, char **argv )
{
	cxxopts::Options options( "cr", "Ray tracer for heterogeneous systems, by KoishiChan~" );
	options.add_options()(
	  "v,visualize", "Visualize BVH using openGL." )(
	  "o", "Place the output into <file>.", cxxopts::value<std::string>()->default_value( "a.png" ) )(
	  "s,sample-per-pixel", "Number of sample points per pixel.", cxxopts::value<uint>()->default_value( "1" ) )(
	  "l,list", "List all valid renderers." )(
	  "h,help", "Show help message." )(
	  "t,tracer", "Specify target tracer.", cxxopts::value<std::string>()->default_value( "CPUMultiCoreTracer" ) )(
	  "k,kernel", "Specify kernel function.", cxxopts::value<std::string>()->default_value( "Radiance" ) )(
	  "a,allocator", "Specify allocator.", cxxopts::value<std::string>()->default_value( "HybridAllocator" ) )(
	  "resolution", "Specify target resolution.", cxxopts::value<std::string>()->default_value( "1024x768" ) );

	try
	{
		auto opt = options.parse( argc, argv );

		if ( opt.count( "h" ) )
		{
			KLOG( options.help() );
		}
		else if ( opt.count( "v" ) )
		{
#ifdef KOISHI_USE_GL
			vis::Renderer r{ 1024, 768 };
			r.render( argv[ 1 ] );
#endif
		}
		else
		{
			if ( opt.count( "l" ) )
			{
				// for ( auto &e : factory.getValidTypes() )
				// {
				// 	KLOG( e );
				// }
			}
			else
			{
				uint w = 1024, h = 768;
				auto resolution = opt[ "resolution" ].as<std::string>();
				sscanf( resolution.c_str(), "%ux%u", &w, &h );

				Renderer renderer( w, h );

				auto spp = opt[ "s" ].as<uint>();
				auto out = opt[ "o" ].as<std::string>();

				KLOG( "Sample", spp, "points per pixel" );

				renderer.render( argv[ 1 ], out, spp );
			}
		}
	}
	catch ( const std::exception &err )
	{
		KINFO( fatal, err.what() );
	}
}
