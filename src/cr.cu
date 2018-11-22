#include <core/misc/random.hpp>
#include <core/kernel/radiance.hpp>
#include <core/kernel/normal.hpp>
#include <core/tracer/cpuMulticore.hpp>
#include <core/tracer/cudaSingleGPU.hpp>
#include <core/renderer/factory.hpp>
#include <vis/renderer.hpp>
#include <cxxopts/cxxopts.hpp>

using namespace koishi;
using namespace core;

int main( int argc, char **argv )
{
	cxxopts::Options options( "cr", "cuda-raytracer" );
	options.add_options()(
	  "v,visualize", "Visualize BVH using openGL." )(
	  "o", "Place the output into <file>.", cxxopts::value<std::string>() )(
	  "s,sample-per-pixel", "Number of sample points per pixel.", cxxopts::value<uint>() )(
	  "l,list", "List all valid renderers." )(
	  "h,help", "Show help message." )(
	  "t,tracer", "Specify target tracer, default 'CPUMultiCoreTracer'.", cxxopts::value<std::string>() )(
	  "k,kernel", "Specify kernel function, default 'Radiance'.", cxxopts::value<std::string>() )(
	  "r,random-number-generator", "Specify random number generator, default 'DRand48'.", cxxopts::value<std::string>() )(
	  "a,allocator", "Specify allocator, default 'HybridAllocator'.", cxxopts::value<std::string>() );

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
			// clang-format off
			Factory<
			  templates2<CPUMultiCoreTracer
#ifdef KOISHI_USE_CUDA
						  , CudaSingleGPUTracer
#endif
						>,
			  types<
				templates1<Radiance, Normal>,
				types<FakeRand, DRand48>
			  >,
			  types<
				HybridAllocator,
				HostAllocator
			  >
			>
			  factory;
			// clang-format on

			if ( opt.count( "l" ) )
			{
				for ( auto &e : factory.getValidTypes() )
				{
					KLOG( e );
				}
			}
			else
			{
				std::string tracer = opt.count( "t" ) ? opt[ "t" ].as<std::string>() : "CPUMultiCoreTracer";
				std::string kernel = opt.count( "k" ) ? opt[ "k" ].as<std::string>() : "Radiance";
				std::string rng = opt.count( "r" ) ? opt[ "r" ].as<std::string>() : "DRand48";
				std::string alloc = opt.count( "a" ) ? opt[ "a" ].as<std::string>() : "HybridAllocator";

				auto targetClass = tracer + "<" + kernel + "<" + rng + ">" + ", " + alloc + ">";

				auto r = factory.create( targetClass, 1024, 768 );

				auto spp = opt[ "s" ].as<uint>();

				std::string out = opt.count( "o" ) ? opt[ "o" ].as<std::string>() : "a.png";

				KLOG( "Sample", spp, "points per pixel" );
				KLOG( "Using renderer:", targetClass );

				KLOG( "=== Start ===" );
				r->render( argv[ 1 ], out, spp );
				KLOG( "=== Finished ===" );

				KLOG( "Written image to '" + out + "'" );
			}
		}
	}
	catch ( const std::exception &err )
	{
		KLOG( "[ fatal ] ", err.what() );
	}
}
