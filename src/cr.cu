#include <sstream>
#include <core/random.hpp>
#include <core/radiance.hpp>
#include <core/renderer.hpp>
#include <core/tracer/cpuMulticore.hpp>
#include <core/tracer/cudaSingleGPU.hpp>
#include <core/factory.hpp>
#include <vis/renderer.hpp>

using namespace koishi;
using namespace core;

int main( int argc, char **argv )
{
	try
	{
		if ( std::string( argv[ 2 ] ) == "-v" )
		{
#ifdef KOISHI_USE_GL
			vis::Renderer r{ 1024, 768 };
			r.render( argv[ 1 ] );
#endif
		}
		else
		{
			uint spp;
			std::istringstream is( argv[ 3 ] );
			is >> spp;

			Factory<
			  templates<CPUMultiCoreTracer
#ifdef KOISHI_USE_CUDA
						  CudaSingleGPUTracer
#endif
						>,
			  templates<Radiance>,
			  types<FakeRand, DRand48>>
			  factory;

			Factory<
			  templates<CPUMultiCoreTracer
#ifdef KOISHI_USE_CUDA
						  CudaSingleGPUTracer
#endif
						>,
			  types<
				templates<Radiance>,
				types<FakeRand, DRand48>>>,
			  factory;

			// for ( auto &e : factory.getValidTypes() )
			// {
			// 	KLOG( e );
			// }

			auto r = factory.create( "CPUMultiCoreTracer@@Radiance@@DRand48", 1024, 768 );

			r->render( argv[ 1 ], argv[ 2 ], spp );
		}
	}
	catch ( std::logic_error err )
	{
		KLOG( "[fatal] ", err.what() );
	}
}
