#include <sstream>
#include <core/tracer.hpp>
#include <core/random.hpp>
#include <core/radiance.hpp>
#include <core/renderer.hpp>
#include <vis/renderer.hpp>

using namespace koishi;
using namespace core;

int main( int argc, char **argv )
{
	if ( std::string( argv[ 2 ] ) == "-v" )
	{
	}
	else
	{
		uint spp;
		std::istringstream is( argv[ 3 ] );
		is >> spp;

		using TraceFn = Radiance<FakeRand>;
		Renderer<
		  // cuda::
		  Tracer<TraceFn>>
		  r{ 1024, 768 };

		r.render( argv[ 1 ], argv[ 2 ], spp );
	}
}
