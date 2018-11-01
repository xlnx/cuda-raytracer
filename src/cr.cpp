#include <sstream>
#include <fallback/renderer.hpp>
#include <vis/renderer.hpp>

using namespace koishi;

int main( int argc, char **argv )
{
	if ( argc < 4 )
	{
		std::cout << "Not enough parameters." << std::endl;
		return 1;
	}
	uint spp;
	std::istringstream is( argv[ 3 ] );
	is >> spp;

	// vis::Renderer r{ 1024, 768 };

	// r.render( argv[ 1 ] );
	fallback::Renderer r{ 1024, 768 };

	r.render( argv[ 1 ], argv[ 2 ], spp );
	// vis::Renderer r{ 1024, 768 };

	// r.render( "./cow.json" );
}