#include <fallback/renderer.hpp>
#include <vis/renderer.hpp>

using namespace koishi;

int main( int argc, char **argv )
{
	// fallback::Renderer r{ 1024, 768 };

	// r.render( "./cornell_box.json", 4 );
	fallback::Renderer r{ 1024, 768 };

	r.render( "./cow.json", "./a.png", 1 );
	// vis::Renderer r{ 1024, 768 };

	// r.render( "./cow.json" );
}