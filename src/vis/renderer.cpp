#include <fstream>
#include "renderer.hpp"
#include "camera.hpp"

namespace koishi
{
namespace vis
{
Renderer::Renderer( uint w, uint h ) :
  w( w ), h( h )
{
	if ( !glfwInit() )
	{
		throw util::Exception( "Failed to init glfw." );
	}
	glfwWindowHint( GLFW_VERSION_MAJOR, 3 );
	glfwWindowHint( GLFW_VERSION_MINOR, 3 );
	glfwWindowHint( GLFW_SAMPLES, 4 );
	glfwWindowHint( GLFW_RESIZABLE, false );
	window = glfwCreateWindow( w, h, "Scene Preview", nullptr, nullptr );
	glfwMakeContextCurrent( window );
	if ( !gladLoadGLLoader( (GLADloadproc)glfwGetProcAddress ) )
	{
		throw util::Exception( "Failed to get GL proc address." );
	}
	glEnable( GL_DEPTH_TEST );
	glClearColor( 0, 0, 0, 0 );
}

Renderer::~Renderer()
{
	glfwTerminate();
}

void Renderer::render( const std::string &path )
{
	jsel::Scene scene;
	std::ifstream( path ) >> scene;
	if ( !scene.camera.size() )
	{
		throw util::Exception( "No valid camera in this scene." );
	}
	Camera camera( w, h, scene.camera[ 0 ] );
	// for (auto )
	// Scene scene;

	while ( !glfwWindowShouldClose( window ) )
	{
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

		static auto prev = glfwGetTime();
		auto curr = glfwGetTime();
		auto detMillis = curr - prev;

		// for ( auto &object : scene )
		// {
		// 	object.render();
		// }

		glfwPollEvents();
		glfwSwapBuffers( window );
	}
}

}  // namespace vis

}  // namespace koishi