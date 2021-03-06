#if defined( KOISHI_USE_GL )

#include <fstream>
#include <utility>
#include <core/meta/scene.hpp>
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
		KTHROW( "failed to init glfw." );
	}
	glfwWindowHint( GLFW_VERSION_MAJOR, 3 );
	glfwWindowHint( GLFW_VERSION_MINOR, 3 );
	glfwWindowHint( GLFW_SAMPLES, 4 );
	glfwWindowHint( GLFW_RESIZABLE, false );
	window = glfwCreateWindow( w, h, "Scene Preview", nullptr, nullptr );
	glfwMakeContextCurrent( window );
	if ( !gladLoadGLLoader( (GLADloadproc)glfwGetProcAddress ) )
	{
		KTHROW( "failed to get GL proc address." );
	}
	glEnable( GL_DEPTH_TEST );
	glClearColor( 0, 0, 0, 0 );
}

Renderer::~Renderer()
{
	glfwTerminate();
}

static GLuint compileShader()
{
	auto vs = glCreateShader( GL_VERTEX_SHADER );
	auto fs = glCreateShader( GL_FRAGMENT_SHADER );
	const char *vsrc[] = {
#include "vis.vert"
	};
	const char *fsrc[] = {
#include "vis.frag"
	};
	glShaderSource( vs, 1, vsrc, nullptr );
	glShaderSource( fs, 1, fsrc, nullptr );
	GLint success;
	glCompileShader( vs );
	glGetShaderiv( vs, GL_COMPILE_STATUS, &success );
	assert( success );
	glCompileShader( fs );
	glGetShaderiv( fs, GL_COMPILE_STATUS, &success );
	assert( success );

	auto prog = glCreateProgram();
	glAttachShader( prog, vs );
	glAttachShader( prog, fs );
	glLinkProgram( prog );
	glGetProgramiv( prog, GL_LINK_STATUS, &success );
	assert( success );

	return prog;
}

struct Mesh
{
	GLuint vao;
	core::BVHTree bvh;
};

void Renderer::render( const std::string &path )
{
	if ( core::Scene scene = path )
	{
		if ( !scene.camera.size() )
		{
			KTHROW( "no camera in the scene." );
		}
		Camera camera( w, h, scene.camera[ 0 ] );
		core::poly::vector<Mesh> mesh;
		// static_assert( std::is_same<core::poly::vector<core::Mesh> &, decltype( ( scene.mesh ) )>::value );
		// static_assert( std::is_same<const core::BVHTree &&, decltype( std::move( e.bvh ) )>::value );
		for ( core::Mesh &e : scene.mesh )
		{
			GLuint vao, vbo, ebo;
			glGenVertexArrays( 1, &vao );
			glGenBuffers( 1, &vbo );
			glGenBuffers( 1, &ebo );
			glBindVertexArray( vao );
			glBindBuffer( GL_ARRAY_BUFFER, vbo );
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, ebo );
			glBufferData( GL_ARRAY_BUFFER, e.vertices.size() * sizeof( e.vertices[ 0 ] ), &e.vertices[ 0 ], GL_STATIC_DRAW );
			glBufferData( GL_ELEMENT_ARRAY_BUFFER, e.indices.size() * sizeof( e.indices[ 0 ] ), &e.indices[ 0 ], GL_STATIC_DRAW );
			glEnableVertexAttribArray( 0 );
			glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, sizeof( float3 ), (const void *)( 0 ) );
			glBindVertexArray( 0 );
			glBindBuffer( GL_ARRAY_BUFFER, 0 );
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
			// static_assert( std::is_same<core::BVHTree, decltype( e.bvh )>::value );
			// static_assert( std::is_same<const core::BVHTree &&, decltype( std::move( e.bvh ) )>::value );
			Mesh m;
			m.vao = vao;
			m.bvh = std::move( e.bvh );
			mesh.emplace_back( std::move( m ) );
		}

		auto prog = compileShader();
		glUseProgram( prog );

		while ( !glfwWindowShouldClose( window ) )
		{
			glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

			static auto prev = glfwGetTime();
			auto curr = glfwGetTime();
			auto detMillis = curr - prev;

			auto mat = camera.getTrans();
			glUniformMatrix4fv( glGetUniformLocation( prog, "wvp" ), 1, GL_FALSE,
								reinterpret_cast<const float *>( &mat ) );

			auto k = 1;
			for ( auto &m : mesh )
			{
				glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
				glUniform1i( glGetUniformLocation( prog, "mode" ), 0 );
				glBindVertexArray( m.vao );
				glDrawElements( GL_TRIANGLES, m.bvh[ 1 ].end - m.bvh[ 1 ].begin, GL_UNSIGNED_INT, nullptr );
				glBindVertexArray( 0 );

				glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
				glUniform1i( glGetUniformLocation( prog, "mode" ), 1 );
				glBindVertexArray( m.vao );
				glDrawElements( GL_TRIANGLES, m.bvh[ k ].end - m.bvh[ k ].begin,
								GL_UNSIGNED_INT, (uint *)nullptr + m.bvh[ k ].begin );
				glBindVertexArray( 0 );
			}

			glfwPollEvents();
			glfwSwapBuffers( window );
		}
	}
}

}  // namespace vis

}  // namespace koishi

#endif
