#pragma once

#include "util.hpp"

namespace koishi
{
namespace vis
{
namespace __gl
{
struct Buffer
{
	operator GLuint() const
	{
		return handle;
	}

protected:
	GLuint handle;
};
}  // namespace __gl

struct VBO : __gl::Buffer
{
	VBO()
	{
		glGenBuffers( 1, &handle );
	}
	void bind() const
	{
		glBindBuffer( GL_ARRAY_BUFFER, handle );
	}
	void unbind() const
	{
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
	}
	void data( GLsizeiptr size, const void *ptr ) const
	{
		glBufferData( GL_ARRAY_BUFFER, size, ptr, GL_STATIC_DRAW );
	}
};

struct VAO : __gl::Buffer
{
	VAO()
	{
		glGenVertexArrays( 1, &handle );
	}
	void bind() const
	{
		glBindVertexArray( handle );
	}
	void unbind() const
	{
		glBindVertexArray( 0 );
	}
};

struct EBO : __gl::Buffer
{
	EBO()
	{
		glGenBuffers( 1, &handle );
	}
	void bind() const
	{
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, handle );
	}
	void unbind() const
	{
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
	}
	void data( GLsizeiptr size, const void *ptr ) const
	{
		glBufferData( GL_ELEMENT_ARRAY_BUFFER, size, ptr, GL_STATIC_DRAW );
	}
};

}  // namespace vis

}  // namespace koishi