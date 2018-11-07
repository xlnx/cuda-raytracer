#pragma once

#if defined( KOISHI_USE_GL )

#include "util.hpp"

namespace koishi
{
namespace vis
{
// perspective camera
class Camera
{
public:
	Camera( uint w, uint h, const jsel::Camera &config )
	{
		glViewport( 0, 0, w, h );
		P = glm::vec3{ config.position.x, config.position.y, config.position.z };
		N = glm::normalize( glm::vec3{ config.target.x, config.target.y, config.target.z } );
		Up = glm::normalize( glm::vec3{ config.upaxis.x, config.upaxis.y, config.upaxis.z } );
		auto as = float( w ) / float( h );
		auto fovy = atan( tan( glm::radians( config.fovx ) * .5 ) / as ) * 2.;
		persTrans = glm::perspective( float( fovy ), as, float( config.zNear ), float( config.zFar ) );
	}
	~Camera() = default;

public:
	void setTarget( float x, float y, float z )
	{
		N = glm::normalize( glm::vec3{ x, y, z } );
	}
	void setPosition( float x, float y, float z )
	{
		P = glm::vec3{ x, y, z };
	}
	void translate( float dx, float dy, float dz )
	{
		auto U = glm::normalize( glm::cross( N, Up ) );
		auto V = glm::normalize( glm::cross( U, N ) );
		P += dx * N + dy * U + dz * V;
	}
	void rotate( float au, float av, float an )
	{
		auto U = glm::normalize( glm::cross( N, Up ) );
		// auto V = glm::cross( U, N );
		N = glm::rotate( N, float( au ), Up );
		N = glm::rotate( N, float( av ), U );
	}
	glm::mat4 getTrans() const
	{
		return persTrans * glm::lookAt( P, P + N, Up );
	}
	glm::vec3 getPosition() const
	{
		return P;
	}

private:
	glm::vec3 N, Up, P;
	glm::mat4 persTrans;
};

}  // namespace vis

}  // namespace koishi

#endif
