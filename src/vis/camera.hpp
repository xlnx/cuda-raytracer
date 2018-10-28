#pragma once

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
		N = glm::vec3{ config.target.x, config.target.y, config.target.z };
		Up = glm::vec3{ 0, 1, 0 };
		const auto zNear = 1e-3f, zFar = 1e5f;
		persTrans = glm::perspective( config.fovy,
									  float( w ) / float( h ), zNear, zFar );
	}
	~Camera() = default;

public:
	void setTarget( float x, float y, float z )
	{
		N = glm::normalize( glm::vec3{ x, y, z } );
	}
	void setUp( float x, float y, float z )
	{
		Up = glm::normalize( glm::vec3{ x, y, z } );
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
		// P += glm::vec3{ dx, dy, dz };
	}
	void rotate( float au, float av, float an )
	{
		auto U = glm::normalize( glm::cross( N, Up ) );
		auto V = glm::cross( U, N );
		N = glm::rotate( N, au, Up );
		N = glm::rotate( N, av, U );
		Up = glm::rotate( Up, an, N );
	}
	glm::mat4 getTrans() const
	{
		auto U = glm::normalize( glm::cross( N, Up ) );
		auto V = glm::cross( U, N );
		glm::mat4 rotTrans = {
			U.x, V.x, -N.x, 0.f,
			U.y, V.y, -N.y, 0.f,
			U.z, V.z, -N.z, 0.f,
			0.f, 0.f, 0.f, 1.f
		};
		glm::mat4 trTrans = {
			1.f, 0.f, 0.f, 0.f,
			0.f, 1.f, 0.f, 0.f,
			0.f, 0.f, 1.f, 0.f,
			-P.x, -P.y, -P.z, 1.f
		};
		return persTrans * rotTrans * trTrans;
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