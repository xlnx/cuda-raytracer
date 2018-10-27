#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <vec/vec.hpp>
#include <vec/vsel.hpp>

namespace koishi
{
namespace sn
{
using namespace nlohmann;

struct Camera
{
	float fovy;
	float2 position;
};

inline void to_json( json &j, const Camera &c )
{
	j = json{
		{ "fovy", c.fovy },
		{ "position", c.position }
	};
}

inline void from_json( const json &j, Camera &c )
{
	j.at( "fovy" ).get_to( c.fovy );
	j.at( "position" ).get_to( c.position );
}

struct Scene
{
public:
	Scene( const std::string &path )
	{
		json data;
		std::ifstream is( path );
		is >> data;
		camera = data[ "camera" ].get<decltype( camera )>();
		std::cout << data.is_object() << " " << data.is_array() << " " << data.is_null() << std::endl;
	}

	std::vector<Camera> camera;
};

}  // namespace sn

using sn::Camera;
using sn::Scene;

}  // namespace koishi
