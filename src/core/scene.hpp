#pragma once

#include <string>
#include <vector>
#include <core/mesh.hpp>
#include <util/config.hpp>

namespace koishi
{
namespace core
{
struct Scene
{
	Scene( const std::string &path );

	std::vector<core::Mesh> mesh;
	std::vector<std::string> material;
	std::vector<jsel::Camera> camera;
};

}  // namespace core

}  // namespace koishi
