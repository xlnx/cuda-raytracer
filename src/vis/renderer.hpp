#pragma once

#include <vis/util.hpp>

namespace koishi
{
namespace vis
{
class Renderer final
{
public:
	Renderer( uint w, uint h );
	~Renderer();

	void render( const std::string &path );

private:
	GLFWwindow *window;
	uint w, h;
};

}  // namespace vis

}  // namespace koishi