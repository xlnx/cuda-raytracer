#pragma once

#include <string>
#include <vector>
#include <vec/vmath.hpp>
#include <vec/vios.hpp>
#include <util/ray.hpp>

namespace koishi
{
namespace fallback
{
class Renderer
{
public:
	Renderer( uint w, uint h );

	void render( const std::string &path, const std::string &dest, uint spp );

private:
	using uchar = unsigned char;
	std::vector<double3> buffer;
	std::vector<util::Ray> rays;
	uint w, h;
};

}  // namespace fallback

}  // namespace koishi