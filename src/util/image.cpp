#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <string>
#include <vec/vec.hpp>

namespace koishi
{
namespace util
{
namespace __impl
{
int write_image( const std::string &path, uint w, uint h, uint channel, const unsigned char *data )
{
	return stbi_write_png( path.c_str(), w, h, channel, data, 0 );
}

}  // namespace __impl

}  // namespace util

}  // namespace koishi