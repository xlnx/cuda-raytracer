#pragma once

#include <exception>
#include <string>

namespace koishi

{
namespace util
{
struct Exception : std::exception
{
	Exception( const std::string &info ) {}
};

}  // namespace util

}  // namespace koishi