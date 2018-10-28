#pragma once

#include <iostream>
#include "trait.hpp"
#if !defined( KOISHI_USE_CUDA )
#include "vmath.hpp"
#endif

namespace koishi
{
namespace vios
{
namespace vec1
{
template <typename T, typename = typename std::enable_if<trait::is_vec1<T>::value>::type>
std::ostream &operator<<( std::ostream &os, const T &v )
{
	return os << "[" << v.x << "]", os;
}
}  // namespace vec1

namespace vec2
{
template <typename T, typename = typename std::enable_if<trait::is_vec2<T>::value>::type>
std::ostream &operator<<( std::ostream &os, const T &v )
{
	return os << "[" << v.x << "," << v.y << "]", os;
}
}  // namespace vec2

namespace vec3
{
template <typename T, typename = typename std::enable_if<trait::is_vec3<T>::value>::type>
std::ostream &operator<<( std::ostream &os, const T &v )
{
	return os << "[" << v.x << "," << v.y << "," << v.z << "]", os;
}
}  // namespace vec3

namespace vec4
{
template <typename T, typename = typename std::enable_if<trait::is_vec4<T>::value>::type>
std::ostream &operator<<( std::ostream &os, const T &v )
{
	return os << "[" << v.x << "," << v.y << "," << v.z << "," << v.w << "]", os;
}
}  // namespace vec4

using namespace vec1;
using namespace vec2;
using namespace vec3;
using namespace vec4;

}  // namespace vios

}  // namespace koishi

using namespace koishi::vios;