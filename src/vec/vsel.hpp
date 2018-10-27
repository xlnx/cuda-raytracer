#pragma once

#include <nlohmann/json.hpp>
#include "trait.hpp"
#include "vec.hpp"

namespace koishi
{
namespace vec
{
template <typename T, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value &&
														 trait::is_vec1<T>::value>::type>
void to_json( nlohmann::json &j, const T &v )
{
	j = nlohmann::json{ v.x };
}

template <typename T, typename = void, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value && trait::is_vec2<T>::value>::type>
void to_json( nlohmann::json &j, const T &v )
{
	j = nlohmann::json{ v.x, v.y };
}

template <typename T, typename = void, typename = void, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value && trait::is_vec3<T>::value>::type>
void to_json( nlohmann::json &j, const T &v )
{
	j = nlohmann::json{ v.x, v.y, v.z };
}

template <typename T, typename = void, typename = void, typename = void, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value && trait::is_vec4<T>::value>::type>
void to_json( nlohmann::json &j, const T &v )
{
	j = nlohmann::json{ v.x, v.y, v.z, v.w };
}

template <typename T, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value &&
														 trait::is_vec1<T>::value>::type>
void from_json( const nlohmann::json &j, T &v )
{
	j[ 0 ].get_to( v.x );
}

template <typename T, typename = void, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value && trait::is_vec2<T>::value>::type>
void from_json( const nlohmann::json &j, T &v )
{
	j[ 0 ].get_to( v.x );
	j[ 1 ].get_to( v.y );
}

template <typename T, typename = void, typename = void, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value && trait::is_vec3<T>::value>::type>
void from_json( const nlohmann::json &j, T &v )
{
	j[ 0 ].get_to( v.x );
	j[ 1 ].get_to( v.y );
	j[ 2 ].get_to( v.z );
}

template <typename T, typename = void, typename = void, typename = void, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value && trait::is_vec4<T>::value>::type>
void from_json( const nlohmann::json &j, T &v )
{
	j[ 0 ].get_to( v.x );
	j[ 1 ].get_to( v.y );
	j[ 2 ].get_to( v.z );
	j[ 3 ].get_to( v.w );
}

}  // namespace vec

}  // namespace koishi
