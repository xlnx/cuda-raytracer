#pragma once

#include <json/json.h>
#include "trait.hpp"
#include "vec.hpp"

namespace koishi
{
namespace vec
{
template <typename T, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value &&
														 trait::is_vec1<T>::value>::type>
void to_json( Json::Value &j, const T &v )
{
	j = Json::arrayValue;
	j[ 0 ] = v.x;
}

template <typename T, typename = void, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value && trait::is_vec2<T>::value>::type>
void to_json( Json::Value &j, const T &v )
{
	j = Json::arrayValue;
	j[ 0 ] = v.x, j[ 1 ] = v.y;
}

template <typename T, typename = void, typename = void, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value && trait::is_vec3<T>::value>::type>
void to_json( Json::Value &j, const T &v )
{
	j = Json::arrayValue;
	j[ 0 ] = v.x, j[ 1 ] = v.y, j[ 2 ] = v.z;
}

template <typename T, typename = void, typename = void, typename = void, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value && trait::is_vec4<T>::value>::type>
void to_json( Json::Value &j, const T &v )
{
	j = Json::arrayValue;
	j[ 0 ] = v.x, j[ 1 ] = v.y, j[ 2 ] = v.z, j[ 3 ] = v.w;
}

template <typename T, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value &&
														 trait::is_vec1<T>::value>::type>
void from_json( const Json::Value &j, T &v )
{
	v.x = typename koishi::trait::com<T>::type( j[ 0 ].asDouble() );
}

template <typename T, typename = void, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value && trait::is_vec2<T>::value>::type>
void from_json( const Json::Value &j, T &v )
{
	v.x = typename koishi::trait::com<T>::type( j[ 0 ].asDouble() );
	v.y = typename koishi::trait::com<T>::type( j[ 1 ].asDouble() );
}

template <typename T, typename = void, typename = void, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value && trait::is_vec3<T>::value>::type>
void from_json( const Json::Value &j, T &v )
{
	v.x = typename koishi::trait::com<T>::type( j[ 0 ].asDouble() );
	v.y = typename koishi::trait::com<T>::type( j[ 1 ].asDouble() );
	v.z = typename koishi::trait::com<T>::type( j[ 2 ].asDouble() );
}

template <typename T, typename = void, typename = void, typename = void, typename = typename std::enable_if<trait::is_in<T, KOISHI_VEC>::value && trait::is_vec4<T>::value>::type>
void from_json( const Json::Value &j, T &v )
{
	v.x = typename koishi::trait::com<T>::type( j[ 0 ].asDouble() );
	v.y = typename koishi::trait::com<T>::type( j[ 1 ].asDouble() );
	v.z = typename koishi::trait::com<T>::type( j[ 2 ].asDouble() );
	v.w = typename koishi::trait::com<T>::type( j[ 3 ].asDouble() );
}

}  // namespace vec

}  // namespace koishi
