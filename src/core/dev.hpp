#pragma once

#include <type_traits>

#if defined( KOISHI_USE_CUDA )
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

namespace koishi
{
namespace core
{
#if defined( KOISHI_USE_CUDA )

namespace dev
{
template <typename T>
using vector = thrust::device_vector<T>;

}

namespace host
{
template <typename T>
using vector = thrust::host_vector<T>;

}

#endif

struct Host
{
};

struct Device
{
};

namespace trait
{

template <typename T, typename = void>
struct is_host_callable: std::integral_constant<bool, false>
{
};
template <typename T, typename = typename std::enable_if<
	std::is_base_of<Host, T>::value>::type>
struct is_host_callable: std::integral_constant<bool, true>
{
};

template <typename T, typename = void>
struct is_device_callable: std::integral_constant<bool, false>
{
};
template <typename T, typename = typename std::enable_if<
        std::is_base_of<Device, T>::value>::type>
struct is_device_callable: std::integral_constant<bool, true>
{
};

}

}  // namespace core

}  // namespace koishi
