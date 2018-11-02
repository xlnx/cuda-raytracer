#pragma once

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

}  // namespace core

}  // namespace koishi