#pragma once

#include <type_traits>
#include <vec/trait.hpp>
#include "mesh.hpp"

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
struct dummy
{
};

template <typename T, typename = void>
struct is_host_callable : std::integral_constant<bool, false>
{
};
template <typename T, typename = typename std::enable_if<
						std::is_base_of<Host, T>::value>::type>
struct is_host_callable : std::integral_constant<bool, true>
{
};

template <typename T, typename = void>
struct is_device_callable : std::integral_constant<bool, false>
{
};
template <typename T, typename = typename std::enable_if<
						std::is_base_of<Device, T>::value>::type>
struct is_device_callable : std::integral_constant<bool, true>
{
};

}  // namespace trait

template <typename T>
struct Require : typename std::conditional<
				   std::is_base_of<Host, T>::value, Host, trait::dummy>::type,
				 typename std::conditional<
				   std::is_base_of<Device, T>::value, Device, trait::dummy>::type
{
};

#define __PolyFunctionImpl( ... )                                                   \
	{                                                                               \
		struct __priv                                                               \
		{                                                                           \
			template <typename T, typename = typename std::enable_if<               \
									is_in<T, Host, Device>::value>::type>           \
			struct func                                                             \
			{                                                                       \
				using call_type = Host;                                             \
				struct poly                                                         \
				{                                                                   \
					template <typename T, typename U>                               \
					using vector = std::vector<T, U>;                               \
					using SubMesh = core::SubMesh;                                  \
				};                                                                  \
				__host__ static auto fn __VA_ARGS__                                 \
			};                                                                      \
			template <typename T>                                                   \
			struct func<T, typename std::enable_if<                                 \
							 std::is_same<T, Device>::value>::type>                 \
			{                                                                       \
				using call_type = Device;                                           \
				struct poly                                                         \
				{                                                                   \
					template <typename T>                                           \
					using vector = dev::vector<T>;                                  \
					using SubMesh = core::dev::SubMesh;                             \
				};                                                                  \
				__device__ static auto fn __VA_ARGS__                               \
			};                                                                      \
			template <typename T>                                                   \
			struct return_type_of;                                                  \
			template <typename T, typename... Args>                                 \
			struct return_type_of<T( Args... )>                                     \
			{                                                                       \
				using type = T;                                                     \
			};                                                                      \
		};                                                                          \
		using value_type = typename __priv::return_type_of<__priv::func::fn>::type; \
	}

#define PolyFunction( name, ... ) \
	struct name : __VA_ARGS__     \
					__PolyFunctionImpl

template <typename F, typename... Args>
__host__ __device__ F::value_type call( Args &&... args )
{
	return F::__priv::func<call_type>::fn( std::forward<Args>( args ) );
}

template <typename T>
struct Require : typename std::conditional<
				   std::is_base_of<Host, T>::value, Host, trait::dummy>::type,
				 typename std::conditional<
				   std::is_base_of<Device, T>::value, Device, trait::dummy>::type
{
};

}  // namespace core

}  // namespace koishi
