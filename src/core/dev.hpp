#pragma once

#include <type_traits>
#include <vec/trait.hpp>
#include <vec/vmath.hpp>
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

#else

namespace dev
{
template <typename T>
using vector = void;
}

namespace host
{
template <typename T>
using vector = void;
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
template <typename T>
struct is_host_callable<T, typename std::enable_if<
							 std::is_base_of<Host, T>::value>::type> : std::integral_constant<bool, true>
{
};

template <typename T, typename = void>
struct is_device_callable : std::integral_constant<bool, false>
{
};
template <typename T>
struct is_device_callable<T, typename std::enable_if<
							   std::is_base_of<Device, T>::value>::type> : std::integral_constant<bool, true>
{
};

}  // namespace trait

template <typename T>
struct Require : std::conditional<std::is_base_of<Host, T>::value, Host, trait::dummy>::type,
				 std::conditional<std::is_base_of<Device, T>::value, Device, trait::dummy>::type
{
};

#define __PolyFunctionImpl( ... )                                                              \
	{                                                                                          \
		struct __priv                                                                          \
		{                                                                                      \
			template <typename _M_T, typename = typename std::enable_if<                       \
									   koishi::trait::is_in<_M_T, Host, Device>::value>::type> \
			struct func                                                                        \
			{                                                                                  \
				using call_type = Host;                                                        \
				struct poly                                                                    \
				{                                                                              \
					template <typename _M__T>                                                  \
					using vector = std::vector<_M__T>;                                         \
					using SubMesh = core::SubMesh;                                             \
				};                                                                             \
				KOISHI_HOST static auto fn __VA_ARGS__                                         \
			};                                                                                 \
			template <typename _M_T>                                                           \
			struct func<_M_T, typename std::enable_if<                                         \
								std::is_same<_M_T, Device>::value>::type>                      \
			{                                                                                  \
				using call_type = Device;                                                      \
				struct poly                                                                    \
				{                                                                              \
					template <typename _M__T>                                                  \
					using vector = dev::vector<_M__T>;                                         \
					using SubMesh = core::dev::SubMesh;                                        \
				};                                                                             \
				KOISHI_DEVICE static auto fn __VA_ARGS__                                       \
			};                                                                                 \
			template <typename _M_T>                                                           \
			struct return_type_of;                                                             \
			template <typename _M_T, typename... _M_Args>                                      \
			struct return_type_of<_M_T( _M_Args... )>                                          \
			{                                                                                  \
				using type = _M_T;                                                             \
			};                                                                                 \
		};                                                                                     \
		using value_type = typename __priv::template return_type_of<                           \
		  decltype( __priv::template func<Host>::fn )>::type;                                  \
	}

#define PolyFunction( name, ... ) \
	struct name : __VA_ARGS__     \
					__PolyFunctionImpl

using call_type = Host;

template <typename F, typename... Args>
KOISHI_HOST_DEVICE typename F::value_type call( Args &&... args )
{
	return F::__priv::template func<call_type>::fn( std::forward<Args>( args )... );
}

}  // namespace core

}  // namespace koishi
