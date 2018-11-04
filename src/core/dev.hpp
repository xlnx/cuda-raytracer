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
using vector = int;
}

namespace host
{
template <typename T>
using vector = int;
}

#endif

struct Host;
struct Device;

struct Host
{
	template <typename F, typename... Args>
	KOISHI_HOST static typename F::value_type call( Args &&... args )
	{
		return F::__priv::template func<Host, Host, Device>::fn( std::forward<Args>( args )... );
	}
};

struct Device
{
	template <typename F, typename... Args>
	KOISHI_DEVICE static typename F::value_type call( Args &&... args )
	{
		return F::__priv::template func<Device, Host, Device>::fn( std::forward<Args>( args )... );
	}
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

template <bool X, bool... Args>
struct make_and : std::integral_constant<bool, X && make_and<Args...>::value>
{
};
template <bool X>
struct make_and<X> : std::integral_constant<bool, X>
{
};

}  // namespace trait

template <typename... Args>
struct Require : std::conditional<trait::make_and<
									std::is_base_of<Host, Args>::value...>::value,
								  Host, trait::dummy>::type,
				 std::conditional<trait::make_and<
									std::is_base_of<Device, Args>::value...>::value,
								  Device, trait::dummy>::type
{
};

#define __PolyFunctionImpl( ... )                                                                                        \
	{                                                                                                                    \
		struct __priv                                                                                                    \
		{                                                                                                                \
			template <typename _M_T, typename _M_Host, typename _M_Device>                                               \
			struct func;                                                                                                 \
			template <typename _M_Host, typename _M_Device>                                                              \
			struct func<_M_Host, _M_Host, _M_Device>                                                                     \
			{                                                                                                            \
				using call_type = Host;                                                                                  \
				struct poly                                                                                              \
				{                                                                                                        \
					template <typename _M__T>                                                                            \
					using vector = std::vector<_M__T>;                                                                   \
					using SubMesh = core::SubMesh;                                                                       \
				};                                                                                                       \
				template <typename _M_F, typename... _M_Args>                                                            \
				KOISHI_HOST static typename _M_F::value_type call( _M_Args &&... args )                                  \
				{                                                                                                        \
					return _M_F::__priv::template func<call_type, Host, Device>::fn( std::forward<_M_Args>( args )... ); \
				}                                                                                                        \
				KOISHI_HOST static auto fn __VA_ARGS__                                                                   \
			};                                                                                                           \
			template <typename _M_Host, typename _M_Device>                                                              \
			struct func<_M_Device, _M_Host, _M_Device>                                                                   \
			{                                                                                                            \
				using call_type = Device;                                                                                \
				struct poly                                                                                              \
				{                                                                                                        \
					template <typename _M__T>                                                                            \
					using vector = dev::vector<_M__T>;                                                                   \
					using SubMesh = core::dev::SubMesh;                                                                  \
				};                                                                                                       \
				template <typename _M_F, typename... _M_Args>                                                            \
				KOISHI_DEVICE static typename _M_F::value_type call( _M_Args &&... args )                                \
				{                                                                                                        \
					return _M_F::__priv::template func<call_type, Host, Device>::fn( std::forward<_M_Args>( args )... ); \
				}                                                                                                        \
				KOISHI_DEVICE static auto fn __VA_ARGS__                                                                 \
			};                                                                                                           \
			template <typename _M_T>                                                                                     \
			struct return_type_of;                                                                                       \
			template <typename _M_T, typename... _M_Args>                                                                \
			struct return_type_of<_M_T( _M_Args... )>                                                                    \
			{                                                                                                            \
				using type = _M_T;                                                                                       \
			};                                                                                                           \
		};                                                                                                               \
		using value_type = typename __priv::template return_type_of<                                                     \
		  decltype( __priv::template func<Host, Host, Device>::fn )>::type;                                              \
	}

#define PolyFunction( name, ... ) \
	struct name : __VA_ARGS__     \
					__PolyFunctionImpl

}  // namespace core

}  // namespace koishi
