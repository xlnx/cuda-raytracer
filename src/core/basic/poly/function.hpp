#pragma once

#include <type_traits>
#include <utility>
#include <string>
#include <vec/trait.hpp>
#include <vec/vmath.hpp>

namespace koishi
{
namespace core
{
struct Host;
struct Device;

namespace __global_semicolon
{
}

namespace trait
{
template <typename T>
struct dummy
{
};

}  // namespace trait

struct Host
{
	template <typename F, typename... Args>
	KOISHI_HOST static auto call( Args &&... args )
	{
		return F::__priv::template func<F, koishi::core::trait::dummy<Host>, Host, Device>::fn( std::forward<Args>( args )... );
	}
};

struct Device
{
	template <typename F, typename... Args>
	KOISHI_DEVICE static auto call( Args &&... args )
	{
		return F::__priv::template func<F, koishi::core::trait::dummy<Device>, Host, Device>::fn( std::forward<Args>( args )... );
	}
};

struct HostDevice : Host, Device
{
};

namespace trait
{
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

template <typename... Args>
struct Restrict : std::conditional<trait::make_and<
									 std::is_base_of<Host, Args>::value...>::value,
								   Host, trait::dummy<Host>>::type,
				  std::conditional<trait::make_and<
									 std::is_base_of<Device, Args>::value...>::value,
								   Device, trait::dummy<Device>>::type
{
};

template <typename T>
struct Check : std::integral_constant<bool, std::is_base_of<Device, T>::value |
											  std::is_base_of<Host, T>::value>
{
};

}  // namespace trait

template <typename... Args>
struct Require : trait::Restrict<Args...>, trait::Check<trait::Restrict<Args...>>
{
};

template <typename X, typename Y>
struct On : std::conditional<std::is_base_of<Y, X>::value, HostDevice, trait::dummy<HostDevice>>::type
{
};

#define __PolyFunctionImpl()                                                                                                                                \
	struct __priv                                                                                                                                           \
	{                                                                                                                                                       \
		template <typename _M_Self, typename _M_T, typename _M_Host, typename _M_Device>                                                                    \
		struct func;                                                                                                                                        \
		template <typename _M_Self, typename _M_Curr, typename _M_Host, typename _M_Device>                                                                 \
		struct func<_M_Self, koishi::core::trait::dummy<_M_Curr>, _M_Host, _M_Device>                                                                       \
		{                                                                                                                                                   \
			using call_type = _M_Curr;                                                                                                                      \
			template <typename _M_F, typename... _M_Args>                                                                                                   \
			KOISHI_HOST_DEVICE static auto call( _M_Args &&... args )                                                                                       \
			  -> decltype( _M_F::__priv::template func<_M_F, koishi::core::trait::dummy<call_type>, Host, Device>::fn( std::forward<_M_Args>( args )... ) ) \
			{                                                                                                                                               \
				return _M_F::__priv::template func<_M_F, koishi::core::trait::dummy<call_type>, Host, Device>::fn( std::forward<_M_Args>( args )... );      \
			}                                                                                                                                               \
			KOISHI_HOST_DEVICE static auto fn

#define PolyFunction( name, ... ) \
	struct name : __VA_ARGS__     \
	{                             \
		PolyStruct( name );       \
		__PolyFunctionImpl()

#define EndPolyFunction() \
	}                     \
	;                     \
	}                     \
	;                     \
	}

#define PolyStruct( name )                          \
	static constexpr const char *className = #name; \
	static std::string &getInstanceName()           \
	{                                               \
		static std::string instance = #name;        \
		return instance;                            \
	}                                               \
	using Self = name

}  // namespace core

}  // namespace koishi
