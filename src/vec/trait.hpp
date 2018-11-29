#pragma once

#include <type_traits>

namespace koishi
{
namespace trait
{
template <typename T, typename X, typename... R>
struct is_in
{
	constexpr static bool value = std::is_same<T, X>::value || is_in<T, R...>::value;
};
template <typename T, typename X>
struct is_in<T, X>
{
	constexpr static bool value = std::is_same<T, X>::value;
};

template <typename... T>
struct make_void
{
	using type = void;
};
template <typename... T>
using void_t = typename make_void<T...>::type;

#define KOISHI_CHECK_HAS_COMP( comp )                                         \
	template <typename T, typename = void>                                    \
	struct has_##comp                                                         \
	{                                                                         \
		static constexpr bool value = false;                                  \
	};                                                                        \
	template <typename T>                                                     \
	struct has_##comp<T, trait::void_t<decltype( std::declval<T &>().comp )>> \
	{                                                                         \
		static constexpr bool value = true;                                   \
	}
KOISHI_CHECK_HAS_COMP( x );
KOISHI_CHECK_HAS_COMP( y );
KOISHI_CHECK_HAS_COMP( z );
KOISHI_CHECK_HAS_COMP( w );

#undef KOISHI_CHECK_HAS_COMP

template <typename T>
struct is_vec1
{
	static constexpr bool value = has_x<T>::value && !has_y<T>::value;
};

template <typename T>
struct is_vec2
{
	static constexpr bool value = has_y<T>::value && !has_z<T>::value;
};

template <typename T>
struct is_vec3
{
	static constexpr bool value = has_z<T>::value && !has_w<T>::value;
};

template <typename T>
struct is_vec4
{
	static constexpr bool value = has_w<T>::value;
};

template <typename T>
struct com
{
	using type = decltype( std::declval<T &>().x );
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

}  // namespace koishi