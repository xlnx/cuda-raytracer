#pragma once

#include <map>
#include <string>
#include <memory>
#include <type_traits>
#include <typeinfo>
#include <functional>
#include <util/debug.hpp>
#include "renderer.hpp"

namespace koishi
{
namespace core
{
namespace __trait
{
inline std::map<std::string, std::string> &name2Typeid()
{
	static std::map<std::string, std::string> m;
	return m;
}

template <typename... Args>
struct types
{
	using type = types<Args...>;
};

template <template <typename> class... Args>
struct templates
{
	using type = templates<Args...>;
};

template <typename... X>
struct ApplyEnum;

template <template <typename> class T, typename X, typename... Args>
struct ApplyEnum<T<X>, Args...>
{
	static void apply()
	{
		T<X>::getInstanceName() = std::string( T<X>::className ) + "@@" + X::getInstanceName();
		name2Typeid().emplace( T<X>::getInstanceName(), typeid( T<X> ).name() );
		ApplyEnum<Args...>::apply();
	}
};

template <>
struct ApplyEnum<>
{
	static void apply() {}
};

template <template <typename> class T, typename L>
struct Enum;

template <template <typename> class T, typename... Args>
struct Enum<T, types<Args...>> : types<T<Args>...>
{
	static void apply()
	{
		ApplyEnum<T<Args>...>::apply();
	}
};

template <typename... Args>
struct DoArgsBuilder;

template <typename T, typename... Args>
struct DoArgsBuilder<T, Args...>
{
	static std::string apply()
	{
		return T::getInstanceName() + ", " + DoArgsBuilder<Args...>::apply();
	}
};

template <typename T>
struct DoArgsBuilder<T>
{
	static std::string apply()
	{
		return T::getInstanceName();
	}
};

template <typename... Args>
struct ArgsBuilder
{
	static std::string apply()
	{
		return "<" + DoArgsBuilder<Args...>::apply() + ">";
	}
};

template <typename... Args>
struct Concat;

template <typename... Args1, typename... Args2, typename... Rest>
struct Concat<types<Args1...>,
			  types<Args2...>, Rest...> : Concat<types<Args1..., Args2...>, Rest...>
{
};

template <typename... Args>
struct Concat<types<Args...>> : types<Args...>
{
};

template <typename T, typename L>
struct Permute;

template <typename... Curr, typename... Args>
struct Permute<types<Curr...>, types<Args...>> : types<types<Curr..., Args>...>
{
};

template <typename C, typename X, typename... Args>
struct DoPermuteSeq;

template <typename... Curr, typename... Args, typename... Rest>
struct DoPermuteSeq<types<Curr...>, types<Args...>, Rest...> : DoPermuteSeq<typename Concat<
																			  typename Permute<Curr, types<Args...>>::type...>::type,
																			Rest...>
{
};

template <typename... Curr, typename... Args>
struct DoPermuteSeq<types<Curr...>, types<Args...>> : Concat<typename Permute<Curr, types<Args...>>::type...>
{
};

template <typename... Args>
struct PermuteSeq : DoPermuteSeq<types<types<>>, Args...>
{
};

// clang-format off
#define KOISHI_DEF_TEMPLATE( N )                                                                                                                                                    \
	template <template <KREP( N, (typename) )> class... Args>                                                                                                                    \
	struct templates##N                                                                                                                                                             \
	{                                                                                                                                                                               \
		using type = templates##N<Args...>;                                                                                                                                         \
	};                                                                                                                                                                              \
	template <template <KREP( N, (typename) )> class T, typename U>                                                                                                              \
	struct Instance##N;                                                                                                                                                             \
	template <template <KREP( N, (typename) )> class T, typename ...Args>                                                                                                        \
	struct Instance##N<T, types<Args...>> : types<T<Args...>>                                                                                                                       \
	{                                                                                                                                                                               \
	};                                                                                                                                                                              \
	template <typename... Args>                                                                                                                                                     \
	struct DoApplyEnum##N;                                                                                                                                                          \
	template <template <KREP( N, (typename) )> class T, typename... Args, typename... Rest>                                                                                      \
	struct DoApplyEnum##N<T<Args...>, Rest...>                                                                                                                                      \
	{                                                                                                                                                                               \
		static void apply()                                                                                                                                                         \
		{                                                                                                                                                                           \
			T<Args...>::getInstanceName() = std::string( T<Args...>::className ) + ArgsBuilder<Args...>::apply();                                                                   \
			name2Typeid().emplace( T<Args...>::getInstanceName(), typeid( T<Args...> ).name() );                                                                                    \
			DoApplyEnum##N<Rest...>::apply();                                                                                                                                       \
		}                                                                                                                                                                           \
	};                                                                                                                                                                              \
	template <>                                                                                                                                                                     \
	struct DoApplyEnum##N<>                                                                                                                                                         \
	{                                                                                                                                                                               \
		static void apply() {}                                                                                                                                                      \
	};                                                                                                                                                                              \
	template <typename T>                                                                                                                                                           \
	struct ApplyEnum##N;                                                                                                                                                            \
	template <typename... Args>                                                                                                                                                     \
	struct ApplyEnum##N<types<Args...>>                                                                                                                                             \
	{                                                                                                                                                                               \
		static void apply()                                                                                                                                                         \
		{                                                                                                                                                                           \
			DoApplyEnum##N<Args...>::apply();                                                                                                                                       \
		}                                                                                                                                                                           \
	};                                                                                                                                                                              \
	template <template <KREP( N, (typename) )> class T, typename... Args>                                                                                                        \
	struct Enum##N : Concat<typename Instance##N<T, Args>::type...>                                                                                                                 \
	{                                                                                                                                                                               \
		static void apply()                                                                                                                                                         \
		{                                                                                                                                                                           \
			ApplyEnum##N<typename Concat<typename Instance##N<T, Args>::type...>::type>::apply();                                                                                   \
		}                                                                                                                                                                           \
	};                                                                                                                                                                              \
	template <typename... Args>                                                                                                                                                     \
	struct ApplyPermutation##N;                                                                                                                                                     \
	template <typename X, typename... Args>                                                                                                                                         \
	struct ApplyPermutation##N<X, Args...>                                                                                                                                          \
	{                                                                                                                                                                               \
		static void apply()                                                                                                                                                         \
		{                                                                                                                                                                           \
			X::apply();                                                                                                                                                             \
			ApplyPermutation##N<Args...>::apply();                                                                                                                                  \
		}                                                                                                                                                                           \
	};                                                                                                                                                                              \
	template <>                                                                                                                                                                     \
	struct ApplyPermutation##N<>                                                                                                                                                    \
	{                                                                                                                                                                               \
		static void apply() {}                                                                                                                                                      \
	};                                                                                                                                                                              \
	template <typename T, typename U>                                                                                                                                               \
	struct DoPermutation##N;                                                                                                                                                          \
	template <template <KREP( N, (typename) )> class... Templates, typename ...Args>                                                                                             \
	struct DoPermutation##N<templates##N<Templates...>, types<Args...>>:                                                                                                              \
		Concat<typename Enum##N<Templates, Args...>::type...>                                                                                                                       \
	{                                                                                                                                                                               \
		static void apply()                                                                                                                                                         \
		{                                                                                                                                                                           \
			ApplyPermutation##N<Enum##N<Templates, Args...>...>::apply();                                                                                                           \
		}                                                                                                                                                                           \
	};                                                                                                                                                                              \
	template <typename T, KREPID( N, (typename U), () )>                                                                                                                            \
	struct Permutation##N;                                                                                                                                                          \
	template <template <KREP( N, (typename) )> class... Templates, KREPID( N, (typename... Args), () )>                                                                          \
	struct Permutation##N<templates##N<Templates...>, KREPID( N, (types<Args), (...>) )>:                                                                                           \
		DoPermutation##N<templates##N<Templates...>, typename PermuteSeq<KREPID(N, (types<Args), (...>) )>::type>                                                                     \
	{                                                                                                                                                                               \
	};                                                                                                                                                                              \
	template <template <KREP( N, (typename) )> class... Templates, KREPID( N, (typename... Args), () )>                                                                          \
	struct DoRecursedPermutation<templates##N<Templates...>,                                                                                                                        \
								   KREPID( N, (types<Args), (...>) )> : Permutation##N<templates##N<Templates...>,                                                                  \
																								KREPID( N, (typename DoRecursedPermutation<Args), (...>::type ) )>                  \
	{                                                                                                                                                                               \
		static int apply()                                                                                                                                                          \
		{                                                                                                                                                                           \
			KREPID( N, ( DoRecursedPermutation<Args), (...>::apply() ) );                                                                                                           \
			Permutation##N<templates##N<Templates...>,                                                                                                                              \
							 KREPID( N, (typename DoRecursedPermutation<Args), (...>::type ) )>::apply();                                                                           \
			return 0;                                                                                                                                                               \
		}                                                                                                                                                                           \
	}
// clang-format on

template <typename... Args>
struct ApplyPermutation;

template <typename X, typename... Args>
struct ApplyPermutation<X, Args...>
{
	static void apply()
	{
		X::apply();
		ApplyPermutation<Args...>::apply();
	}
};

template <>
struct ApplyPermutation<>
{
	static void apply() {}
};

template <typename... Args>
struct DoRecursedPermutation : types<Args...>
{
	static int apply() { return 0; }
};

KOISHI_DEF_TEMPLATE( 1 );
KOISHI_DEF_TEMPLATE( 2 );
KOISHI_DEF_TEMPLATE( 3 );
KOISHI_DEF_TEMPLATE( 4 );
KOISHI_DEF_TEMPLATE( 5 );
KOISHI_DEF_TEMPLATE( 6 );
KOISHI_DEF_TEMPLATE( 7 );
KOISHI_DEF_TEMPLATE( 8 );

template <typename... Args>
struct RecursedPermutation : DoRecursedPermutation<Args...>
{
};

template <typename T>
struct RemoveInvalid;

template <typename T, typename... Args>
struct RemoveInvalid<types<T, Args...>> : Concat<typename std::conditional<T::value, types<T>, types<>>::type,
												 typename RemoveInvalid<types<Args...>>::type>
{
};

template <typename T>
struct RemoveInvalid<types<T>> : std::conditional<T::value, types<T>, types<>>
{
};

using CtorType = std::function<std::shared_ptr<RendererBase>( uint w, uint h )>;

template <typename U>
struct CtorForEach;

template <typename T, typename... Args>
struct CtorForEach<types<T, Args...>>
{
	static void apply( std::map<std::string, CtorType> &ctors )
	{
		ctors[ typeid( T ).name() ] = []( uint w, uint h ) -> std::shared_ptr<RendererBase> {
			return std::static_pointer_cast<RendererBase>(
			  std::make_shared<Renderer<T>>( w, h ) );
		};
		CtorForEach<types<Args...>>::apply( ctors );
	}
};

template <>
struct CtorForEach<types<>>
{
	static void apply( std::map<std::string, CtorType> & )
	{
	}
};

}  // namespace __trait

template <typename... Args>
struct TemplateFactory
{
	template <typename T>
	struct dummy;

	using value_type = std::shared_ptr<RendererBase>;

	TemplateFactory()
	{
		using permute = __trait::RecursedPermutation<Args...>;
		permute::apply();
		__trait::CtorForEach<typename __trait::RemoveInvalid<
		  typename permute::type>::type>::apply( ctors );
	}

	value_type create( const std::string &name, uint w, uint h )
	{
		if ( __trait::name2Typeid().count( name ) && ctors.count( __trait::name2Typeid()[ name ] ) )
		{
			return ctors[ __trait::name2Typeid()[ name ] ]( w, h );
		}
		else
		{
			KTHROW( "unregistered factory type:", name );
		}
	}

	std::vector<std::string> getValidTypes()
	{
		std::vector<std::string> vec;
		for ( auto &e : __trait::name2Typeid() )
		{
			if ( ctors.count( e.second ) )
			{
				vec.emplace_back( e.first );
			}
		}
		vec.emplace_back();
		return vec;
	}

private:
	std::map<std::string, __trait::CtorType> ctors;
};

using __trait::templates1;
using __trait::templates2;
using __trait::templates3;
using __trait::templates4;
using __trait::templates5;
using __trait::templates6;
using __trait::templates7;
using __trait::templates8;
using __trait::types;

}  // namespace core

}  // namespace koishi