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

template <template <typename> typename... Args>
struct templates
{
	using type = templates<Args...>;
};

template <typename... X>
struct ApplyEnum;

template <template <typename> typename T, typename X, typename... Args>
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

template <template <typename> typename T, typename L>
struct Enum;

template <template <typename> typename T, typename... Args>
struct Enum<T, types<Args...>> : types<T<Args>...>
{
	static void apply()
	{
		ApplyEnum<T<Args>...>::apply();
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

template <typename T, typename L>
struct Permutation;

template <template <typename> typename... Templates, typename... Args>
struct Permutation<templates<Templates...>,
				   types<Args...>> : Concat<typename Enum<Templates, types<Args...>>::type...>
{
	static void apply()
	{
		ApplyPermutation<Enum<Templates, types<Args...>>...>::apply();
	}
};

template <typename... Args>
struct ChainedPermutation;

template <typename T, typename U, typename... Args>
struct ChainedPermutation<T, U, Args...> : Permutation<
											 T, typename ChainedPermutation<U, Args...>::type>
{
	static void apply()
	{
		ChainedPermutation<U, Args...>::apply();
		Permutation<
		  T, typename ChainedPermutation<U, Args...>::type>::apply();
	}
};

template <typename T, typename U>
struct ChainedPermutation<T, U> : Permutation<T, U>
{
	static void apply()
	{
		Permutation<T, U>::apply();
	}
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
struct Factory
{
	using value_type = std::shared_ptr<RendererBase>;

	Factory()
	{
		using chain = __trait::ChainedPermutation<Args...>;
		chain::apply();
		__trait::CtorForEach<typename __trait::RemoveInvalid<
		  typename chain::type>::type>::apply( ctors );
	}

	value_type create( const std::string &name, uint w, uint h )
	{
		if ( __trait::name2Typeid().count( name ) && ctors.count( __trait::name2Typeid()[ name ] ) )
		{
			return ctors[ __trait::name2Typeid()[ name ] ]( w, h );
		}
		else
		{
			KTHROW( unregistered factory type );
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

using __trait::templates;
using __trait::types;

}  // namespace core

}  // namespace koishi