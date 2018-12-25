#pragma once

#include <map>
#include <string>
#include <functional>
#include <util/config.hpp>

namespace koishi
{
namespace core
{
template <typename U, typename = typename std::enable_if<
						std::is_base_of<emittable, U>::value>::type>
struct Factory
{
	static poly::object<U> create( const Config &conf )
	{
		if ( !ctors().count( conf.name ) )
		{
			KTHROW( "no such type:", conf.name );
		}
		return ctors()[ conf.name ]( conf.props );
	}

	static poly::object<U> create( const std::string &name )
	{
		if ( !ctors().count( name ) )
		{
			KTHROW( "no such type:", name );
		}
		return ctors()[ name ]( Properties() );
	}

	template <typename T, typename = typename std::enable_if<std::is_base_of<U, T>::value>::type>
	static int reg( const std::string &name )
	{
		if ( ctors().count( name ) )
		{
			KTHROW( "multiple defination of type:", name );
		}
		ctors().emplace( name, []( const Properties &props ) { return std::move( poly::make_object<T>( props ) ); } );
		return 0;
	}

private:
	static std::map<std::string, std::function<
								   poly::object<U>( const Properties & )>> &
	  ctors()
	{
		static std::map<std::string, std::function<
									   poly::object<U>( const Properties & )>>
		  cc;
		return cc;
	}

	Factory() = default;
};

}  // namespace core

}  // namespace koishi