#pragma once

#include <map>
#include <vec/vmath.hpp>
#include <core/basic/allocator.hpp>
#include <core/basic/poly.hpp>
#include <util/config.hpp>
#include "mesh.hpp"
#include "interreact.hpp"

#define MAX_MATERIALS 256

namespace koishi
{
namespace core
{
struct Material : emittable
{
	Material() = default;
	Material( const MaterialProps &config ) {}

	KOISHI_HOST_DEVICE virtual void apply( Interreact &res, Allocator &pool ) const = 0;
	virtual void print( std::ostream &os ) const { os << "{}"; }
};

struct MaterialFactory
{
	static poly::object<Material> create( const std::string &name, const MaterialProps &props )
	{
		if ( !ctors().count( name ) )
		{
			KTHROW( "no such material:", name );
		}
		return ctors()[ name ]( props );
	}

	template <typename T, typename = typename std::enable_if<std::is_base_of<Material, T>::value>::type>
	static int reg( const std::string &name )
	{
		if ( ctors().count( name ) )
		{
			KTHROW( "multiple defination of material:", name );
		}
		ctors().emplace( name, []( const MaterialProps &props ) { return std::move( poly::make_object<T>( props ) ); } );
		return 0;
	}

private:
	static std::map<std::string, std::function<
								   poly::object<Material>( const MaterialProps & )>> &
	  ctors()
	{
		static std::map<std::string, std::function<
									   poly::object<Material>( const MaterialProps & )>>
		  cc;
		return cc;
	}

	MaterialFactory() = default;
};

}  // namespace core

}  // namespace koishi
