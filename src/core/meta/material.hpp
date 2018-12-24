#pragma once

#include <vec/vmath.hpp>
#include <core/basic/allocator.hpp>
#include <core/basic/poly.hpp>
#include <core/basic/factory.hpp>
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
	Material( const Properties &props ) :
	  color( get( props, "color", float3{ 0.5, 0.5, 0.5 } ) )
	{
	}

	KOISHI_HOST_DEVICE virtual void apply( SurfaceInterreact &res, Allocator &pool ) const
	{
		res.color = color;
	}
	virtual void print( std::ostream &os ) const { os << "{}"; }

private:
	float3 color;
};

}  // namespace core

}  // namespace koishi
