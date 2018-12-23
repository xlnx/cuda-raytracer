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
	Material( const Properties &config ) {}

	KOISHI_HOST_DEVICE virtual void apply( SurfaceInterreact &res, Allocator &pool ) const = 0;
	virtual void print( std::ostream &os ) const { os << "{}"; }
};

}  // namespace core

}  // namespace koishi
