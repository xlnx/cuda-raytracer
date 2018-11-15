#pragma once

#include <vec/vmath.hpp>
#include <core/basic/allocator.hpp>
#include <core/basic/poly.hpp>
#include "mesh.hpp"
#include "interreact.hpp"

namespace koishi
{
namespace core
{
struct Material: Poly<Material>
{
	KOISHI_HOST_DEVICE virtual void fetchTo( Interreact & res, Allocator & pool ) const = 0;
};

}  // namespace core

}  // namespace koishi