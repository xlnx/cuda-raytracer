#pragma once

#include <vec/vmath.hpp>
#include <core/allocator.hpp>
#include "hit.hpp"

namespace koishi
{
namespace core
{
namespace dev
{
struct Material
{
	KOISHI_HOST_DEVICE virtual void fetchTo( dev::Hit &res, Allocator &pool ) const = 0;
};

}  // namespace dev

}  // namespace core

}  // namespace koishi