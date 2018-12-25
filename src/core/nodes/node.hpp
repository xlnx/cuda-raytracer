#pragma once

#include <iostream>
#include <core/basic/basic.hpp>

namespace koishi
{
namespace core
{
struct Node : emittable
{
	virtual void writeNode( json &j ) const = 0;
};

}  // namespace core

}  // namespace koishi