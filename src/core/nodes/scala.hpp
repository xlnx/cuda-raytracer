#pragma once

#include <core/meta/varyings.hpp>
#include "node.hpp"

namespace koishi
{
namespace core
{
template <typename T>
struct Scala : Node
{
	using Node::Node;

	KOISHI_HOST_DEVICE virtual T compute(
	  const Varyings &varyings, Allocator &pool ) const = 0;
};

}  // namespace core

}  // namespace koishi
