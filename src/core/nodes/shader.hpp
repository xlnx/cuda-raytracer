#pragma once

#include <core/misc/sampler.hpp>
#include "node.hpp"
#include "target.hpp"

namespace koishi
{
namespace core
{
struct Shader : Node
{
	Shader( const Properties &props ) :
	  Node( props )
	{
	}

	KOISHI_HOST_DEVICE virtual void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, uint target = 0 ) const = 0;
};

}  // namespace core

}  // namespace koishi