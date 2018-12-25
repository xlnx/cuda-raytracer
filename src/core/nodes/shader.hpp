#pragma once

#include <core/basic/basic.hpp>
#include <core/misc/sampler.hpp>
#include "target.hpp"

namespace koishi
{
namespace core
{
struct Shader : emittable
{
	Shader() = default;
	Shader( const Properties &props )
	{
	}

	KOISHI_HOST_DEVICE virtual void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, uint target = 0 ) const = 0;

	virtual void print( std::ostream &os ) const
	{
		os << "{}";
	}
};

}  // namespace core

}  // namespace koishi