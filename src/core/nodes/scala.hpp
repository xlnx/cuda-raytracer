#pragma once

#include <core/basic/basic.hpp>
#include <core/meta/varyings.hpp>

namespace koishi
{
namespace core
{
template <typename T>
struct Scala : emittable
{
	Scala() = default;
	Scala( const Properties &props )
	{
	}

	KOISHI_HOST_DEVICE virtual T compute( const Varyings &varyings, Allocator &pool ) const = 0;
	virtual void print( std::ostream &os ) const { os << "{}"; }
};

}  // namespace core

}  // namespace koishi
