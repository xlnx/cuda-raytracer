#pragma once

#include <vec/vmath.hpp>
#include <core/basic/allocator.hpp>
#include <core/basic/poly.hpp>
#include <core/basic/factory.hpp>
#include <util/config.hpp>
#include "input.hpp"

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

	KOISHI_HOST_DEVICE virtual T compute( const Input &input, Allocator &pool ) const = 0;
	virtual void print( std::ostream &os ) const { os << "{}"; }
};

}  // namespace core

}  // namespace koishi
