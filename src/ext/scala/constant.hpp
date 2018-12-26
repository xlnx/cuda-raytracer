#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
template <typename T>
struct Constant : Scala<T>
{
	Constant( const Properties &props ) :
	  Scala<T>( props ),
	  value( get<T>( props, "value" ) )
	{
	}

	KOISHI_HOST_DEVICE T compute( const Varyings &, Allocator & ) const override
	{
		return value;
	}

	void writeNode( json &j ) const override
	{
		j = value;
	}

private:
	const T value;
};

}  // namespace ext

}  // namespace koishi