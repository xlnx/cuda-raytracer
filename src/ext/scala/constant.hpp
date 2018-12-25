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
	  value( get<T>( props, "value" ) )
	{
	}

	KOISHI_HOST_DEVICE T compute( const Varyings &, Allocator & ) const override
	{
		return value;
	}

	void print( std::ostream &os ) const override
	{
		nlohmann::json json = {
			{ "Constant", { { "value", value } } }
		};
		os << json.dump();
	}

private:
	const T value;
};

}  // namespace ext

}  // namespace koishi