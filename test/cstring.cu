#include <util/cstring.hpp>
#include <gtest/gtest.hpp>

TEST( cstring, compile_and_concat )
{
	constexpr auto a = StringLiteral( "abcd" );
}