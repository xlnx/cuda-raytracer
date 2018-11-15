#include <core/basic/poly.hpp>
#include <gtest/gtest.h>

using namespace koishi;
using namespace core;

TEST( test_poly_struct, struct_inherit )
{
#ifdef KOISHI_USE_CUDA

#else
	LOG( "no cuda toolkit provided" );
#endif
}