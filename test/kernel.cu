#include <gtest/gtest.h>
#include <core/basic/poly/kernel.hpp>
#include <core/basic/poly/vector.hpp>

using namespace koishi;
using namespace core;

#ifdef KOISHI_USE_CUDA
__global__
void f( unsigned a )
{
}
__global__
void g( PolyVector<int> &a, unsigned b )
{
}

__global__ 
void h( unsigned a, PolyVector<int> &b )
{
}
#endif

TEST( test_kernel,  )
{
#ifdef KOISHI_USE_CUDA
	PolyVector<int> a;
	kernel( h, 1, 1 )( 0u, a );
	kernel( g, 1, 1 )( a, 0u );
	kernel( f, 1, 1 )( 0u );
#endif
}
