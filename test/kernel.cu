#include <gtest/gtest.h>
#include <poly/kernel.hpp>
#include <poly/vector.hpp>

using namespace koishi;

#ifdef KOISHI_USE_CUDA
__global__ void f( unsigned a )
{
}
__global__ void g( poly::vector<int> &a, unsigned b )
{
}

__global__ void h( unsigned a, poly::vector<int> &b )
{
}
#endif

TEST( test_kernel, )
{
#ifdef KOISHI_USE_CUDA
	poly::vector<int> a;
	poly::kernel( h, 1, 1 )( 0u, a );
	poly::kernel( g, 1, 1 )( a, 0u );
	poly::kernel( f, 1, 1 )( 0u );
#endif
}
