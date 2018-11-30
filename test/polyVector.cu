#include <core/basic/poly.hpp>
#include <gtest/gtest.h>

using namespace koishi;
using namespace core;

#ifdef KOISHI_USE_CUDA

struct A : emittable
{
	A( int i ) :
	  n( i )
	{
		poly::vector<int> vv;
		for ( int i = 0; i <= n; ++i )
		{
			vv.emplace_back( i );
		}
		v = std::move( vv );
	}

	__host__ __device__ virtual int f() const
	{
		int s = v.size() * 1000;
		for ( int i = 0; i != v.size(); ++i )
		{
			s += v[ i ];
		}
		return s;
	}

	int n;
	poly::vector<int> v;
};

__global__ void add( const poly::vector<A> &vec, poly::vector<int> &n, poly::vector<const int *> &p )
{
	//n[0] = 1; n[1] = 2;
	//n[0] = 1;
	for ( auto i = 0; i != vec.size(); ++i )
		//n[i] = 1;
		n[ i ] = vec[ i ].f(), p[ i ] = nullptr;
}

#endif

TEST( test_poly_vector, struct_with_non_standard_layout )
{
//	testing::internal::CaptureStdout();
#ifdef KOISHI_USE_CUDA
	poly::vector<A> view;

	int n = 200;

	for ( int i = 0; i != n; ++i )
	{
		view.emplace_back( i );
	}

	EXPECT_EQ( view.size(), n );
	KLOG( view.data() );

	poly::vector<int> nn( view.size() );
	poly::vector<const int *> pp( view.size() );

	KLOG( nn.data() );
	KLOG( pp.data() );

	EXPECT_EQ( n, nn.size() );
	EXPECT_EQ( n, pp.size() );
	EXPECT_EQ( n, view.size() );

	poly::kernel( add, 1, 1 )( view, nn, pp );

	KLOG( nn.data() );
	KLOG( pp.data() );

	EXPECT_EQ( n, nn.size() );

	KLOG( nn.data() );
	KLOG( pp.data() );

	EXPECT_EQ( n, nn.size() );

	int ss = 0;

	for ( int i = 0; i != nn.size(); ++i )
	{
		ss += i;
		EXPECT_EQ( nn[ i ], ss + 1000 * ( i + 1 ) );
		std::cout << nn[ i ] << std::endl;
	}
#else
	KLOG( "no cuda toolkit provided" );
#endif
}

TEST( test_poly_vector, initializer_list )
{
	poly::vector<int> a = { 1, 2, 3 };
	poly::vector<int> b{};
	a = {};
	for ( int i = 0; i != 10000; ++i )
	{
		a.resize( i + 1 );
		a[ i ] = i;
		b.emplace_back( i );
	}
	for ( int i = 0; i != 10000; ++i )
	{
		ASSERT_EQ( a[ i ], i );
		ASSERT_EQ( b[ i ], i );
	}
}

TEST( test_poly_vector, emit_empty_vector )
{
#ifdef KOISHI_USE_CUDA
	poly::vector<A> view;
	poly::vector<int> a;
	poly::vector<const int *> b;
	poly::kernel( add, 1, 1 )( view, a, b );
#endif
}
