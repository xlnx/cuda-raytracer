#include <core/basic/poly.hpp>
#include <gtest/gtest.h>

using namespace koishi;
using namespace core;

#ifdef KOISHI_USE_CUDA

struct A : Emittable
{
	PolyStruct( A );

	A( int i ) :
	  n( i )
	{
		PolyVector<int> vv;
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
	PolyVectorView<int> v;
};

__global__ void add( const PolyVectorView<A> &vec, PolyVectorView<int> &n, PolyVectorView<const int *> &p )
{
	//n[0] = 1; n[1] = 2;
	//n[0] = 1;
	for ( auto i = 0; i != vec.size(); ++i )
		//n[i] = 1;
		n[ i ] = vec[ i ].f(), p[ i ] = vec[ i ].v.data();
}

#endif

TEST( test_poly_vector, struct_with_non_standard_layout )
{
//	testing::internal::CaptureStdout();
#ifdef KOISHI_USE_CUDA
	PolyVector<A> vec;

	int n = 20;

	for ( int i = 0; i != n; ++i )
	{
		vec.emplace_back( i );
	}
	PolyVectorView<A> view = std::move( vec );

	EXPECT_EQ( view.size(), n );
	LOG( view.data() );

	view.emitAndReplace();

	EXPECT_EQ( view.size(), n );
	LOG( view.data() );

	PolyVectorView<int> nn( view.size() );
	PolyVectorView<const int *> pp( view.size() );

	EXPECT_EQ( nn.space(), 0 );
	LOG( nn.data() );
	EXPECT_EQ( pp.space(), 0 );
	LOG( pp.data() );

	EXPECT_EQ( n, nn.size() );

	nn.emitAndReplace();
	pp.emitAndReplace();

	EXPECT_EQ( nn.space(), 1 );
	LOG( nn.data() );
	EXPECT_EQ( pp.space(), 1 );
	LOG( pp.data() );

	EXPECT_EQ( n, nn.size() );

	kernel( add, 1, 1 )( view, nn, pp );

	EXPECT_EQ( nn.space(), 1 );
	LOG( nn.data() );
	EXPECT_EQ( pp.space(), 1 );
	LOG( pp.data() );

	EXPECT_EQ( n, nn.size() );

	nn.fetchAndReplace();
	pp.fetchAndReplace();

	EXPECT_EQ( nn.space(), 0 );
	LOG( nn.data() );
	EXPECT_EQ( pp.space(), 0 );
	LOG( pp.data() );

	EXPECT_EQ( n, nn.size() );

	for ( auto &e : nn )
		std::cout << e << std::endl;

	int ss = 0;

	for ( int i = 0; i != nn.size(); ++i )
	{
		ss += i;
		EXPECT_EQ( nn[ i ], ss + 1000 * ( i + 1 ) );
	}
#else
	LOG( "no cuda toolkit provided" );
#endif
}
