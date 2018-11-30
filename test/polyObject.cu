#include <core/basic/poly.hpp>
#include <gtest/gtest.h>

using namespace koishi;
using namespace core;

TEST( test_poly_object, object_create )
{
	poly::object<int> a = poly::make_object<int>( 1 );
	ASSERT_EQ( *a, 1 );
	*a = 2;
	ASSERT_EQ( *a, 2 );
}

struct A : emittable<A>
{
	A() { n++; }
	A( const A & ) { n++; }
	A( A && ) { 
		printf("move A\n");
#ifndef __CUDA_ARCH__
		n++;
#endif
	}
	A &operator=( A && ) = default;
	A &operator=( const A & ) = default;
	~A() { n--; }
	
	KOISHI_HOST_DEVICE virtual int f() const
	{
		return 1;
	}

	int x = 1;
	static int n;
};

struct B : emittable<B, A>
{
	B() { n++; }
	B( const B & ) { n++; }
	B( B && ) { 
		printf("move B\n");
#ifndef __CUDA_ARCH__
		n++;
#endif
	}
	B &operator=( B && ) = default;
	B &operator=( const B & ) = default;
	~B() { n--; }

	KOISHI_HOST_DEVICE int f() const override
	{
		return 2;
	}

	int y = 2;
	static int n;
};

int A::n = 0;
int B::n = 0;

TEST( test_poly_object, object_sealed )
{
	ASSERT_EQ( A::n, 0 );
	ASSERT_EQ( B::n, 0 );
	{
		poly::object<A> a = poly::make_object<A>();
		ASSERT_EQ( A::n, 1 );
		ASSERT_EQ( B::n, 0 );
		ASSERT_EQ( a->x, 1 );
		{
			poly::object<A> b = poly::make_object<B>();
			ASSERT_EQ( A::n, 2 );
			ASSERT_EQ( B::n, 1 );
			ASSERT_EQ( b->x, 1 );
			// ASSERT_EQ( b->y, 2 );
		}
		ASSERT_EQ( A::n, 1 );
		ASSERT_EQ( B::n, 0 );
	}
	ASSERT_EQ( A::n, 0 );
	ASSERT_EQ( B::n, 0 );
}

TEST( test_poly_object, object_sealed_2 )
{
	ASSERT_EQ( A::n, 0 );
	ASSERT_EQ( B::n, 0 );
	{
		poly::object<A> a = poly::make_object<A>();
		ASSERT_EQ( A::n, 1 );
		ASSERT_EQ( B::n, 0 );
		ASSERT_EQ( a->x, 1 );
		{
			poly::object<A> b = poly::make_object<B>();
			ASSERT_EQ( A::n, 2 );
			ASSERT_EQ( B::n, 1 );
			ASSERT_EQ( b->x, 1 );
			a = std::move( b );
			ASSERT_EQ( A::n, 1 );
			ASSERT_EQ( B::n, 1 );
		}
		ASSERT_EQ( A::n, 1 );
		ASSERT_EQ( B::n, 1 );
	}
	ASSERT_EQ( A::n, 0 );
	ASSERT_EQ( B::n, 0 );
}

TEST( test_poly_object, object_sealed_3 )
{
	ASSERT_EQ( A::n, 0 );
	ASSERT_EQ( B::n, 0 );
	{
		poly::object<A> a = poly::make_object<A>();
		ASSERT_EQ( A::n, 1 );
		ASSERT_EQ( B::n, 0 );
		ASSERT_EQ( a->x, 1 );
		{
			poly::object<A> b = poly::make_object<B>();
			ASSERT_EQ( A::n, 2 );
			ASSERT_EQ( B::n, 1 );
			ASSERT_EQ( b->x, 1 );
			poly::object<B> c = poly::static_object_cast<B>( std::move( b ) );
			ASSERT_EQ( A::n, 2 );
			ASSERT_EQ( B::n, 1 );
			ASSERT_EQ( c->x, 1 );
			ASSERT_EQ( c->y, 2 );
		}
		ASSERT_EQ( A::n, 1 );
		ASSERT_EQ( B::n, 0 );
	}
	ASSERT_EQ( A::n, 0 );
	ASSERT_EQ( B::n, 0 );
}

#ifdef KOISHI_USE_CUDA
__global__ void g( 
	poly::vector<int> &b,
	const poly::object<A> &a
)
{
	// b[ 0 ] = a->x;
	//b[ 1 ] = a->f();
}
#endif

TEST( test_poly_object, object_polymorphism )
{
#ifdef KOISHI_USE_CUDA
	poly::object<A> a = poly::make_object<B>();
	poly::vector<int> b( 2 );
	poly::kernel( g, 1, 1 )( b, a );
	EXPECT_EQ( b[ 0 ], 1 );
	EXPECT_EQ( b[ 1 ], 2 );
#endif
}
