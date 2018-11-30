#include <core/basic/poly.hpp>
#include <gtest/gtest.h>

using namespace koishi;
using namespace core;

struct A : emittable
{
	A() { n++; }
	A( const A & ) { n++; }
	KOISHI_HOST_DEVICE A( A &&other ):
	  emittable( std::move( other ) )
	{ 
		printf("move A\n");
#ifndef __CUDA_ARCH__
		n++;
#endif
	}
	KOISHI_HOST_DEVICE A &operator=( A && ) = default;
	A &operator=( const A & ) = default;
	~A() { n--; }
	
	KOISHI_HOST_DEVICE virtual int f() const
	{
		return 1;
	}

	int x = 1;
	static int n;
};

struct B : A
{
	B() { n++; }
	B( const B & ) { n++; }
	KOISHI_HOST_DEVICE B( B &&other ):
	  A( std::move( other ) )
	{ 
		printf("move B\n");
#ifndef __CUDA_ARCH__
		n++;
#endif
	}
	KOISHI_HOST_DEVICE B &operator=( B && ) = default;
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

TEST( test_poly_object, object_create )
{
	poly::object<A> a = poly::make_object<A>();
	ASSERT_EQ( a->x, 1 );
	a->x = 2;
	ASSERT_EQ( a->x, 2 );
}

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
	const poly::object<A> &a,
	poly::vector<int> &b
)
{
	b[ 0 ] = a->x;
	b[ 1 ] = a->f();
}
#endif

TEST( test_poly_object, object_polymorphism )
{
#ifdef KOISHI_USE_CUDA
	poly::object<A> a = poly::make_object<B>();
	poly::vector<int> b( 2 );
	poly::kernel( g, 1, 1 )( a, b );
	EXPECT_EQ( b[ 0 ], 1 );
	EXPECT_EQ( b[ 1 ], 2 );
#endif
}

