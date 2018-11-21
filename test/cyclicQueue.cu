#include <core/basic/allocator.hpp>
#include <core/basic/queue.hpp>
#include <util/debug.hpp>
#include <gtest/gtest.h>

using namespace koishi;
using namespace core;

#ifdef KOISHI_USE_CUDA

#endif

TEST( test_cyclic_queue, push_queue )
{
	HostAllocator al;
	CyclicQueue<int> q( al );
	KLOG( q.capacity() );
	q.emplace( 1 );
	q.emplace( 2 );
	ASSERT_EQ( q.front(), 1 );
	q.pop();
	ASSERT_EQ( q.front(), 2 );
}

TEST( test_cyclic_queue, overflow )
{
	HostAllocator al;
	CyclicQueue<int> q( al );
	for ( auto i = 0; i != q.capacity(); ++i )
	{
		ASSERT_EQ( q.overflow(), false );
		q.emplace( i );
	}
	ASSERT_EQ( q.overflow(), true );
	q.emplace( 1 );
	ASSERT_EQ( q.overflow(), true );
	q.emplace( 1 );
	ASSERT_EQ( q.overflow(), true );
}

TEST( test_cyclic_queue, huge_amount_of_data )
{
	HostAllocator al;
	CyclicQueue<int> q( al );
	for ( auto i = 0; i != 1 << 24; ++i )
	{
		q.emplace( i );
	}
}