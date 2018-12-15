#include <gtest/gtest.h>
#include <core/basic/ray.hpp>

using namespace koishi;
using namespace core;

TEST( test_intersect, segment )
{
	float3 vmin{ 0, 0, 0 }, vmax{ 1, 1, 1 };
	Seg a;
	a.o = { 2, 0, 0 };
	a.d = { -1, 0, 1 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), true );
	a.o = { 2, 0, 0 };
	a.d = { -1, 0, 1.01 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), false );
	a.o = { 2, 0, 0 };
	a.d = { -1.01, 0, 1 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), true );
	a.o = { 2, 0, 0 };
	a.d = { -0.99, 0, 1 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), false );
	a.o = { 2, 2, 0 };
	a.d = { -1, -1, 1 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), true );
	a.o = { 2, 2, 0 };
	a.d = { -1, -1, 1.01 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), false );
	a.o = { 2, 2, 0 };
	a.d = { -1, -1, 0 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), true );
	a.o = { 2, 2, 0 };
	a.d = { -0.99, -0.99, 0 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), false );
	a.o = { 0, 0, 2 };
	a.d = { 0, 0, -2 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), true );
	a.o = { 0, 0, 2 };
	a.d = { 0, 0, -1 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), true );
	a.o = { 0, 0, 2 };
	a.d = { 0, 0, -0.99 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), false );
	a.o = { 0, 0, 1e5 };
	a.d = { 0, 0, -2e5 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), true );
	a.o = { 0.5, 0.5, 1e5 };
	a.d = { 0, 0, -2e5 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), true );
	a.o = { 0.5, 0.5, 0.5 };
	a.d = { 0, 0, 0.1 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), true );
	a.o = { 1e-5, 1e-5, 1e-5 };
	a.d = { 1 - 2e-5, 1 - 2e-5, 1 - 2e-5 };
	EXPECT_EQ( a.intersect_bbox( vmin, vmax ), true );
}