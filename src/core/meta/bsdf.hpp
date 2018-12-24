#pragma once

#include <type_traits>
#include <vec/vmath.hpp>
#include <util/hemisphere.hpp>
#include <core/basic/allocator.hpp>

namespace koishi
{
namespace core
{
struct BxDF
{
	KOISHI_HOST_DEVICE virtual ~BxDF() = default;

	KOISHI_HOST_DEVICE virtual float3 f( const solid &wo, const solid &wi ) const = 0;
	KOISHI_HOST_DEVICE virtual solid sample( const solid &wo, const float3 &rn, float3 &f ) const = 0;
	KOISHI_HOST_DEVICE solid sample( const solid &wo, const float2 &rn, float3 &f ) const
	{
		return sample( wo, float3{ rn.x, rn.y, 0.f }, f );
	}
};

}  // namespace core

}  // namespace koishi
