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

struct BSDF final
{
	template <typename T, typename... Args, typename = typename std::enable_if<std::is_base_of<BxDF, T>::value>::type>
	KOISHI_HOST_DEVICE void add( Allocator &pool, Args &&... args )
	{
		bxdfs[ numBxdfs++ ] = create<T>( pool, std::forward<Args>( args )... );
	}

	KOISHI_HOST_DEVICE BxDF *sampleBxDF( float rn )
	{
		int comp = min( (int)floor( rn * numBxdfs ), (int)numBxdfs - 1 );
		return bxdfs[ comp ];
	}

public:
	static constexpr uint maxBxdfs = 6;
	BxDF *bxdfs[ maxBxdfs ];
	uint numBxdfs = 0;
};

}  // namespace core

}  // namespace koishi
