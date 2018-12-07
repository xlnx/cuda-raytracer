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

	KOISHI_HOST_DEVICE virtual float3 f( const float3 &wo, const float3 &wi ) const = 0;
	KOISHI_HOST_DEVICE virtual float3 sample( const float3 &wo, const float3 &rn, float &pdf ) const
	{
		auto wi = hemisphere::sampleCos( float2{ rn.x, rn.y } );
		pdf = hemisphere::isSame( wo, wi ) ? hemisphere::h( wi ) * invPI : 0.;
		return wi;
	}
	KOISHI_HOST_DEVICE float3 sample( const float3 &wo, const float2 &rn, float &pdf ) const
	{
		return sample( wo, float3{ rn.x, rn.y, 0.f }, pdf );
	}
};

struct BSDF final
{
	template <typename T, typename... Args, typename = typename std::enable_if<std::is_base_of<BxDF, T>::value>::type>
	KOISHI_HOST_DEVICE void add( Allocator &pool, Args &&... args )
	{
		bxdfs[ numBxdfs++ ] = create<T>( pool, std::forward<Args>( args )... );
	}

	KOISHI_HOST_DEVICE float3 f( const float3 &wo, const float3 &wi ) const
	{
		// Vector3f wi = WorldToLocal( wiW ), wo = WorldToLocal( woW );
		// if ( wo.z == 0 ) return 0.;
		// bool reflect = Dot( wiW, ng ) * Dot( woW, ng ) > 0;
		float3 f = { 0, 0, 0 };
		// float3 f = reflect(wo, )
		// for ( int i = 0; i < nBxdfs; ++i )
		// 	if ( bxdfs[ i ]->MatchesFlags( flags ) &&
		// 		 ( ( reflect && ( bxdfs[ i ]->type & BSDF_REFLECTION ) ) ||
		// 		   ( !reflect && ( bxdfs[ i ]->type & BSDF_TRANSMISSION ) ) ) )
		// 		f += bxdfs[ i ]->f( wo, wi );
		return f;
	}
	KOISHI_HOST_DEVICE float pdf( const float3 &wo, const float3 &wi ) const
	{
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
