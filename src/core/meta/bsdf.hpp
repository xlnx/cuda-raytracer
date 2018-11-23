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
	KOISHI_HOST_DEVICE virtual float pdf( const float3 &wo, const float3 &wi ) const  // cosHemisphere
	{
		return hemisphere::isSame( wo, wi ) ? hemisphere::h( wi ) * invPI : 0.;
	}
	KOISHI_HOST_DEVICE virtual float3 sample_f( const float3 &wo, float3 &wi, const float2 &rn, float &pdf ) const
	{
		wi = hemisphere::sampleCos( rn );
		return f( wo, wi );
	}
};

struct BSDF final
{
	template <typename T, typename = typename std::enable_if<
							std::is_base_of<BxDF, T>::value>::type>
	KOISHI_HOST_DEVICE void add( Allocator &pool )
	{
		bxdfs[ numBxdfs++ ] = create<T>( pool );
	}

	KOISHI_HOST_DEVICE float3 f( const float3 &wo, const float3 &wi ) const
	{
		// Vector3f wi = WorldToLocal( wiW ), wo = WorldToLocal( woW );
		// if ( wo.z == 0 ) return 0.;
		// bool reflect = Dot( wiW, ng ) * Dot( woW, ng ) > 0;
		float3 f = { 0, 0, 0 };
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
	KOISHI_HOST_DEVICE float3 sample_f( const float3 &wo, float3 &wi, const float2 &rn, float &pdf ) const
	{
		int comp = min( (int)floor( rn.x * numBxdfs ), (int)numBxdfs - 1 );
		auto bxdf = bxdfs[ comp ];
		auto f = bxdf->sample_f( wo, wi, rn, pdf );
		return f;
	}

public:
	static constexpr uint maxBxdfs = 6;
	BxDF *bxdfs[ maxBxdfs ];
	uint numBxdfs = 0;
};

}  // namespace core

}  // namespace koishi
