#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct Fresnel : Scala<float>
{
	Fresnel( const Properties &props ) :
	  Scala( props ),
	  ior( get<float>( props, "ior" ) )
	{
	}

	KOISHI_HOST_DEVICE float compute( const Varyings &varyings, Allocator &pool ) const override
	{
		float cosi = H::cosTheta( varyings.wo );
		float e = cosi > 0.f ? 1.f / this->ior : this->ior;
		cosi = fabs( cosi );
		float sini = sqrt( max( 0.f, 1 - cosi * cosi ) );
		float sinr = e * sini;
		if ( sinr >= 1.f ) return 1;
		float cosr = sqrt( max( 0.f, 1 - sinr * sinr ) );
		float cosior = cosi / cosr;
		float rparl = ( cosior - e ) / ( cosior + e );
		float rperp = ( e * cosior - 1 ) / ( e * cosior + 1 );
		return .5f * ( rparl * rparl + rperp * rperp );
	}
	void writeNode( json &j ) const override
	{
		j[ "Fresnel" ] = ior;
	}

private:
	const float ior;
};

}  // namespace ext

}  // namespace koishi