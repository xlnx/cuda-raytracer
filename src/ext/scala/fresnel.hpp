#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct Fresnel : Scala<float>
{
	Fresnel( const Properties &props ) :
	  ior( 1.f / get<float>( props, "ior" ) )
	{
	}

	KOISHI_HOST_DEVICE float compute( const Input &input, Allocator &pool ) const override
	{
		float cosi = H::cosTheta( input.wo );
		float ior = this->ior;
		if ( cosi <= 0.f )
		{
			cosi = -cosi;
			ior = 1.f / ior;
		}
		// KLOG( cosi );
		// KLOG( ior );
		float sini = sqrt( max( 0.f, 1 - cosi * cosi ) );
		float sinr = ior * sini;
		if ( sinr >= 1.f )
		{
			return 1;
		}
		float cosr = sqrt( max( 0.f, 1 - sinr * sinr ) );
		float cosior = cosi / cosr;
		float rparl = ( cosior - ior ) / ( cosior + ior );
		float rperp = ( ior * cosior - 1 ) / ( ior * cosior + 1 );
		return .5f * ( rparl * rparl + rperp * rperp );
	}
	void print( std::ostream &os ) const override
	{
		nlohmann::json json = {
			{ "Fresnel", {} }
		};
		os << json.dump();
	}

private:
	const float ior;
};

}  // namespace ext

}  // namespace koishi