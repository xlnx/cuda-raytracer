#pragma once

#include <map>
#include <vector>
#include <string>
#include <vec/vec.hpp>
#include <poly/kernel.hpp>
#include <util/jsel.hpp>

namespace koishi
{
struct CameraConfig : serializable<CameraConfig>, emittable
{
	Property( uint, lens, std::string( "pinhole" ),
			  { []( const std::string &value ) -> uint {
				   if ( value == "pinhole" ) return 0;
				   if ( value == "orthographic" ) return 1;
				   KTHROW( "unknown lens type: " + value );
			   },
				[]( const uint &value ) -> std::string {
					switch ( value )
					{
					case 0: return "pinhole";
					case 1: return "orthographic";
					default: KTHROW( "unknown lens type" );
					}
				} } );
	Property( float, fovx, 90 );
	Property( float, aspect, 0 );
	Property( float3, position, { 1, 0, 0 } );
	Property( float3, target, { -1, 0, 0 } );
	Property( float3, upaxis, { 0, 0, 1 } );
	Property( float, zNear, 1e-3 );
	Property( float, zFar, 1e5 );
};

using Properties = std::map<std::string, json>;

struct Config : serializable<Config, as_array>
{
	Property( std::string, name );
	Property( Properties, props, {} );

	Config() = default;
	Config( const std::string &name, const Properties &props ) :
	  name( name ),
	  props( props )
	{
	}
};

}  // namespace koishi
