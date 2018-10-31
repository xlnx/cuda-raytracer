#pragma once

#include <vector>
#include <string>
#include <initializer_list>
#include <vec/vec.hpp>
#include <vec/vmath.hpp>

namespace koishi
{
namespace util
{
namespace __com
{
template <uint Channel>
struct Component : Component<Channel - 1>
{
	constexpr Component( const std::initializer_list<unsigned char> &l )
	{
		unsigned char *p = reinterpret_cast<unsigned char *>( this );
		for ( auto c : l ) *p++ = c;
	}
	constexpr Component( const std::initializer_list<float> &l )
	{
		unsigned char *p = reinterpret_cast<unsigned char *>( this );
		for ( auto c : l ) *p++ = min( max( c, 0.f ), 1.f ) * 255;
	}
	constexpr Component() = default;
	unsigned char &operator[]( uint n )
	{
		return reinterpret_cast<unsigned char *>( this )[ n ];
	}

private:
	unsigned char value;
};
template <>
struct Component<0>
{
};

}  // namespace __com

namespace __impl
{
int write_image( const std::string &path, uint w, uint h, uint channel, const unsigned char *data );

}

template <uint Channel>
class Image
{
public:
	using value_type = __com::Component<Channel>;

	Image( uint w, uint h ) :
	  w( w ), h( h ), value( w * h * Channel )
	{
	}

	value_type &at( uint x, uint y )
	{
		return *reinterpret_cast<value_type *>( &value[ ( x + y * w ) * Channel ] );
	}
	unsigned char *data()
	{
		return &value[ 0 ];
	}
	int dump( const std::string &path )
	{
		return __impl::write_image( path, w, h, Channel, data() );
	}

private:
	uint w, h;
	std::vector<unsigned char> value;
};

}  // namespace util

}  // namespace koishi