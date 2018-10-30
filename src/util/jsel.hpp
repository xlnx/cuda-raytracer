#pragma once

#include <vector>
#include <string>
#include <iosfwd>
#include <functional>
#include <type_traits>
#include <nlohmann/json.hpp>

namespace koishi
{
namespace jsel
{
namespace __flag
{
struct IsSerializable
{
};

}  // namespace __flag

template <typename T, typename = typename std::enable_if<
						std::is_base_of<__flag::IsSerializable, T>::value>::type>
inline void to_json( nlohmann::json &j, const T &e )
{
	e.serialize( j );
}

template <typename T, typename = typename std::enable_if<
						std::is_base_of<__flag::IsSerializable, T>::value>::type>
inline void from_json( const nlohmann::json &j, T &e )
{
	e.deserialize( j );
}

#define Property( T, name, ... ) \
	T name = init_metadata<T>( #name, &type::name, ##__VA_ARGS__ )  // if no init value the comma will be removed.

#define Serializable( type ) \
	type:                    \
	Serializer<type>

template <typename T>
struct Serializer : __flag::IsSerializable
{
	using type = T;

	Serializer()
	{
		get_state() = get_state() == 0 ? 1 : 2;
	}
	void serialize( nlohmann::json &j ) const
	{
		j = nlohmann::json{};
		for ( auto &e : get_components() )
			e.first( j, static_cast<const T &>( *this ) );
	}
	void deserialize( const nlohmann::json &j )
	{
		for ( auto &e : get_components() )
			e.second( j, static_cast<T &>( *this ) );
	}

protected:
	template <typename U>
	static U init_metadata( const std::string &name, U T::*offset, const U &default_value )
	{
		if ( get_state() == 1 )  // if this object is the first instance of this class
		{
			get_components().emplace_back(
			  [=]( nlohmann::json &j, const T &t ) {
				  j.at( name ) = t.*offset;
			  },
			  [=]( const nlohmann::json &j, T &t ) {
				  if ( j.find( name ) != j.end() )
				  {
					  j.at( name ).get_to( t.*offset );
				  }
				  else
				  {
					  t.*offset = default_value;
				  }
			  } );
		}
		return default_value;
	}
	template <typename U>
	static U init_metadata( const std::string &name, U T::*offset )
	{
		if ( get_state() == 1 )  // if this object is the first instance of this class
		{
			get_components().emplace_back(
			  [=]( nlohmann::json &j, const T &t ) {
				  j.at( name ) = t.*offset;
			  },
			  [=]( const nlohmann::json &j, T &t ) {
				  if ( j.find( name ) != j.end() )
				  {
					  j.at( name ).get_to( t.*offset );
				  }
				  else
				  {
					  throw "No such key named \"" + name + "\".";
				  }
			  } );
		}
		return U();
	}

private:
	static std::vector<std::pair<
	  std::function<void( nlohmann::json &, const T & )>,
	  std::function<void( const nlohmann::json &, T & )>>> &
	  get_components()
	{
		static std::vector<std::pair<
		  std::function<void( nlohmann::json &, const T & )>,
		  std::function<void( const nlohmann::json &, T & )>>>
		  val;
		return val;
	}
	static int &get_state()
	{
		static int state = 0;
		return state;
	}
};

template <typename T, typename = typename std::enable_if<
						std::is_base_of<__flag::IsSerializable, T>::value>::type>
inline std::istream &operator>>( std::istream &is, T &t )
{
	nlohmann::json data;
	is >> data, t = data;
	return is;
}

}  // namespace jsel

}  // namespace koishi
