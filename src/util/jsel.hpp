#pragma once

#include <vector>
#include <string>
#include <iosfwd>
#include <functional>
#include <type_traits>
#include <json/json.h>
#include <vec/trait.hpp>

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

namespace __trait
{
template <typename T, typename = void>
struct to_json_assignable
{
	static constexpr bool value = false;
};

template <typename T>
struct to_json_assignable<T, trait::void_t<decltype( std::declval<Json::Value &>() = std::declval<const T &>() )>>
{
	static constexpr bool value = true;
};

template <typename T, typename = void>
struct from_json_assignable
{
	static constexpr bool value = false;
};

template <typename T>
struct from_json_assignable<T, trait::void_t<decltype( std::declval<const T &>() = std::declval<Json::Value &>() )>>
{
	static constexpr bool value = true;
};

template <typename V>
struct is_vector
{
	static constexpr bool value = false;
};

template <typename T, typename U>
struct is_vector<std::vector<T, U>>
{
	static constexpr bool value = true;
};

}  // namespace __trait

template <typename T, typename = void, typename = void,
		  typename = typename std::enable_if<__trait::to_json_assignable<T>::value>::type>
inline void to_json( Json::Value &j, const T &e )
{
	j = e;
}

template <typename T, typename = void, typename = void,
		  typename = typename std::enable_if<std::is_same<T, bool>::value>::type>
inline void from_json( const Json::Value &j, T &e )
{
	e = j.asBool();
}

template <typename T, typename = void, typename = void, typename = void,
		  typename = typename std::enable_if<std::is_integral<T>::value &&
											 !std::is_same<T, bool>::value &&
											 std::is_signed<T>::value>::type>
inline void from_json( const Json::Value &j, T &e )
{
	e = j.asInt64();
}

template <typename T, typename = void, typename = void, typename = void, typename = void,
		  typename = typename std::enable_if<std::is_integral<T>::value &&
											 !std::is_same<T, bool>::value &&
											 !std::is_signed<T>::value>::type>
inline void from_json( const Json::Value &j, T &e )
{
	e = j.asUInt64();
}

template <typename T, typename = void, typename = void, typename = void, typename = void, typename = void,
		  typename = typename std::enable_if<std::is_same<T, std::string>::value>::type>
inline void from_json( const Json::Value &j, T &e )
{
	e = j.asString();
}

template <typename T, typename = void, typename = void, typename = void, typename = void,
		  typename = void, typename = void,
		  typename = typename std::enable_if<std::is_arithmetic<T>::value &&
											 !std::is_integral<T>::value>::type>
inline void from_json( const Json::Value &j, T &e )
{
	e = T( j.asDouble() );
}

template <typename T, typename = void, typename = typename std::enable_if<__trait::is_vector<T>::value>::type>
inline void to_json( Json::Value &j, const T &e )
{
	j = Json::arrayValue;
	for ( uint i = 0; i != e.size(); ++i )
	{
		to_json( j[ i ], e[ i ] );
	}
}

template <typename T, typename = void, typename = typename std::enable_if<__trait::is_vector<T>::value>::type>
inline void from_json( const Json::Value &j, T &e )
{
	e.resize( j.size() );
	for ( uint i = 0; i != e.size(); ++i )
	{
		from_json( j[ i ], e[ i ] );
	}
}

template <typename T, typename = typename std::enable_if<
						std::is_base_of<__flag::IsSerializable, T>::value>::type>
inline void to_json( Json::Value &j, const T &e )
{
	e.serialize( j );
}

template <typename T, typename = typename std::enable_if<
						std::is_base_of<__flag::IsSerializable, T>::value>::type>
inline void from_json( const Json::Value &j, T &e )
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
	void serialize( Json::Value &j ) const
	{
		j = Json::Value{};
		for ( auto &e : get_components() )
			e.first( j, static_cast<const T &>( *this ) );
	}
	void deserialize( const Json::Value &j )
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
			  [=]( Json::Value &j, const T &t ) {
				  to_json( j[ name ], t.*offset );
			  },
			  [=]( const Json::Value &j, T &t ) {
				  if ( j[ name ] )
				  {
					  from_json( j[ name ], t.*offset );
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
			  [=]( Json::Value &j, const T &t ) {
				  to_json( j[ name ], t.*offset );
			  },
			  [=]( const Json::Value &j, T &t ) {
				  if ( j[ name ] )
				  {
					  from_json( j[ name ], t.*offset );
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
	  std::function<void( Json::Value &, const T & )>,
	  std::function<void( const Json::Value &, T & )>>> &
	  get_components()
	{
		static std::vector<std::pair<
		  std::function<void( Json::Value &, const T & )>,
		  std::function<void( const Json::Value &, T & )>>>
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
	Json::Value data;
	is >> data;
	from_json( data, t );
	return is;
}

}  // namespace jsel

}  // namespace koishi
