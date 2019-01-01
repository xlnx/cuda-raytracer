#pragma once

#include <vector>
#include <string>
#include <iosfwd>
#include <functional>
#include <type_traits>
#include <nlohmann/json.hpp>
#include <vec/vsel.hpp>

namespace koishi
{
using namespace nlohmann;

namespace jsel
{
struct as_object
{
};

struct as_array
{
};

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

template <typename T, typename U = as_object>
struct serializable;

template <typename T>
struct serializable<T, as_object> : __flag::IsSerializable
{
	using type = T;

	serializable()
	{
		switch ( get_state() )
		{
		case 0: get_state() = 2, T(), get_state() = 3; break;
		case 2: get_state() = 1; break;
		}
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
				  j[ name ] = t.*offset;
			  },
			  [=]( const nlohmann::json &j, T &t ) {
				  if ( j.find( name ) != j.end() )
				  {
					  t.*offset = j.at( name ).get<U>();
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
				  j[ name ] = t.*offset;
			  },
			  [=]( const nlohmann::json &j, T &t ) {
				  if ( j.find( name ) != j.end() )
				  {
					  t.*offset = j.at( name ).get<U>();
				  }
				  else
				  {
					  KTHROW( "No such key named \"" + name + "\"." );
				  }
			  } );
		}
		return U();
	}
	template <typename U, typename X>
	static U init_metadata( const std::string &name, U T::*offset, const X &default_value,
							const std::pair<std::function<U( const X & )>,
											std::function<X( const U & )>> &conv )
	{
		if ( get_state() == 1 )  // if this object is the first instance of this class
		{
			get_components().emplace_back(
			  [=]( nlohmann::json &j, const T &t ) {
				  j[ name ] = conv.second( t.*offset );
			  },
			  [=]( const nlohmann::json &j, T &t ) {
				  if ( j.find( name ) != j.end() )
				  {
					  t.*offset = conv.first( j.at( name ).get<X>() );
				  }
				  else
				  {
					  t.*offset = conv.first( default_value );
				  }
			  } );
		}
		return conv.first( default_value );
	}
	template <typename U, typename X>
	static U init_metadata( const std::string &name, U T::*offset,
							const std::pair<std::function<U( const X & )>,
											std::function<X( const U & )>> &conv )
	{
		if ( get_state() == 1 )  // if this object is the first instance of this class
		{
			get_components().emplace_back(
			  [=]( nlohmann::json &j, const T &t ) {
				  j[ name ] = conv.second( t.*offset );
			  },
			  [=]( const nlohmann::json &j, T &t ) {
				  if ( j.find( name ) != j.end() )
				  {
					  t.*offset = conv.first( j.at( name ).get<X>() );
				  }
				  else
				  {
					  KTHROW( "No such key named \"" + name + "\"." );
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

template <typename T>
struct serializable<T, as_array> : __flag::IsSerializable
{
	using type = T;

	serializable()
	{
		switch ( get_state() )
		{
		case 0: get_state() = 2, T(), get_state() = 3; break;
		case 2: get_state() = 1; break;
		}
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
			auto index = get_index();
			get_components().emplace_back(
			  [=]( nlohmann::json &j, const T &t ) {
				  j[ index ] = t.*offset;
			  },
			  [=]( const nlohmann::json &j, T &t ) {
				  if ( j.size() > index )
				  {
					  t.*offset = j.at( index ).get<U>();
				  }
				  else
				  {
					  t.*offset = default_value;
				  }
			  } );
			get_index()++;
		}
		return default_value;
	}
	template <typename U>
	static U init_metadata( const std::string &name, U T::*offset )
	{
		if ( get_state() == 1 )  // if this object is the first instance of this class
		{
			auto index = get_index();
			get_components().emplace_back(
			  [=]( nlohmann::json &j, const T &t ) {
				  j[ index ] = t.*offset;
			  },
			  [=]( const nlohmann::json &j, T &t ) {
				  if ( j.size() > index )
				  {
					  t.*offset = j.at( index ).get<U>();
				  }
				  else
				  {
					  KTHROW( "No such key named \"" + name + "\"." );
				  }
			  } );
			get_index()++;
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
	static int &get_index()
	{
		static int index = 0;
		return index;
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

using jsel::serializable;

using jsel::as_array;
using jsel::as_object;

template <typename T>
inline T get( const nlohmann::json &json, const std::string &key, const T &val )
{
	int start = 0, next;
	auto j = &json;
	do
	{
		next = key.find_first_of( '.', start );
		auto sub = key.substr( start, next );
		if ( j->find( sub ) != j->end() )
		{
			j = &j->at( sub );
		}
		else
		{
			return val;
		}
		start = next + 1;
	} while ( next != key.npos );
	return *j;
}

template <typename T>
inline T get( const nlohmann::json &json, const std::string &key )
{
	int start = 0, next;
	auto j = &json;
	do
	{
		next = key.find_first_of( '.', start );
		auto sub = key.substr( start, next );
		j = &j->at( sub );
		start = next + 1;
	} while ( next != key.npos );
	return *j;
}

}  // namespace koishi
