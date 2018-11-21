#pragma once

#include <utility>
#include <type_traits>

namespace koishi
{
namespace util
{
namespace __impl
{
struct ArrayImpl;
struct RefImpl;

template <int... I>
struct sequence
{
};

// auxiliary meta-function for making (N+1)-sized sequence
// from an N-sized sequence

template <typename T>
struct append;

template <int... I>
struct append<sequence<I...>>
{
	using type = sequence<I..., sizeof...( I )>;
};

// recursive implementation of make_sequence

template <int I>
struct make_sequence_;

template <int I>
using make_sequence = typename make_sequence_<I>::type;

template <>
struct make_sequence_<0>  // recursion end
{
	using type = sequence<>;
};

template <int I>
struct make_sequence_ : append<make_sequence<I - 1>>
{
	static_assert( I >= 0, "negative size" );
};

template <int N, typename Impl>
struct CString
{
};

template <int N>
using ArrayString = CString<N, ArrayImpl>;

template <int N>
using StringLiteral = CString<N, RefImpl>;

template <int N>
class CString<N, ArrayImpl>
{
	char data[ N + 1 ];

	template <int N1, int... PACK1, int... PACK2>
	constexpr CString( const StringLiteral<N1> &s1,
					   const StringLiteral<N - N1> &s2,
					   sequence<PACK1...>,
					   sequence<PACK2...> ) :
	  data{ s1[ PACK1 ]..., s2[ PACK2 ]..., '\0' }
	{
	}

public:
	template <int N1, REQUIRES( N1 <= N )>
	constexpr CString( const StringLiteral<N1> &s1,
					   const StringLiteral<N - N1> &s2 )
	  // delegate to the other constructor
	  :
	  CString{ s1, s2, make_sequence<N1>{},
			   make_sequence<N - N1>{} }
	{
	}

	constexpr char operator[]( std::size_t i ) const { return data[ i ]; }

	constexpr std::size_t size() const { return N; }
};

template <int N>
struct CString<N, RefImpl>
{
};

}  // namespace __impl

using __impl::ArrayString;
using __impl::CString;
using __impl::StringLiteral;

template <int N1, int N2, typename Tag1, typename Tag2>
constexpr auto operator+( const CString<N1, Tag1> &s1,
						  const CString<N2, Tag2> &s2 )
  -> __impl::ArrayString<N1 + N2>
{
	return __impl::ArrayString<N1 + N2>( s1, s2 );
}

}  // namespace util

}  // namespace koishi