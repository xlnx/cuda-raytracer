#include <sstream>
#include <core/tracer.hpp>
#include <core/random.hpp>
#include <core/radiance.hpp>
#include <core/renderer.hpp>
#include <vis/renderer.hpp>

using namespace koishi;
using namespace core;

#if 1
struct PolyStruct( A )
{
	Poly( int i ) :
	  n( i )
	{
		PolyVector<int> vv;
		for ( int i = 0; i != n; ++i )
		{
			vv.emplace_back( i );
		}
		v = std::move( vv );
	}
	KOISHI_HOST_DEVICE Poly(Poly &&other) = default;
	Poly(const Poly &other):
	  n( other.n ),
	  v( other.v )
	{
		std::cout << other.n << " " << other.v.size() << " "<< other.v.data() << std::endl;
		std::cout << n << " " <<  v.size() << " " << v.data() << std::endl;
	}

	__host__ __device__ virtual int f()
	{
		int s = 0;
		for ( int i = 0; i != v.size(); ++i )
		{
		//	s += v[ i ];
		}
		return s;
	}

	int n;
	PolyVectorView<int> v;
};

__global__ void add( PolyVectorView<A> vec, PolyVectorView<int*> res, PolyVectorView<int> n )
{
	res[0] = vec[1].v.data();
	n[0] = vec[1].v.size();
}
#endif

int main( int argc, char **argv )
{
#if 1
	PolyVector<A> vec;
	for ( int i = 0; i != 2; ++i )
	{
		vec.emplace_back( i );
	}
	PolyVectorView<A> view = std::move( vec );
	view.emitAndReplace();
	PolyVectorView<int*> res( 3 );
	PolyVectorView<int> nn( 3 );
	res.emitAndReplace();
	nn.emitAndReplace();
	add<<<1, 1>>>( view.forward(), res.forward(), nn.forward() );
	cudaDeviceSynchronize();
	res.fetchAndReplace();
	nn.fetchAndReplace();
	std::cout << res[ 0 ] << " " << nn[0] << std::endl;
	return 0;
#endif

	if ( std::string( argv[ 2 ] ) == "-v" )
	{
	}
	else
	{
		uint spp;
		std::istringstream is( argv[ 3 ] );
		is >> spp;

		using TraceFn = Radiance<DRand48>;
		Renderer<Tracer<TraceFn>> r{ 1024, 768 };

		r.render( argv[ 1 ], argv[ 2 ], spp );
	}
}
