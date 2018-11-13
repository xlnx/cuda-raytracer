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
		for ( int i = 0; i <= n; ++i )
		{
			vv.emplace_back( i );
		}
		v = std::move( vv );
	}

	__host__ __device__ virtual int f() const
	{
		int s = v.size() * 1000;
		for ( int i = 0; i != v.size(); ++i )
		{
			s += v[ i ];
		}
		return s;
	}

	int n;
	PolyVectorView<int> v;
};

__global__ void add( const PolyVectorView<A> &vec, PolyVectorView<int> &n, PolyVectorView<const int*> &p )
{
	//n[0] = 1; n[1] = 2;
	//n[0] = 1;
	for (auto i = 0; i != vec.size(); ++i)
		//n[i] = 1;
		n[i] = vec[i].f(), p[i] = vec[i].v.data();
}
#endif

int main( int argc, char **argv )
{
#if 1
	PolyVector<A> vec;
	for ( int i = 0; i != 10; ++i )
	{
		vec.emplace_back( i );
	}
	PolyVectorView<A> view = std::move( vec );

	LOG( view.size(), view.data() );
	
	view.emitAndReplace();
	
	LOG( view.size(), view.data() );
	
	PolyVectorView<int> nn( view.size() );
	PolyVectorView<const int*> pp( view.size() );
	
	LOG(nn.space(), nn.data() );
	LOG(pp.space(), pp.data() );
	
	LOG("size of nn", nn.size());

	nn.emitAndReplace();
	pp.emitAndReplace();

	LOG(nn.space(), nn.data());
	LOG(pp.space(), pp.data());
	
	LOG("size of nn", nn.size());

	kernel(add, 1, 1)(view, nn, pp);
	
	LOG(nn.space(), nn.data());
	LOG(pp.space(), pp.data());

	LOG("size of nn", nn.size());
	
	nn.fetchAndReplace();
	pp.fetchAndReplace();
	
	LOG(nn.space(), nn.data());
	LOG(pp.space(), pp.data());

	LOG("size of nn", nn.size());

	for ( auto &e: nn )
		std::cout << e << std::endl;
	
	LOG( "normal exit" );

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
