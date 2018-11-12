#include <sstream>
#include <core/tracer.hpp>
#include <core/random.hpp>
#include <core/radiance.hpp>
#include <core/renderer.hpp>
#include <vis/renderer.hpp>

using namespace koishi;

#if 0
struct A
{
	A( int i ): n( i ) {}
	__host__ __device__ virtual int f() { return n; }
private:
	int n;
};

__global__ void add( core::PolyVectorView<A> vec, core::PolyVectorView<int> res )
{
	res[0] = 0;
	for (auto &e: vec)
		res[0] += e.f();
}
#endif

int main( int argc, char **argv )
{
#if 0
	core::PolyVector<A> vec;
	for ( int i = 0; i != 10; ++i )
	{
		vec.emplace_back( i );
	}
	core::PolyVectorView<A> view = std::move( vec );
	view.emitAndReplace();
	core::PolyVectorView<int> res(3);
	res.emitAndReplace();
	add<<<1, 1>>>( view.forward(), res.forward() );
	cudaDeviceSynchronize();
	res.fetchAndReplace();
	std::cout << res[0] << std::endl;
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

		using TraceFn = core::Radiance<core::DRand48>;
		core::Renderer<core::Tracer<TraceFn>> r{ 1024, 768 };

		r.render( argv[ 1 ], argv[ 2 ], spp );
	}
}
