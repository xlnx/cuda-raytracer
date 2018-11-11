#include <sstream>
#include <core/tracer.hpp>
#include <core/random.hpp>
#include <core/radiance.hpp>
#include <core/renderer.hpp>
#include <vis/renderer.hpp>

using namespace koishi;

__global__ void add( core::PolyVectorView<core::PolyVectorView<int>> vec, int *sum, int **data)
{
	*sum = 0;
	*sum = vec.size();
	for (auto &e: vec)
	for (auto &f: e)
		*sum += f;
}

int main( int argc, char **argv )
{
	core::PolyVector<int> vec;
	for (int i = 0; i != 10; ++i)
	{
		vec.emplace_back(i);
	}
	core::PolyVector<core::PolyVectorView<int>> vv;
	for (int i = 0; i != 10; ++i)
	{
		vv.emplace_back(std::move(core::PolyVector<int>(vec)));
	}
	std::cout << 1 << std::endl;
	core::PolyVectorView<core::PolyVectorView<int>> view = std::move( vv );
	std::cout << 2 << std::endl;
	auto gpuView = std::move(view.emit());
	int *p, **pp;
	std::cout << 4 << std::endl;
	cudaMallocManaged(&p, sizeof(*p));
	cudaMallocManaged(&pp, sizeof(*pp));
	std::cout << 5 << std::endl;
	add<<<1, 1>>>(gpuView.forward(), p, pp);
	std::cout << 6 << std::endl;
	cudaDeviceSynchronize();
	std::cout << *p << " " << *pp << std::endl;
	return 0;
//	if ( std::string( argv[ 2 ] ) == "-v" )
//	{
//	}
//	else
	{
		uint spp;
		std::istringstream is( argv[ 3 ] );
		is >> spp;

		using TraceFn = core::Radiance<core::DRand48>;
		core::Renderer<core::Tracer<TraceFn>> r{ 1024, 768 };

		r.render( argv[ 1 ], argv[ 2 ], spp );
	}
}
