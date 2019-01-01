#include "radiance.hpp"

namespace koishi
{
namespace core
{
KOISHI_HOST_DEVICE float3 RadianceKernel::execute( Ray ray, const Scene &scene, Allocator &pool,
												   Sampler &rng, ProfileSlice *prof )
{
	Varyings varyings;
	float3 L = { 0, 0, 0 }, beta = { 1, 1, 1 };  // brdf can contain 3 components
	auto slice = static_cast<Slice *>( prof );
	constexpr auto maxBounce =  //1;
	  1024;

	int bounce = 0;

	for ( ; scene.intersect( ray, varyings, pool ) &&
			bounce != maxBounce;
		  ++bounce )
	{
		auto &shader = scene.shaders[ varyings.shaderId ];

		// evaluate direct lighting
		if ( scene.lights.size() )
		{
			float3 li;
			uint idx = min( (uint)floor( rng.sample() * scene.lights.size() ),
							( uint )( scene.lights.size() - 1 ) );
			float lpdf = 1.f / scene.lights.size();
			varyings.wi = scene.lights[ idx ]->sample( scene, varyings, rng.sample2(), li, pool );
			shader->execute( varyings, rng, pool, compute_f_by_wi_wo );
			L += beta * varyings.f * li * fabs( dot( varyings.wi, float3{ 0, 0, 1 } ) ) / lpdf;
		}
		L += beta * varyings.emission;
		// emit new light for indirect lighting, according to BSDF
		{
			shader->execute( varyings, rng, pool, sample_wi_f_by_wo );
			beta *= varyings.f * fabs( dot( varyings.wi, float3{ 0, 0, 1 } ) );
			ray = varyings.emitRay( varyings.global( varyings.wi ) );
		}

		if ( slice )
		{
			if ( bounce < slice->bounces.size() )
			{
				auto &b = slice->bounces[ bounce ];
				b.ray = ray;
				b.p = varyings.p;
				b.f = varyings.f;
				b.L = L;
			}
		}

		auto rr = max( beta.x, max( beta.y, beta.z ) );
		if ( rr < 1. && bounce > 3 )
		{
			auto q = max( .05f, 1 - rr );
			if ( rng.sample() < q ) break;
			beta /= 1 - q;
		}

		pool.clear();
		varyings = Varyings();
	}

	if ( slice )
	{
		slice->bounce = bounce;
		slice->L = L;
	}

	return L;
}

}  // namespace core

}  // namespace koishi