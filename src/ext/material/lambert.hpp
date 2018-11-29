#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct LambertDiffuse : BxDF
{
	KOISHI_HOST_DEVICE LambertDiffuse( const float3 &R ) :
	  R( R )
	{
	}

	KOISHI_HOST_DEVICE float3 f( const float3 &wo, const float3 &wi ) const override
	{
		return R * invPI;
	}

private:
	float3 R;
};

struct LambertMaterial : emittable<LambertMaterial, Material>
{
	KOISHI_HOST_DEVICE virtual void fetchTo( Interreact &res, Allocator &pool ) const
	{
		res.bsdf = create<BSDF>( pool );
		res.bsdf->add<LambertDiffuse>( pool, 0.5, );
	}
};

}  // namespace ext

}  // namespace koishi