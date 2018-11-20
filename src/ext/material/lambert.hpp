#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct LambertDiffuse : BxDF
{
	KOISHI_HOST_DEVICE LambertDiffuse( const double3 &R ) :
	  R( R )
	{
	}

	KOISHI_HOST_DEVICE double3 f( const double3 &wo, const double3 &wi ) const override
	{
		return R * invPI;
	}

private:
	double3 R;
};

struct LambertMaterial : Material
{
	KOISHI_HOST_DEVICE virtual void fetchTo( Interreact &res, Allocator &pool ) const
	{
		res.bsdf = create<BSDF>( pool );
		res.bsdf->add<LambertDiffuse>( pool, 0.5, );
	}
};

}  // namespace ext

}  // namespace koishi