#pragma once

#include <core/basic/basic.hpp>
#include "kernel.hpp"

namespace koishi {
namespace core {
struct CustomKernel : Kernel {
    CustomKernel(const Properties &props) : Kernel(props) {}

    KOISHI_HOST_DEVICE float3 execute(Ray ray, const Scene &scene,
                                      Allocator &pool, Sampler &rng,
                                      ProfileSlice *prof) override {
        float3 L = {0, 0, 0};

        Varyings varyings;

        if (scene.intersect(ray, varyings, pool)) {
            auto &shader = scene.shaders[varyings.shaderId];

            float3 li;
            uint idx = 0;
            // min( (uint)floor( rng.sample() * scene.lights.size() ),
            // 				( uint )( scene.lights.size() - 1 ) );
            varyings.wi = scene.lights[idx]->sample(scene, varyings,
                                                    rng.sample2(), li, pool);
            shader->execute(varyings, rng, pool, compute_f_by_wi_wo);
            auto ndl = fabs(dot(varyings.wi, float3{0, 0, 1}));
            auto f = varyings.f * ndl;

            // L = float3{f0.x, f0.y * ndl, ndl};
            L = f;
            // L = float3{ f.x, li.y * f.y, li.z };
            //   normalize( varyings.wo + wi );
            // varyings.bxdf->f( varyings.wo, wi );
            // float3{ 1, 1, 1 } * fabs( dot( wi, float3{ 0, 0, 1 } ) );
            // li;
            // L = li;
        }
        pool.clear();

        return L;
    }
};

}  // namespace core

}  // namespace koishi
