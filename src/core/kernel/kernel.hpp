#pragma once

#include <core/basic/basic.hpp>
#include <core/meta/scene.hpp>
#include <core/misc/sampler.hpp>
#include <fstream>

namespace koishi {
namespace core {
struct ProfileSlice : emittable {
    ProfileSlice(const Properties &props) {}

    virtual void writeSlice(std::ostream &os) const {}
    virtual void readSlice(std::istream &is) const {}
};

struct Kernel : emittable {
    Kernel(const Properties &props) {}

    KOISHI_HOST_DEVICE virtual float3 execute(Ray ray, const Scene &scene,
                                              Allocator &pool, Sampler &rng,
                                              ProfileSlice *prof) = 0;

    virtual poly::object<ProfileSlice> profileSlice(
        const Properties &props) const {
        return poly::make_object<ProfileSlice>(props);
    }
};

struct Profiler final : emittable {
    struct Configuration : serializable<Configuration> {
        Property(bool, enable, true);
        Property(std::string, output, "a.prof");
        Property(uint4, area, uint4{0, 0, -1u, -1u});
        Property(Properties, props, {});
    };

    Profiler(const std::string &filename);

    Profiler(const Properties &props, const poly::object<Kernel> &kernel,
             uint w, uint h, uint spp);

    ~Profiler();

    KOISHI_HOST_DEVICE bool enabled() const { return enable; }

    KOISHI_HOST_DEVICE bool enabled(uint x, uint y) const {
        auto pos = uint2{x, y};
        return enable && pos.x >= area.x && pos.x < area.z && pos.y >= area.y &&
               pos.y < area.w;
    }

    KOISHI_HOST_DEVICE ProfileSlice *at(uint x, uint y, uint k) {
        auto pos = uint2{x, y} - uint2{area.x, area.y};
        KASSERT(k < spp);
        if (!enabled(x, y)) return nullptr;
        return &*slices[(areaSize.x * pos.y + pos.x) * spp + k];
    }

    uint4 getArea() const { return area; }

   private:
    bool enable;
    uint spp;
    uint4 area;
    uint2 areaSize;

   private:
    std::string output;
    const Kernel *kernel;
    poly::vector<poly::object<ProfileSlice>> slices;
};

}  // namespace core

}  // namespace koishi
