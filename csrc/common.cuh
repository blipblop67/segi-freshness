#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace segi {
namespace detail {

// Returns (b, g, r) as floats in [0, 255].
__device__ __forceinline__ float3 load_bgr(const uint8_t* src, int pitch, int channels, int x,
                                           int y) {
  const uint8_t* p = src + static_cast<size_t>(y) * pitch + static_cast<size_t>(x) * channels;
  return make_float3(p[0], p[1], p[2]);
}

// Bilinear sample with edge clamping. Matches reference._bilinear exactly.
__device__ __forceinline__ float3 sample_bilinear(const uint8_t* src, int w, int h, int pitch,
                                                  int channels, float fx, float fy) {
  fx = fminf(fmaxf(fx, 0.0f), static_cast<float>(w - 1));
  fy = fminf(fmaxf(fy, 0.0f), static_cast<float>(h - 1));
  const int x0 = static_cast<int>(floorf(fx));
  const int y0 = static_cast<int>(floorf(fy));
  const int x1 = min(x0 + 1, w - 1);
  const int y1 = min(y0 + 1, h - 1);
  const float ax = fx - static_cast<float>(x0);
  const float ay = fy - static_cast<float>(y0);

  const float3 p00 = load_bgr(src, pitch, channels, x0, y0);
  const float3 p01 = load_bgr(src, pitch, channels, x1, y0);
  const float3 p10 = load_bgr(src, pitch, channels, x0, y1);
  const float3 p11 = load_bgr(src, pitch, channels, x1, y1);

  float3 top, bot, out;
  top.x = p00.x * (1.0f - ax) + p01.x * ax;
  top.y = p00.y * (1.0f - ax) + p01.y * ax;
  top.z = p00.z * (1.0f - ax) + p01.z * ax;
  bot.x = p10.x * (1.0f - ax) + p11.x * ax;
  bot.y = p10.y * (1.0f - ax) + p11.y * ax;
  bot.z = p10.z * (1.0f - ax) + p11.z * ax;
  out.x = top.x * (1.0f - ay) + bot.x * ay;
  out.y = top.y * (1.0f - ay) + bot.y * ay;
  out.z = top.z * (1.0f - ay) + bot.z * ay;
  return out;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

inline unsigned int div_up(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

}  // namespace detail
}  // namespace segi
