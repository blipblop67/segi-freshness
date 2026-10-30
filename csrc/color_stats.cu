// HSV colour-defect analysis: one thread block per region of interest.
//
// Each block samples a grid x grid lattice inside its box, classifies every
// sample (dark / brown / mould / yellow) and reduces the counts with warp
// shuffles + shared memory. On Orin Nano this processes ~100 ROIs at a
// 48x48 lattice in well under a millisecond.
#include "common.cuh"
#include "kernels.cuh"

namespace segi {
namespace {

constexpr int kBlockSize = 256;
constexpr int kWarps = kBlockSize / 32;

__device__ __forceinline__ float hue_degrees(float r, float g, float b, float mx, float mn) {
  const float d = mx - mn;
  if (d <= 0.0f) return 0.0f;
  float h;
  if (mx == r) {
    h = 60.0f * ((g - b) / d);
  } else if (mx == g) {
    h = 60.0f * ((b - r) / d + 2.0f);
  } else {
    h = 60.0f * ((r - g) / d + 4.0f);
  }
  return h < 0.0f ? h + 360.0f : h;
}

__global__ void __launch_bounds__(kBlockSize)
    color_stats_kernel(const uint8_t* __restrict__ src, int src_w, int src_h, int src_pitch,
                       int src_channels, const float* __restrict__ boxes, int grid,
                       ColorThresholds t, float* __restrict__ out) {
  const int n = blockIdx.x;
  const float x1 = fmaxf(boxes[4 * n + 0], 0.0f);
  const float y1 = fmaxf(boxes[4 * n + 1], 0.0f);
  const float x2 = fminf(boxes[4 * n + 2], static_cast<float>(src_w));
  const float y2 = fminf(boxes[4 * n + 3], static_cast<float>(src_h));
  const float bw = fmaxf(x2 - x1, 1.0f);
  const float bh = fmaxf(y2 - y1, 1.0f);
  const int total = grid * grid;

  float acc[kNumColorStats] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int i = threadIdx.x; i < total; i += kBlockSize) {
    const int gx = i % grid;
    const int gy = i / grid;
    int px = static_cast<int>(x1 + (static_cast<float>(gx) + 0.5f) * bw / grid);
    int py = static_cast<int>(y1 + (static_cast<float>(gy) + 0.5f) * bh / grid);
    px = min(max(px, 0), src_w - 1);
    py = min(max(py, 0), src_h - 1);

    const float3 bgr = detail::load_bgr(src, src_pitch, src_channels, px, py);
    const float b = bgr.x / 255.0f, g = bgr.y / 255.0f, r = bgr.z / 255.0f;
    const float mx = fmaxf(r, fmaxf(g, b));
    const float mn = fminf(r, fminf(g, b));
    const float v = mx;
    const float s = mx > 0.0f ? (mx - mn) / mx : 0.0f;
    const float h = hue_degrees(r, g, b, mx, mn);

    const bool dark = v < t.dark_v_max;
    const bool brown = !dark && h >= t.brown_h_lo && h <= t.brown_h_hi && s >= t.brown_s_min &&
                       v <= t.brown_v_max;
    const bool mold = !dark && s <= t.mold_s_max && v >= t.mold_v_min;
    const bool yellow =
        !dark && !brown && h >= t.yellow_h_lo && h <= t.yellow_h_hi && s >= t.yellow_s_min;

    acc[kDarkRatio] += dark ? 1.0f : 0.0f;
    acc[kBrownRatio] += brown ? 1.0f : 0.0f;
    acc[kMoldRatio] += mold ? 1.0f : 0.0f;
    acc[kYellowRatio] += yellow ? 1.0f : 0.0f;
    acc[kMeanSaturation] += s;
    acc[kMeanValue] += v;
  }

  __shared__ float partial[kNumColorStats][kWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int k = 0; k < kNumColorStats; ++k) {
    const float s = detail::warp_reduce_sum(acc[k]);
    if (lane == 0) partial[k][warp] = s;
  }
  __syncthreads();

  if (warp == 0) {
    const float inv_total = 1.0f / static_cast<float>(total);
#pragma unroll
    for (int k = 0; k < kNumColorStats; ++k) {
      float s = lane < kWarps ? partial[k][lane] : 0.0f;
      s = detail::warp_reduce_sum(s);
      if (lane == 0) out[n * kNumColorStats + k] = s * inv_total;
    }
  }
}

}  // namespace

cudaError_t color_stats_batch(const uint8_t* src, int src_w, int src_h, int src_pitch,
                              int src_channels, const float* boxes, int n, int grid,
                              const ColorThresholds& t, float* out, cudaStream_t stream) {
  if (n <= 0) return cudaSuccess;
  color_stats_kernel<<<n, kBlockSize, 0, stream>>>(src, src_w, src_h, src_pitch, src_channels,
                                                   boxes, grid, t, out);
  return cudaGetLastError();
}

}  // namespace segi
