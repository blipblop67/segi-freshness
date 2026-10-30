#include "common.cuh"
#include "kernels.cuh"

namespace segi {
namespace {

__global__ void letterbox_kernel(const uint8_t* __restrict__ src, float* __restrict__ dst,
                                 LetterboxParams p) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= p.dst_w || y >= p.dst_h) return;

  float r = p.pad_value, g = p.pad_value, b = p.pad_value;
  const int lx = x - p.pad_x;
  const int ly = y - p.pad_y;
  if (lx >= 0 && lx < p.new_w && ly >= 0 && ly < p.new_h) {
    const float fx = (static_cast<float>(lx) + 0.5f) / p.scale - 0.5f;
    const float fy = (static_cast<float>(ly) + 0.5f) / p.scale - 0.5f;
    const float3 c =
        detail::sample_bilinear(src, p.src_w, p.src_h, p.src_pitch, p.src_channels, fx, fy);
    b = c.x / 255.0f;
    g = c.y / 255.0f;
    r = c.z / 255.0f;
  }

  const size_t plane = static_cast<size_t>(p.dst_w) * p.dst_h;
  const size_t idx = static_cast<size_t>(y) * p.dst_w + x;
  dst[idx] = r;
  dst[plane + idx] = g;
  dst[2 * plane + idx] = b;
}

__global__ void crop_resize_kernel(const uint8_t* __restrict__ src, int src_w, int src_h,
                                   int src_pitch, int src_channels,
                                   const float* __restrict__ boxes, float* __restrict__ dst,
                                   int out_size, float3 mean, float3 inv_std) {
  const int pixels = out_size * out_size;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = blockIdx.y;
  if (i >= pixels) return;

  const int ox = i % out_size;
  const int oy = i / out_size;
  const float x1 = boxes[4 * n + 0];
  const float y1 = boxes[4 * n + 1];
  const float bw = fmaxf(boxes[4 * n + 2] - x1, 1.0f);
  const float bh = fmaxf(boxes[4 * n + 3] - y1, 1.0f);

  const float fx = x1 + (static_cast<float>(ox) + 0.5f) * bw / out_size - 0.5f;
  const float fy = y1 + (static_cast<float>(oy) + 0.5f) * bh / out_size - 0.5f;
  const float3 c = detail::sample_bilinear(src, src_w, src_h, src_pitch, src_channels, fx, fy);

  float* out = dst + static_cast<size_t>(n) * 3 * pixels;
  out[i] = (c.z / 255.0f - mean.x) * inv_std.x;               // R
  out[pixels + i] = (c.y / 255.0f - mean.y) * inv_std.y;      // G
  out[2 * pixels + i] = (c.x / 255.0f - mean.z) * inv_std.z;  // B
}

}  // namespace

cudaError_t letterbox_to_chw(const uint8_t* src, float* dst, const LetterboxParams& p,
                             cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid(detail::div_up(p.dst_w, block.x), detail::div_up(p.dst_h, block.y));
  letterbox_kernel<<<grid, block, 0, stream>>>(src, dst, p);
  return cudaGetLastError();
}

cudaError_t crop_resize_batch(const uint8_t* src, int src_w, int src_h, int src_pitch,
                              int src_channels, const float* boxes, int n, float* dst,
                              int out_size, float3 mean, float3 inv_std, cudaStream_t stream) {
  if (n <= 0) return cudaSuccess;
  const dim3 block(256);
  const dim3 grid(detail::div_up(out_size * out_size, block.x), n);
  crop_resize_kernel<<<grid, block, 0, stream>>>(src, src_w, src_h, src_pitch, src_channels,
                                                 boxes, dst, out_size, mean, inv_std);
  return cudaGetLastError();
}

}  // namespace segi
