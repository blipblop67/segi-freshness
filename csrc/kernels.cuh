// Segi Fresh Monitor - CUDA kernels for the Jetson Orin Nano (SM 8.7).
//
// All kernels read an interleaved 8-bit BGR or BGRx frame that already lives in
// device memory and write directly into TensorRT input bindings, so a frame is
// uploaded exactly once per inference cycle.
//
// The numerical behaviour of every kernel is mirrored 1:1 by the NumPy
// reference implementation in segi_freshness/reference.py. Keep them in sync:
// tests/test_cuda_parity.py compares the two on the device.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace segi {

struct LetterboxParams {
  int src_w;
  int src_h;
  int src_pitch;     // bytes per source row
  int src_channels;  // 3 (BGR) or 4 (BGRx)
  int dst_w;
  int dst_h;
  float scale;       // dst / src, identical on both axes
  int pad_x;
  int pad_y;
  int new_w;
  int new_h;
  float pad_value;   // already normalised to [0, 1]
};

// Thresholds for the HSV colour-defect analysis. H in degrees [0, 360),
// S and V in [0, 1].
struct ColorThresholds {
  float dark_v_max = 0.20f;     // rot / bruising / deep spots
  float brown_h_lo = 8.0f;      // enzymatic browning
  float brown_h_hi = 35.0f;
  float brown_s_min = 0.25f;
  float brown_v_max = 0.55f;
  float mold_s_max = 0.18f;     // whitish-grey fuzzy mould
  float mold_v_min = 0.65f;
  float yellow_h_lo = 35.0f;    // chlorophyll loss in leafy greens
  float yellow_h_hi = 65.0f;
  float yellow_s_min = 0.30f;
};

// Output layout of color_stats_batch, per box.
enum ColorStat : int {
  kDarkRatio = 0,
  kBrownRatio = 1,
  kMoldRatio = 2,
  kYellowRatio = 3,
  kMeanSaturation = 4,
  kMeanValue = 5,
  kNumColorStats = 6,
};

// Letterbox resize (bilinear) + BGR->RGB + /255 + HWC->CHW, float32 output.
cudaError_t letterbox_to_chw(const uint8_t* src, float* dst, const LetterboxParams& p,
                             cudaStream_t stream);

// Crop N boxes (x1, y1, x2, y2 in source pixels, device float array N x 4) and
// stretch-resize each to out_size x out_size, normalised with (x - mean) / std.
// Output: N x 3 x out_size x out_size float32, RGB planes.
cudaError_t crop_resize_batch(const uint8_t* src, int src_w, int src_h, int src_pitch,
                              int src_channels, const float* boxes, int n, float* dst,
                              int out_size, float3 mean, float3 inv_std, cudaStream_t stream);

// Per-box colour-defect statistics over a grid x grid sample lattice.
// Output: N x kNumColorStats float32 (ratios / means in [0, 1]).
cudaError_t color_stats_batch(const uint8_t* src, int src_w, int src_h, int src_pitch,
                              int src_channels, const float* boxes, int n, int grid,
                              const ColorThresholds& t, float* out, cudaStream_t stream);

}  // namespace segi
