// pybind11 bindings for the Segi CUDA kernels.
//
// Device pointers and CUDA streams cross the boundary as plain integers
// (uintptr_t), so the module interoperates with cuda-python allocations and
// TensorRT bindings without copying. The GIL is released for every launch.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>

#include <stdexcept>
#include <string>

#include "kernels.cuh"

namespace py = pybind11;

namespace {

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(err));
  }
}

template <typename T>
T* as_ptr(uintptr_t p) {
  return reinterpret_cast<T*>(p);
}

cudaStream_t as_stream(uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }

void validate_channels(int channels) {
  if (channels != 3 && channels != 4) {
    throw std::invalid_argument("src_channels must be 3 (BGR) or 4 (BGRx)");
  }
}

}  // namespace

PYBIND11_MODULE(_cuda_ops, m) {
  m.doc() = "Segi Fresh Monitor CUDA preprocessing and colour-analysis kernels";

  py::class_<segi::ColorThresholds>(m, "ColorThresholds")
      .def(py::init<>())
      .def_readwrite("dark_v_max", &segi::ColorThresholds::dark_v_max)
      .def_readwrite("brown_h_lo", &segi::ColorThresholds::brown_h_lo)
      .def_readwrite("brown_h_hi", &segi::ColorThresholds::brown_h_hi)
      .def_readwrite("brown_s_min", &segi::ColorThresholds::brown_s_min)
      .def_readwrite("brown_v_max", &segi::ColorThresholds::brown_v_max)
      .def_readwrite("mold_s_max", &segi::ColorThresholds::mold_s_max)
      .def_readwrite("mold_v_min", &segi::ColorThresholds::mold_v_min)
      .def_readwrite("yellow_h_lo", &segi::ColorThresholds::yellow_h_lo)
      .def_readwrite("yellow_h_hi", &segi::ColorThresholds::yellow_h_hi)
      .def_readwrite("yellow_s_min", &segi::ColorThresholds::yellow_s_min);

  m.attr("NUM_COLOR_STATS") = static_cast<int>(segi::kNumColorStats);

  m.def(
      "letterbox",
      [](uintptr_t src, int src_w, int src_h, int src_pitch, int src_channels, uintptr_t dst,
         int dst_w, int dst_h, float scale, int pad_x, int pad_y, int new_w, int new_h,
         float pad_value, uintptr_t stream) {
        validate_channels(src_channels);
        segi::LetterboxParams p{src_w, src_h, src_pitch, src_channels, dst_w, dst_h,
                                scale, pad_x, pad_y,     new_w,        new_h, pad_value};
        check(segi::letterbox_to_chw(as_ptr<const uint8_t>(src), as_ptr<float>(dst), p,
                                     as_stream(stream)),
              "letterbox");
      },
      py::arg("src"), py::arg("src_w"), py::arg("src_h"), py::arg("src_pitch"),
      py::arg("src_channels"), py::arg("dst"), py::arg("dst_w"), py::arg("dst_h"),
      py::arg("scale"), py::arg("pad_x"), py::arg("pad_y"), py::arg("new_w"), py::arg("new_h"),
      py::arg("pad_value"), py::arg("stream"), py::call_guard<py::gil_scoped_release>());

  m.def(
      "crop_resize",
      [](uintptr_t src, int src_w, int src_h, int src_pitch, int src_channels, uintptr_t boxes,
         int n, uintptr_t dst, int out_size, std::array<float, 3> mean, std::array<float, 3> stdev,
         uintptr_t stream) {
        validate_channels(src_channels);
        const float3 m3 = make_float3(mean[0], mean[1], mean[2]);
        const float3 inv = make_float3(1.0f / stdev[0], 1.0f / stdev[1], 1.0f / stdev[2]);
        check(segi::crop_resize_batch(as_ptr<const uint8_t>(src), src_w, src_h, src_pitch,
                                      src_channels, as_ptr<const float>(boxes), n,
                                      as_ptr<float>(dst), out_size, m3, inv, as_stream(stream)),
              "crop_resize");
      },
      py::arg("src"), py::arg("src_w"), py::arg("src_h"), py::arg("src_pitch"),
      py::arg("src_channels"), py::arg("boxes"), py::arg("n"), py::arg("dst"),
      py::arg("out_size"), py::arg("mean"), py::arg("std"), py::arg("stream"),
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "color_stats",
      [](uintptr_t src, int src_w, int src_h, int src_pitch, int src_channels, uintptr_t boxes,
         int n, int grid, const segi::ColorThresholds& t, uintptr_t out, uintptr_t stream) {
        validate_channels(src_channels);
        if (grid <= 0) throw std::invalid_argument("grid must be positive");
        check(segi::color_stats_batch(as_ptr<const uint8_t>(src), src_w, src_h, src_pitch,
                                      src_channels, as_ptr<const float>(boxes), n, grid, t,
                                      as_ptr<float>(out), as_stream(stream)),
              "color_stats");
      },
      py::arg("src"), py::arg("src_w"), py::arg("src_h"), py::arg("src_pitch"),
      py::arg("src_channels"), py::arg("boxes"), py::arg("n"), py::arg("grid"),
      py::arg("thresholds"), py::arg("out"), py::arg("stream"),
      py::call_guard<py::gil_scoped_release>());
}
