"""TensorRT 10 engine runner with preallocated, max-shape I/O buffers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt  # type: ignore[import-not-found]

from .gpu import DeviceBuffer, PinnedArray, Stream, d2h

log = logging.getLogger(__name__)


class _TrtLogger(trt.ILogger):
    _LEVELS = {
        trt.ILogger.INTERNAL_ERROR: logging.CRITICAL,
        trt.ILogger.ERROR: logging.ERROR,
        trt.ILogger.WARNING: logging.WARNING,
        trt.ILogger.INFO: logging.DEBUG,
        trt.ILogger.VERBOSE: logging.DEBUG,
    }

    def __init__(self) -> None:
        trt.ILogger.__init__(self)

    def log(self, severity, msg):  # noqa: D401 - TensorRT callback
        logging.getLogger("tensorrt").log(self._LEVELS.get(severity, logging.INFO), msg)


_LOGGER = _TrtLogger()
trt.init_libnvinfer_plugins(_LOGGER, "")


@dataclass
class _Tensor:
    name: str
    dtype: np.dtype
    max_shape: tuple[int, ...]
    device: DeviceBuffer
    host: PinnedArray | None = None


class TrtEngine:
    """Runs a serialized engine. Inputs are written straight into device buffers
    (see input_ptr) by the CUDA preprocessing kernels; outputs are copied back
    to pinned host memory."""

    def __init__(self, path: Path, stream: Stream, max_batch: int | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.path}")
        self.stream = stream
        self._runtime = trt.Runtime(_LOGGER)
        self.engine = self._runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"failed to deserialize {self.path}; engines must be built on the same "
                "JetPack/TensorRT version and device they run on"
            )
        self.context = self.engine.create_execution_context()

        self.inputs: dict[str, _Tensor] = {}
        self.outputs: dict[str, _Tensor] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            shape = tuple(self.engine.get_tensor_shape(name))
            if -1 in shape:
                shape = tuple(self.engine.get_tensor_profile_shape(name, 0)[2])
                if max_batch is not None:
                    shape = (min(shape[0], max_batch),) + shape[1:]
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            self.context.set_input_shape(name, shape)
            buf = DeviceBuffer(int(np.prod(shape)) * dtype.itemsize)
            self.inputs[name] = _Tensor(name, dtype, shape, buf)
            self.context.set_tensor_address(name, buf.ptr)

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            buf = DeviceBuffer(int(np.prod(shape)) * dtype.itemsize)
            self.outputs[name] = _Tensor(name, dtype, shape, buf, PinnedArray(shape, dtype))
            self.context.set_tensor_address(name, buf.ptr)

        if len(self.inputs) != 1:
            raise ValueError(f"{self.path.name}: expected exactly one input, got {list(self.inputs)}")
        self.input_name = next(iter(self.inputs))
        self.output_names = list(self.outputs)
        log.info(
            "loaded engine %s inputs=%s outputs=%s",
            self.path.name,
            {n: t.max_shape for n, t in self.inputs.items()},
            {n: t.max_shape for n, t in self.outputs.items()},
        )

    @property
    def input_ptr(self) -> int:
        return self.inputs[self.input_name].device.ptr

    @property
    def input_dtype(self) -> np.dtype:
        return self.inputs[self.input_name].dtype

    @property
    def max_batch(self) -> int:
        return self.inputs[self.input_name].max_shape[0]

    def run(self, batch: int | None = None) -> dict[str, np.ndarray]:
        """Execute on the shared stream and return host copies of the outputs.
        The input buffer must already be populated on the same stream."""
        inp = self.inputs[self.input_name]
        shape = inp.max_shape if batch is None else (batch,) + inp.max_shape[1:]
        if shape[0] > inp.max_shape[0]:
            raise ValueError(f"batch {shape[0]} exceeds engine max {inp.max_shape[0]}")
        self.context.set_input_shape(self.input_name, shape)
        if not self.context.execute_async_v3(self.stream.ptr):
            raise RuntimeError(f"{self.path.name}: execute_async_v3 failed")

        shapes: dict[str, tuple[int, ...]] = {}
        for name, out in self.outputs.items():
            shapes[name] = tuple(self.context.get_tensor_shape(name))
            nbytes = int(np.prod(shapes[name])) * out.dtype.itemsize
            assert out.host is not None
            d2h(out.host.ptr, out.device.ptr, nbytes, self.stream)
        self.stream.synchronize()

        result = {}
        for name, out in self.outputs.items():
            assert out.host is not None
            count = int(np.prod(shapes[name]))
            result[name] = out.host.array.reshape(-1)[:count].reshape(shapes[name]).astype(np.float32)
        return result

    def close(self) -> None:
        for t in list(self.inputs.values()) + list(self.outputs.values()):
            t.device.free()
            if t.host is not None:
                t.host.free()
        self.context = None
        self.engine = None
