"""Thin, explicit wrappers over the CUDA runtime (cuda-python).

Device memory is owned here and handed to TensorRT and to the custom kernels
in segi_freshness._cuda_ops as raw integer pointers.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

try:  # cuda-python >= 12.6 layout
    from cuda.bindings import runtime as cudart  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - older cuda-python
    from cuda import cudart  # type: ignore[import-not-found,no-redef]


class CudaError(RuntimeError):
    pass


def check(result: tuple[Any, ...]) -> Any:
    err, *rest = result
    if err != cudart.cudaError_t.cudaSuccess:
        _, name = cudart.cudaGetErrorName(err)
        raise CudaError(f"CUDA error {int(err)}: {name.decode() if isinstance(name, bytes) else name}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else tuple(rest)


def device_name(index: int = 0) -> str:
    props = check(cudart.cudaGetDeviceProperties(index))
    name = props.name
    return name.decode(errors="replace").rstrip("\x00") if isinstance(name, bytes) else str(name)


class Stream:
    def __init__(self) -> None:
        self.handle = check(cudart.cudaStreamCreate())

    @property
    def ptr(self) -> int:
        return int(self.handle)

    def synchronize(self) -> None:
        check(cudart.cudaStreamSynchronize(self.handle))

    def close(self) -> None:
        if self.handle is not None:
            check(cudart.cudaStreamDestroy(self.handle))
            self.handle = None


class DeviceBuffer:
    """Growable device allocation. ensure() never shrinks."""

    def __init__(self, nbytes: int = 0) -> None:
        self.ptr = 0
        self.nbytes = 0
        if nbytes:
            self.ensure(nbytes)

    def ensure(self, nbytes: int) -> int:
        if nbytes > self.nbytes:
            self.free()
            self.ptr = int(check(cudart.cudaMalloc(nbytes)))
            self.nbytes = nbytes
        return self.ptr

    def free(self) -> None:
        if self.ptr:
            check(cudart.cudaFree(self.ptr))
            self.ptr = 0
            self.nbytes = 0


class PinnedArray:
    """Page-locked host array, required for truly asynchronous copies."""

    def __init__(self, shape: tuple[int, ...], dtype: Any) -> None:
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.ptr = int(check(cudart.cudaMallocHost(max(self.nbytes, 1))))
        raw = (ctypes.c_byte * max(self.nbytes, 1)).from_address(self.ptr)
        self.array = np.frombuffer(raw, dtype=self.dtype, count=int(np.prod(self.shape))).reshape(self.shape)

    def free(self) -> None:
        if self.ptr:
            self.array = None  # type: ignore[assignment]
            check(cudart.cudaFreeHost(self.ptr))
            self.ptr = 0


def h2d(dst: int, src: int, nbytes: int, stream: Stream) -> None:
    check(cudart.cudaMemcpyAsync(dst, src, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream.handle))


def d2h(dst: int, src: int, nbytes: int, stream: Stream) -> None:
    check(cudart.cudaMemcpyAsync(dst, src, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream.handle))
