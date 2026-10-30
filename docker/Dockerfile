# Segi Fresh Monitor - Jetson Orin (JetPack 6.x / L4T r36) container.
# Build ON the Jetson (or with an arm64 builder):
#   docker build -f docker/Dockerfile -t segi-freshness:1.0.0 .
# Match BASE to the device's L4T release (cat /etc/nv_tegra_release).
ARG BASE=nvcr.io/nvidia/l4t-jetpack:r36.4.0

# ---------------------------------------------------------------- build stage
FROM ${BASE} AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-dev python3-pip cmake build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir "pybind11>=2.12,<3"
WORKDIR /src
COPY csrc ./csrc
RUN cmake -S csrc -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=87 \
    && cmake --build build -j"$(nproc)"

# -------------------------------------------------------------- runtime stage
FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-opencv curl \
      gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY segi_freshness ./segi_freshness
COPY --from=build /src/build/_cuda_ops*.so ./segi_freshness/
RUN pip3 install --no-cache-dir ".[jetson]" \
    && python3 -c "import segi_freshness._cuda_ops as m; print('cuda ops ok', m.NUM_COLOR_STATS)"
COPY configs ./configs
COPY scripts ./scripts
COPY samples ./samples

RUN useradd --system --uid 1001 --groups video segi && mkdir -p /app/data /app/models && chown -R segi /app/data
USER segi
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/api/health || exit 1
ENTRYPOINT ["segi-freshness"]
CMD ["run", "-c", "/app/configs/store.yaml"]
