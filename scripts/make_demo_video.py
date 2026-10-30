"""Generate a synthetic CCTV-style wet-market clip for demos without a camera.

The clip imitates a ceiling-mounted store CCTV camera looking down at four
display bins (tomato, banana, sawi, carrot) at an angle:

  * perspective view, so the bins appear as trapezoids (polygon zones),
  * timestamp + camera-name overlay, lens vignetting, sensor noise, light flicker,
  * H.264 / yuv420p encoding at CCTV-like bitrates (when ffmpeg is available).

Timeline (default 120 s):
  0-15 s    fresh stock
  15-95 s   tomato bin develops brown/black rot, sawi leaves turn yellow
  95-106 s  spoiled stock left on display (alerts should be active)
  106-108 s a staff member reaches over the bins (brief occlusion)
  108-120 s bins restocked with fresh produce (alerts should clear)

    python scripts/make_demo_video.py                # -> samples/wetmarket_cctv_demo.mp4
    python scripts/make_demo_video.py --print-zones  # zone polygons for the config

ffmpeg is taken from PATH, or from the imageio-ffmpeg package if installed;
without either, OpenCV's mp4v encoder is used (plays in VLC, not in browsers).
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

W, H = 1920, 1080
CAMERA_NAME = "SEGI-WM-CAM01"
BINS = {  # normalised rects in the top-down layout
    "tomato": (0.03, 0.06, 0.48, 0.47),
    "banana": (0.52, 0.06, 0.97, 0.47),
    "sawi": (0.03, 0.53, 0.48, 0.94),
    "carrot": (0.52, 0.53, 0.97, 0.94),
}
# Where the top-down layout's corners land in the camera image (far edge narrower).
_VIEW = np.float32([[0.15 * W, 0.13 * H], [0.85 * W, 0.13 * H], [0.99 * W, 0.97 * H], [0.01 * W, 0.97 * H]])
HOMOGRAPHY = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), _VIEW)


def zone_polygons() -> dict[str, list[tuple[float, float]]]:
    """Each bin's outline in normalised camera-image coordinates."""
    out = {}
    for name, (x1, y1, x2, y2) in BINS.items():
        pts = np.float32([[[x1 * W, y1 * H]], [[x2 * W, y1 * H]], [[x2 * W, y2 * H]], [[x1 * W, y2 * H]]])
        warped = cv2.perspectiveTransform(pts, HOMOGRAPHY).reshape(-1, 2)
        out[name] = [(round(float(x) / W, 3), round(float(y) / H, 3)) for x, y in warped]
    return out


def _px(rect):
    x1, y1, x2, y2 = rect
    return int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)


def _items(rng, rect, n, r_lo, r_hi):
    x1, y1, x2, y2 = _px(rect)
    return [(int(rng.integers(x1 + r_hi, x2 - r_hi)), int(rng.integers(y1 + r_hi, y2 - r_hi)),
             int(rng.integers(r_lo, r_hi)), float(rng.uniform(0, 180))) for _ in range(n)]


def build_scene(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    tx1, ty1, tx2, ty2 = _px(BINS["tomato"])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    return {
        "tomato": _items(rng, BINS["tomato"], 260, 34, 48),
        "banana": _items(rng, BINS["banana"], 110, 30, 36),
        "sawi": _items(rng, BINS["sawi"], 170, 50, 75),
        "carrot": _items(rng, BINS["carrot"], 380, 14, 18),
        "spots": [(int(rng.integers(tx1, tx2)), int(rng.integers(ty1, ty2)), int(rng.integers(14, 36)),
                   float(rng.uniform(0.0, 0.8))) for _ in range(130)],
        "vignette": (1.0 - 0.35 * np.clip(r - 0.35, 0, None) ** 1.5)[..., None].astype(np.float32),
        "noise": [rng.normal(0, 3.5, (H, W, 1)).astype(np.float32) for _ in range(6)],
    }


def render_topdown(scene: dict, spoil: float) -> np.ndarray:
    """Top-down bins. spoil in [0, 1] is rot / yellowing progress."""
    img = np.full((H, W, 3), (58, 62, 66), np.uint8)  # concrete floor
    for rect in BINS.values():
        x1, y1, x2, y2 = _px(rect)
        cv2.rectangle(img, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (35, 95, 160), -1)  # wooden crate
        cv2.rectangle(img, (x1, y1), (x2, y2), (30, 40, 30), -1)

    for x, y, r, _ in scene["tomato"]:
        cv2.circle(img, (x, y), r, (30, 30, 205), -1, cv2.LINE_AA)
        cv2.circle(img, (x - r // 3, y - r // 3), r // 4, (140, 140, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 5, (40, 120, 40), -1)
    for x, y, r, onset in scene["spots"]:  # rot appears at its onset and grows
        if spoil > onset:
            k = min(1.0, (spoil - onset) / 0.3)
            cv2.circle(img, (x, y), int(r * (0.4 + 0.8 * k)), (20, 45, 75), -1, cv2.LINE_AA)  # brown
            cv2.circle(img, (x, y), int(r * 0.45 * k), (15, 18, 22), -1, cv2.LINE_AA)        # black core

    for x, y, r, ang in scene["banana"]:
        cv2.ellipse(img, (x, y), (r * 3, r), ang, 0, 360, (40, 215, 240), -1, cv2.LINE_AA)
        cv2.ellipse(img, (x, y), (r * 3, r), ang, 0, 360, (30, 160, 190), 3, cv2.LINE_AA)

    leaf = np.array([60, 150, 40], np.float32)
    yellowed = np.array([40, 200, 230], np.float32)  # hue ~50 deg: wilting, chlorophyll loss
    mix = min(1.0, max(0.0, (spoil - 0.25) / 0.6))
    for i, (x, y, r, ang) in enumerate(scene["sawi"]):
        m = mix if i % 3 else 0.5 * mix
        c = leaf * (1 - m) + yellowed * m
        cv2.ellipse(img, (x, y), (r, int(r * 0.55)), ang, 0, 360, tuple(int(v) for v in c), -1, cv2.LINE_AA)
        cv2.ellipse(img, (x, y), (r, 3), ang, 0, 360, (200, 235, 210), -1, cv2.LINE_AA)

    for x, y, r, ang in scene["carrot"]:
        cv2.ellipse(img, (x, y), (r * 4, r), ang, 0, 360, (20, 110, 240), -1, cv2.LINE_AA)
    return img


def _osd(img: np.ndarray, text: str, org: tuple[int, int]) -> None:
    for color, thick in (((0, 0, 0), 5), ((235, 235, 235), 2)):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thick, cv2.LINE_AA)


def camera_frame(scene: dict, spoil: float, i: int, fps: int, start: dt.datetime, occlusion: float) -> np.ndarray:
    top = render_topdown(scene, spoil)
    if occlusion > 0:  # staff arm / body reaching across the bins
        cx = int(W * (1.1 - 1.2 * occlusion))
        cv2.ellipse(top, (cx, int(H * 0.45)), (260, 700), 15, 0, 360, (70, 75, 95), -1, cv2.LINE_AA)
        cv2.ellipse(top, (cx - 120, int(H * 0.30)), (90, 420), 70, 0, 360, (120, 150, 190), -1, cv2.LINE_AA)
    img = cv2.warpPerspective(top, HOMOGRAPHY, (W, H), flags=cv2.INTER_LINEAR, borderValue=(52, 56, 60))

    t = i / fps
    gain = 1.0 + 0.04 * np.sin(2 * np.pi * t / 37.0) + 0.01 * np.sin(2 * np.pi * t * 0.9)  # daylight + tube flicker
    f = img.astype(np.float32) * scene["vignette"] * np.float32(gain) + scene["noise"][i % len(scene["noise"])]
    img = np.clip(f, 0, 255).astype(np.uint8)

    now = start + dt.timedelta(seconds=int(t))
    _osd(img, now.strftime("%Y-%m-%d %a %H:%M:%S"), (40, 60))
    _osd(img, CAMERA_NAME, (W - 330, H - 30))
    return img


def timeline(t: float, total: float) -> tuple[float, float]:
    """(spoil progress, occlusion progress) at time t for a clip of `total` seconds."""
    s = total / 120.0
    if t < 15 * s:
        return 0.0, 0.0
    if t < 95 * s:
        return (t - 15 * s) / (80 * s), 0.0
    if t < 106 * s:
        return 1.0, 0.0
    if t < 108 * s:
        return 1.0, (t - 106 * s) / (2 * s)
    return 0.0, 0.0


def _ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="samples/wetmarket_cctv_demo.mp4")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--crf", type=int, default=27, help="H.264 quality (lower = bigger file)")
    ap.add_argument("--print-zones", action="store_true")
    args = ap.parse_args()

    if args.print_zones:
        for name, poly in zone_polygons().items():
            print(f"{name}: {[list(p) for p in poly]}")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    start = dt.datetime(2026, 10, 30, 7, 45, 0)
    n = args.seconds * args.fps

    ffmpeg = _ffmpeg()
    if ffmpeg:
        proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
             "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
             "-g", str(args.fps * 2), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE,
        )
        write = proc.stdin.write  # type: ignore[union-attr]
    else:
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        if not writer.isOpened():
            raise SystemExit(f"cannot open video writer for {out}")
        write = writer.write  # type: ignore[assignment]

    for i in range(n):
        spoil, occ = timeline(i / args.fps, args.seconds)
        frame = camera_frame(scene, spoil, i, args.fps, start, occ)
        write(frame.tobytes() if ffmpeg else frame)

    if ffmpeg:
        proc.stdin.close()  # type: ignore[union-attr]
        if proc.wait() != 0:
            raise SystemExit("ffmpeg encoding failed")
    else:
        writer.release()
    size = out.stat().st_size / 1e6
    print(f"wrote {out} ({n} frames, {args.seconds}s @ {args.fps} fps, {'H.264' if ffmpeg else 'mp4v'}, {size:.1f} MB)")


if __name__ == "__main__":
    main()
