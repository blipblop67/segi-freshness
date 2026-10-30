# samples/

`wetmarket_cctv_demo.mp4` is a **synthetic** 2-minute clip (1080p, 15 fps, H.264) that imitates
a ceiling-mounted store CCTV camera over four wet-market bins. Use it to demo the system
before the real CCTV is connected.

| Time | What happens | What the dashboard should show |
|---|---|---|
| 0–15 s | Fresh stock | All bins grade A. Auto-calibration runs at 3 s (demo config only). |
| 15–95 s | Tomatoes develop brown/black rot; sawi leaves turn yellow | Tomato drops to D, with low-freshness and spoiled-item alerts. Sawi drops to C/D. |
| 95–106 s | Spoiled stock left on display | Alerts stay active and escalate to critical |
| 106–108 s | A staff member reaches across the bins | Brief occlusion. Debounce and smoothing prevent false alerts. |
| 108–120 s | Bins restocked | Scores recover and alerts clear, then the clip loops |

Regenerate or customise it with `scripts/make_demo_video.py` (`--seconds`, `--fps`, `--crf`).
The clip shows the pipeline working, but it is no substitute for testing on real store footage.
Record a few hours from the actual camera before tuning thresholds.
