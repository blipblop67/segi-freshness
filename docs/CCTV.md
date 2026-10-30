# Connecting store CCTV

The monitor reads any IP camera or NVR channel over **RTSP**. On the Jetson the video is
decoded by the hardware decoder (NVDEC), so a 1080p stream costs almost no CPU.

```
IP camera ──RTSP──┐
                  ├── store LAN ──> Jetson Orin Nano ──> dashboard / MQTT / alerts
NVR channel ─RTSP─┘
```

## 1. Camera placement

| | Recommendation | Why |
|---|---|---|
| Position | Above the display, looking down at 45–70°, all target bins in view | Bins stay large in the frame. Fewer customers block the view. |
| Coverage | One camera covers about 4–8 bins | Each bin should be at least ~250 × 200 px in a 1080p frame |
| Resolution | 1080p main stream (or 4MP) | Rot spots are small. The sub-stream (D1/720p) is too coarse for freshness checks. |
| Lighting | Neutral white LED, 4000–5000 K, steady | Colour is the signal. Avoid red "meat" lamps and direct sunlight on bins. |
| Mounting | Rigid mount, no pan/tilt patrol | Zones are fixed image coordinates |

## 2. Camera settings (important)

Set these in the camera or NVR web UI. Out of the box, CCTV image processing is tuned for
security footage, not for judging colour.

- **White balance: Manual / Locked** (not Auto). Auto white balance shifts colours when a
  customer in a red shirt walks past.
- **Saturation / "Vivid" / colour enhancement: default or off**. Don't use a vivid colour boost.
- **WDR / HLC / BLC: off**, unless the bins are backlit.
- **Day/Night: force Colour (Day)**. Infrared night mode is black and white, which makes freshness checks impossible.
- **Encoding: H.264, CBR or VBR at 4–6 Mbps, 10–15 fps, I-frame interval ≤ 2 × fps.**
  H.265 also works (set `rtsp_codec: h265`).
- **Create a view-only user** for the monitor (e.g. `segi-viewer`). Never use the admin account.

## 3. Find the RTSP URL

| Brand | Main stream | Notes |
|---|---|---|
| Hikvision / HiLook | `rtsp://USER:PASS@IP:554/Streaming/Channels/101` | Channel 2 on an NVR = `201` |
| Dahua / Imou | `rtsp://USER:PASS@IP:554/cam/realmonitor?channel=1&subtype=0` | |
| Uniview | `rtsp://USER:PASS@IP:554/unicast/c1/s0/live` | |
| Axis | `rtsp://USER:PASS@IP/axis-media/media.amp` | |
| Others / ONVIF | Check the manual, or use ONVIF Device Manager | |

Test the URL from a laptop on the same network in VLC (*Media > Open Network Stream*) before
configuring the Jetson.

If the password contains `@ : / ?` or `#`, URL-encode it (`@` becomes `%40`).

## 4. Configure the monitor

```bash
sudo nano /etc/segi-freshness/env
# SEGI_CAMERA_URI=rtsp://segi-viewer:PASSWORD@192.168.1.64:554/Streaming/Channels/101
```

In `configs/store.yaml`, set `rtsp_codec` to match the camera, then draw the zones (next section).

```bash
segi-freshness check -c configs/store.yaml   # opens the stream and grabs one frame
```

Passwords are masked (`***`) in logs, `check` output and the API.

## 5. Draw the bin zones

1. Grab a frame: `curl -o frame.jpg http://JETSON:8080/api/snapshot.jpg`, or take a VLC snapshot.
2. Open it in any image editor and note the corner pixels of each bin.
3. Divide x by the image width and y by the image height, and enter the result as a `polygon`
   (clockwise, 4+ points) in `zones:`.
4. Set `produce:` per bin, so the right produce profile is used (e.g. `sawi` is treated as a leafy green).
5. Restart the service. Once the bins are stocked with fresh produce, press **Calibrate bins** on the dashboard.

`configs/demo.yaml` has a worked example of a camera viewing the bins at an angle.

## 6. Network

- Put the cameras, the NVR and the Jetson on the CCTV VLAN. The Jetson only needs outbound
  access to the MQTT broker or webhook, plus inbound TCP 8080 from the supervisor's PC/tablet.
- Use RTSP over TCP (`rtsp_transport: tcp`, the default). It is more robust on busy store networks.
- If a stream drops, the monitor reconnects with backoff (1 s up to 30 s). After 30 s without
  frames it raises a **camera_offline** alert.

## Demo without a camera

```bash
# Option A: replay the sample clip directly
segi-freshness run -c configs/demo.yaml

# Option B: exercise the real RTSP path with a simulated CCTV camera
docker compose -f docker/cctv-simulator.yml up -d
segi-freshness run -c configs/demo-rtsp.yaml
```
