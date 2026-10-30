import pytest

from segi_freshness.camera import build_pipeline, redact_uri, use_hw_decoder
from segi_freshness.config import CameraConfig, load_config

from .conftest import ROOT

URI = "rtsp://viewer:s3cr3t@192.168.1.64:554/Streaming/Channels/101"


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_rtsp_pipeline_uses_nvdec_for_codec(codec):
    p = build_pipeline(CameraConfig(source="rtsp", uri=URI, rtsp_codec=codec))
    assert f"rtp{codec}depay ! {codec}parse ! nvv4l2decoder" in p
    assert "protocols=tcp" in p and p.endswith("sync=false")


def test_redact_uri():
    assert redact_uri(URI) == "rtsp://viewer:***@192.168.1.64:554/Streaming/Channels/101"
    assert redact_uri("rtsp://10.0.0.5/live") == "rtsp://10.0.0.5/live"
    assert redact_uri("samples/x.mp4") == "samples/x.mp4"


def test_rtsp_uri_validation():
    with pytest.raises(ValueError):
        CameraConfig(source="rtsp", uri="")
    with pytest.raises(ValueError):
        CameraConfig(source="rtsp", uri="http://cam/stream")


def test_decoder_selection():
    assert use_hw_decoder(CameraConfig(source="rtsp", uri=URI, decoder="nvidia"))
    assert not use_hw_decoder(CameraConfig(source="rtsp", uri=URI, decoder="software"))


def test_demo_configs_extend_store_template():
    store = load_config(ROOT / "configs" / "store.yaml")
    demo = load_config(ROOT / "configs" / "demo.yaml")
    rtsp = load_config(ROOT / "configs" / "demo-rtsp.yaml")
    assert store.camera.source == "rtsp" and store.runtime.auto_calibrate_after_s is None
    assert demo.camera.source == "file" and demo.runtime.auto_calibrate_after_s == 3
    assert (ROOT / demo.camera.uri).is_file()
    # Inherited from store.yaml, not repeated in demo.yaml:
    assert demo.scoring.produce_profile_map == store.scoring.produce_profile_map
    assert [z.id for z in demo.zones] == [z.id for z in rtsp.zones]
    assert rtsp.camera.source == "rtsp" and rtsp.camera.loop_file is True
    assert all(len(z.polygon) == 4 for z in demo.zones)


def test_env_overrides_camera_uri(monkeypatch):
    monkeypatch.setenv("SEGI_CAMERA_URI", URI)
    assert load_config(ROOT / "configs" / "store.yaml").camera.uri == URI
