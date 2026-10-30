from segi_freshness.__main__ import TARGET_L4T, parse_l4t_release


def test_parse_jetpack_62_release_string():
    text = "# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025\n"
    assert parse_l4t_release(text) == "36.4.3"
    assert parse_l4t_release(text).startswith(TARGET_L4T)


def test_parse_other_release_and_garbage():
    assert parse_l4t_release("# R35 (release), REVISION: 5.0, GCID: 1") == "35.5.0"
    assert parse_l4t_release("not a tegra release file") is None
