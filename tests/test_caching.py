from utils.caching import set_cache, get_cache


def test_cache_set_get(tmp_path, monkeypatch):
    # override cache file location to tmp
    monkeypatch.setattr('utils.caching.CACHE_FILE', tmp_path / "cache.json")
    set_cache("foo", {"a": 1})
    assert get_cache("foo") == {"a": 1}
