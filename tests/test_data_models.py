import time

from vec_cache.data_models import StoredText


def test_stored_text_model():
    st1 = StoredText("test")
    assert isinstance(st1.timestamp, float)
    time.sleep(0.01)
    st2 = StoredText("test")
    assert st2.timestamp > st1.timestamp
