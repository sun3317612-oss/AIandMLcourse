import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

# parse_layer_string 테스트
def test_parse_layer_string_valid():
    from week4_explorer import parse_layer_string
    assert parse_layer_string("[128, 64]") == [128, 64]
    assert parse_layer_string("[256, 128, 64]") == [256, 128, 64]
    assert parse_layer_string("[64]") == [64]

def test_parse_layer_string_spaces():
    from week4_explorer import parse_layer_string
    assert parse_layer_string("[ 128,  64 ]") == [128, 64]

def test_parse_layer_string_invalid_raises():
    from week4_explorer import parse_layer_string
    with pytest.raises(ValueError):
        parse_layer_string("[abc, 64]")
    with pytest.raises(ValueError):
        parse_layer_string("")
    with pytest.raises(ValueError):
        parse_layer_string("[]")
    with pytest.raises(ValueError):
        parse_layer_string("[0, 64]")  # 0개 유닛 불가

def test_training_thread_stop_flag():
    """stop() 호출 후 _stop이 True가 되는지 확인"""
    from week4_explorer import TrainingThread
    thread = TrainingThread.__new__(TrainingThread)
    thread._stop = False
    thread.stop()
    assert thread._stop is True
