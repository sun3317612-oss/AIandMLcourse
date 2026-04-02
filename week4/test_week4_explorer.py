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


def test_lab1_data_shapes():
    from week4_explorer import make_lab1_data
    x_tr, y_tr, x_te, y_te = make_lab1_data('sin(x)')
    assert x_tr.shape == (200, 1)
    assert y_tr.shape == (200, 1)
    assert x_te.shape == (400, 1)
    assert y_te.shape == (400, 1)

def test_lab1_data_functions():
    from week4_explorer import make_lab1_data
    _, y_tr, _, _ = make_lab1_data('sin(x)')
    x = np.linspace(-2*np.pi, 2*np.pi, 200).reshape(-1, 1)
    np.testing.assert_allclose(y_tr, np.sin(x), atol=1e-10)

def test_lab1_data_unknown_function():
    from week4_explorer import make_lab1_data
    with pytest.raises(ValueError):
        make_lab1_data('unknown_func')
