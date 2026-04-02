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

def test_lab2_data_shapes():
    from week4_explorer import make_lab2_data
    X_tr, Y_tr, X_te, Y_te = make_lab2_data(n_train=200, n_test=50)
    assert X_tr.shape[1] == 3   # (v0, theta, t)
    assert Y_tr.shape[1] == 2   # (x, y)
    assert X_te.shape[1] == 3
    # y >= 0 보장 검증
    assert (Y_tr[:, 1] >= 0).all()

def test_lab2_trajectory_shape():
    from week4_explorer import make_lab2_trajectory_physics
    x, y = make_lab2_trajectory_physics(v0=30, theta_deg=45, n_points=50)
    assert len(x) == 50
    assert len(y) == 50
    assert y[0] >= 0
    assert x[0] >= 0

def test_lab3_data_shapes():
    from week4_explorer import make_lab3_data
    x_tr, y_tr, x_val, y_val, x_te, y_te = make_lab3_data(noise_level=0.3)
    assert x_tr.shape == (100, 1)
    assert y_tr.shape == (100, 1)
    assert x_val.shape == (50, 1)
    assert x_te.shape == (200, 1)

def test_lab3_true_function():
    from week4_explorer import lab3_true_function
    x = np.array([[0.0], [1.0]])
    y = lab3_true_function(x)
    # y = sin(2x) + 0.5x
    expected = np.sin(2 * x) + 0.5 * x
    np.testing.assert_allclose(y, expected, atol=1e-10)

def test_lab4_period_small_angle():
    """작은 각도에서 T ≈ 2π√(L/g)"""
    from week4_explorer import calculate_pendulum_period
    g = 9.81
    L = 1.0
    T_approx = 2 * np.pi * np.sqrt(L / g)
    T_calc   = calculate_pendulum_period(L, theta0_deg=5.0)
    assert abs(T_calc - T_approx) / T_approx < 0.01  # 1% 이내

def test_lab4_period_increases_with_angle():
    """큰 각도일수록 주기가 길어짐"""
    from week4_explorer import calculate_pendulum_period
    T_small = calculate_pendulum_period(1.0, 10.0)
    T_large = calculate_pendulum_period(1.0, 70.0)
    assert T_large > T_small

def test_lab4_data_shapes():
    from week4_explorer import make_lab4_data
    X_tr, Y_tr = make_lab4_data(n_samples=100)
    assert X_tr.shape == (100, 2)   # (L, theta0)
    assert Y_tr.shape == (100, 1)   # T
