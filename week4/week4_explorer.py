"""
Week4 Neural Explorer — PySide6 인터랙티브 트레이닝 GUI
실행: uv run week4/week4_explorer.py
"""

import os
import sys
import re
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ─────────────────────────────────────────────
# 모듈 수준 PySide6 / Keras 임포트
# (TrainingThread 클래스 정의에 필요; GUI 없는 환경에서도 가져올 수 있도록 시도)
# ─────────────────────────────────────────────
try:
    from PySide6.QtCore import QThread, Signal
    import tensorflow as tf
    from tensorflow import keras
    _GUI_IMPORTS_OK = True
except ImportError:
    # 테스트 환경 등 GUI/TF가 없는 경우: 더미 베이스 클래스로 대체
    class QThread:  # type: ignore
        pass
    def Signal(*args):  # type: ignore
        return None
    keras = None  # type: ignore
    tf = None  # type: ignore
    _GUI_IMPORTS_OK = False


# ─────────────────────────────────────────────
# 한글 폰트 설정
# ─────────────────────────────────────────────

def _setup_korean_font():
    try:
        import matplotlib
        matplotlib.use('QtAgg')
        import matplotlib.font_manager as fm
        font_list = [f.name for f in fm.fontManager.ttflist]
        for font in ['Malgun Gothic', 'Gulim', 'Batang', 'Dotum', 'NanumGothic']:
            if font in font_list:
                import matplotlib.pyplot as plt
                plt.rcParams['font.family'] = font
                plt.rcParams['axes.unicode_minus'] = False
                break
    except Exception:
        # Fail silently if GUI imports not available (e.g., in testing)
        pass


def _setup_gui_imports():
    """Import GUI modules (only when GUI is actually needed)"""
    global FigureCanvas, Figure, QApplication, QMainWindow, QTabWidget
    global QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
    global QComboBox, QSlider, QLineEdit, QStatusBar
    global Qt, QThread, Signal, QFont
    global tf, keras

    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QTabWidget, QWidget,
        QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
        QComboBox, QSlider, QLineEdit, QStatusBar,
    )
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QFont

    import tensorflow as tf
    from tensorflow import keras


# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

def parse_layer_string(s: str) -> list:
    """
    "[128, 64]" → [128, 64]
    유효하지 않으면 ValueError.
    """
    s = s.strip()
    if not s:
        raise ValueError("레이어 입력이 비어 있습니다.")
    # 대괄호 제거 후 쉼표로 분리
    inner = re.sub(r'[\[\]]', '', s).strip()
    if not inner:
        raise ValueError("레이어가 하나도 없습니다.")
    parts = [p.strip() for p in inner.split(',')]
    result = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"숫자가 아닌 값: '{p}'")
        n = int(p)
        if n <= 0:
            raise ValueError(f"레이어 크기는 1 이상이어야 합니다: {n}")
        result.append(n)
    if not result:
        raise ValueError("레이어 목록이 비어 있습니다.")
    return result


# ─────────────────────────────────────────────
# TrainingThread 베이스
# ─────────────────────────────────────────────

class TrainingThread(QThread):
    """
    Keras 학습을 별도 스레드에서 실행하는 베이스 클래스.
    서브클래스는 _run_training()을 구현해야 함.
    """
    epoch_done  = Signal(int, dict)   # (epoch, logs)
    train_done  = Signal(object)      # trained keras model
    train_error = Signal(str)         # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False

    def stop(self):
        self._stop = True

    def _make_epoch_callback(self):
        """매 epoch 끝에 epoch_done 시그널을 발생시키는 Keras 콜백 반환."""
        thread = self

        def on_epoch_end(epoch, logs=None):
            if thread._stop:
                thread.model.stop_training = True
                return
            thread.epoch_done.emit(epoch, dict(logs or {}))

        return keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

    def run(self):
        try:
            self._run_training()
        except Exception as e:
            self.train_error.emit(str(e))

    def _run_training(self):
        raise NotImplementedError


# ─────────────────────────────────────────────
# 공통 UI 헬퍼
# ─────────────────────────────────────────────

def make_canvas(width_inch=5, height_inch=3.5):
    fig = Figure(figsize=(width_inch, height_inch), tight_layout=True)
    canvas = FigureCanvas(fig)
    canvas.setMinimumHeight(200)
    return canvas


def make_slider(min_val, max_val, default, step=1):
    sl = QSlider(Qt.Horizontal)
    sl.setRange(min_val, max_val)
    sl.setSingleStep(step)
    sl.setValue(default)
    return sl


# ─────────────────────────────────────────────
# Lab 1 — 1D 함수 근사
# ─────────────────────────────────────────────

_LAB1_FUNCTIONS = {
    'sin(x)': lambda x: np.sin(x),
    'cos(x)+0.5sin(2x)': lambda x: np.cos(x) + 0.5 * np.sin(2 * x),
    'x·sin(x)': lambda x: x * np.sin(x),
}


def make_lab1_data(func_name):
    """
    Returns (x_train, y_train, x_test, y_test) for Lab1.
    x range: [-2π, 2π]
    """
    if func_name not in _LAB1_FUNCTIONS:
        raise ValueError(f"알 수 없는 함수: {func_name}")
    fn = _LAB1_FUNCTIONS[func_name]
    x_train = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
    x_test  = np.linspace(-2 * np.pi, 2 * np.pi, 400).reshape(-1, 1)
    return x_train, fn(x_train).reshape(-1, 1), x_test, fn(x_test).reshape(-1, 1)


def build_lab1_model(layers, activation, lr):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(1,)))
    for units in layers:
        model.add(keras.layers.Dense(units, activation=activation))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model


class Lab1TrainingThread(TrainingThread):
    def __init__(self, func_name, layers, epochs, lr, activation, parent=None):
        super().__init__(parent)
        self.func_name  = func_name
        self.layers     = layers
        self.epochs     = epochs
        self.lr         = lr
        self.activation = activation

    def _run_training(self):
        x_tr, y_tr, x_te, y_te = make_lab1_data(self.func_name)
        self.model = build_lab1_model(self.layers, self.activation, self.lr)
        self.model.fit(
            x_tr, y_tr,
            validation_data=(x_te, y_te),
            epochs=self.epochs,
            batch_size=32,
            verbose=0,
            callbacks=[self._make_epoch_callback()],
        )
        self.train_done.emit(self.model)
