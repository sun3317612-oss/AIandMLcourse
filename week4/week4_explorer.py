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
    class QWidget:  # type: ignore
        pass
    class QStatusBar:  # type: ignore
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


class Lab1Widget(QWidget):
    """탭 0 — 1D 함수 근사"""

    def __init__(self, status_bar: QStatusBar, parent=None):
        super().__init__(parent)
        self._status  = status_bar
        self._thread  = None
        self._history = {'loss': [], 'val_loss': []}
        self._setup_ui()

    # ── UI 구성 ──────────────────────────────

    def _setup_ui(self):
        root = QHBoxLayout(self)

        # 왼쪽 파라미터 패널 (고정 너비 260px)
        param_panel = QWidget()
        param_panel.setFixedWidth(260)
        pv = QVBoxLayout(param_panel)
        pv.setAlignment(Qt.AlignTop)

        # 함수 선택
        pv.addWidget(QLabel("함수 선택"))
        self._func_combo = QComboBox()
        self._func_combo.addItems(list(_LAB1_FUNCTIONS.keys()))
        pv.addWidget(self._func_combo)

        # 히든 레이어
        pv.addWidget(QLabel("Hidden Layers"))
        self._layer_edit = QLineEdit("[128, 128, 64]")
        pv.addWidget(self._layer_edit)

        # Epochs 슬라이더
        pv.addWidget(QLabel("Epochs: 3000"))
        self._epoch_slider = make_slider(100, 5000, 3000, 100)
        self._epoch_label  = QLabel("3000")
        self._epoch_slider.valueChanged.connect(
            lambda v: self._epoch_label.setText(str(v)))
        pv.addWidget(self._epoch_slider)
        pv.addWidget(self._epoch_label)

        # Learning Rate
        pv.addWidget(QLabel("Learning Rate"))
        self._lr_combo = QComboBox()
        self._lr_combo.addItems(["0.01", "0.001", "0.0001"])
        self._lr_combo.setCurrentIndex(0)
        pv.addWidget(self._lr_combo)

        # Activation
        pv.addWidget(QLabel("Activation"))
        self._act_combo = QComboBox()
        self._act_combo.addItems(["tanh", "relu"])
        pv.addWidget(self._act_combo)

        # 실행/중지 버튼
        self._run_btn = QPushButton("실행")
        self._run_btn.clicked.connect(self._on_run_stop)
        pv.addWidget(self._run_btn)

        root.addWidget(param_panel)

        # 오른쪽 차트 패널
        chart_panel = QWidget()
        cv = QVBoxLayout(chart_panel)

        self._loss_canvas = make_canvas()
        self._pred_canvas = make_canvas()
        cv.addWidget(self._loss_canvas)
        cv.addWidget(self._pred_canvas)

        root.addWidget(chart_panel, stretch=1)

    # ── 슬롯 ────────────────────────────────

    def _on_run_stop(self):
        if self._thread and self._thread.isRunning():
            self._thread.stop()
            self._run_btn.setText("실행")
            self._set_params_enabled(True)
            return

        # 입력 검증
        try:
            layers = parse_layer_string(self._layer_edit.text())
        except ValueError as e:
            self._layer_edit.setStyleSheet("border: 2px solid red;")
            self._status.showMessage(f"레이어 입력 오류: {e}")
            return
        self._layer_edit.setStyleSheet("")

        # 학습 시작
        self._history = {'loss': [], 'val_loss': []}
        self._clear_charts()

        func_name  = self._func_combo.currentText()
        self._current_func_name = func_name
        epochs     = self._epoch_slider.value()
        lr         = float(self._lr_combo.currentText())
        activation = self._act_combo.currentText()

        self._thread = Lab1TrainingThread(func_name, layers, epochs, lr, activation)
        self._thread.epoch_done.connect(self._on_epoch)
        self._thread.train_done.connect(self._on_done)
        self._thread.train_error.connect(
            lambda msg: self._status.showMessage(f"오류: {msg}"))
        self._thread.finished.connect(
            lambda: (self._run_btn.setText("실행"),
                     self._set_params_enabled(True)))
        self._thread.start()

        self._run_btn.setText("중지")
        self._set_params_enabled(False)
        self._status.showMessage("Lab1 학습 중…")

    def _set_params_enabled(self, enabled: bool):
        for w in [self._func_combo, self._layer_edit, self._epoch_slider,
                  self._lr_combo, self._act_combo]:
            w.setEnabled(enabled)

    def _clear_charts(self):
        for canvas in [self._loss_canvas, self._pred_canvas]:
            canvas.figure.clear()
            canvas.draw()

    def _on_epoch(self, epoch: int, logs: dict):
        self._history['loss'].append(logs.get('loss', 0))
        self._history['val_loss'].append(logs.get('val_loss', 0))
        self._status.showMessage(
            f"Lab1 | epoch {epoch+1} | loss={logs.get('loss', 0):.5f}")

        fig = self._loss_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(self._history['loss'],     label='Train Loss')
        ax.plot(self._history['val_loss'], label='Val Loss', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.set_title('Loss 곡선')
        ax.legend(fontsize=8)
        ax.set_yscale('log')
        self._loss_canvas.draw()

    def _on_done(self, model):
        try:
            func_name = self._current_func_name
            _, _, x_te, y_te = make_lab1_data(func_name)
            y_pred = model.predict(x_te, verbose=0)

            fig = self._pred_canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.plot(x_te, y_te,   color='green',  label='실제값')
            ax.plot(x_te, y_pred, color='orange', linestyle='--', label='예측값')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{func_name} 근사 결과')
            ax.legend(fontsize=8)
            self._pred_canvas.draw()

            self._status.showMessage("Lab1 학습 완료")
        except Exception as e:
            self._status.showMessage(f"결과 표시 오류: {e}")

    def stop_training(self):
        """MainWindow closeEvent에서 호출"""
        if self._thread and self._thread.isRunning():
            self._thread.stop()
            self._thread.wait()


# ─────────────────────────────────────────────
# Lab 2 — 포물선 운동
# ─────────────────────────────────────────────

_G = 9.81


def make_lab2_data(n_train: int = 2000, n_test: int = 500):
    """
    Returns (X_train, Y_train, X_test, Y_test)
    X: (v0, theta_deg, t)   Y: (x, y)
    """
    def _gen(n, noise=0.5):
        v0    = np.random.uniform(10, 50, n)
        theta = np.random.uniform(20, 70, n)
        tr    = np.deg2rad(theta)
        t_max = 2 * v0 * np.sin(tr) / _G
        t     = np.random.uniform(0, t_max * 0.9, n)
        x = v0 * np.cos(tr) * t + np.random.normal(0, noise, n)
        y = v0 * np.sin(tr) * t - 0.5 * _G * t**2 + np.random.normal(0, noise, n)
        mask = y >= 0
        return (np.column_stack([v0[mask], theta[mask], t[mask]]),
                np.column_stack([x[mask],  y[mask]]))

    X_tr, Y_tr = _gen(n_train, noise=0.5)
    X_te, Y_te = _gen(n_test,  noise=0.0)
    return X_tr, Y_tr, X_te, Y_te


def make_lab2_trajectory_physics(v0: float, theta_deg: float,
                                  n_points: int = 50):
    """물리 공식으로 전체 궤적 (x, y) 반환."""
    tr    = np.deg2rad(theta_deg)
    t_max = 2 * v0 * np.sin(tr) / _G
    t     = np.linspace(0, t_max, n_points)
    x     = v0 * np.cos(tr) * t
    y     = v0 * np.sin(tr) * t - 0.5 * _G * t**2
    return x, y


def build_lab2_model(layers: list, lr: float):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(3,)))
    for units in layers:
        model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(2, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model


class Lab2TrainingThread(TrainingThread):
    def __init__(self, layers: list, epochs: int, lr: float, parent=None):
        super().__init__(parent)
        self.layers = layers
        self.epochs = epochs
        self.lr     = lr

    def _run_training(self):
        X_tr, Y_tr, X_te, Y_te = make_lab2_data()
        self.model = build_lab2_model(self.layers, self.lr)
        self.model.fit(
            X_tr, Y_tr,
            validation_data=(X_te, Y_te),
            epochs=self.epochs,
            batch_size=32,
            verbose=0,
            callbacks=[self._make_epoch_callback()],
        )
        self.train_done.emit(self.model)


class Lab2Widget(QWidget):
    """탭 1 — 포물선 운동"""

    def __init__(self, status_bar, parent=None):
        super().__init__(parent)
        self._status  = status_bar
        self._thread  = None
        self._history = {'loss': [], 'val_loss': []}
        self._setup_ui()

    def _setup_ui(self):
        root = QHBoxLayout(self)

        param_panel = QWidget()
        param_panel.setFixedWidth(260)
        pv = QVBoxLayout(param_panel)
        pv.setAlignment(Qt.AlignTop)

        pv.addWidget(QLabel("Hidden Layers"))
        self._layer_edit = QLineEdit("[128, 64, 32]")
        pv.addWidget(self._layer_edit)

        pv.addWidget(QLabel("Epochs"))
        self._epoch_slider = make_slider(100, 2000, 500, 100)
        self._epoch_label  = QLabel("500")
        self._epoch_slider.valueChanged.connect(
            lambda v: self._epoch_label.setText(str(v)))
        pv.addWidget(self._epoch_slider)
        pv.addWidget(self._epoch_label)

        pv.addWidget(QLabel("Learning Rate"))
        self._lr_combo = QComboBox()
        self._lr_combo.addItems(["0.01", "0.001", "0.0001"])
        self._lr_combo.setCurrentIndex(1)
        pv.addWidget(self._lr_combo)

        pv.addWidget(QLabel("테스트 초기속력 v₀ (m/s)"))
        self._v0_slider = make_slider(10, 50, 30)
        self._v0_label  = QLabel("30")
        self._v0_slider.valueChanged.connect(
            lambda v: self._v0_label.setText(str(v)))
        pv.addWidget(self._v0_slider)
        pv.addWidget(self._v0_label)

        pv.addWidget(QLabel("테스트 발사각 θ (°)"))
        self._theta_slider = make_slider(10, 80, 45)
        self._theta_label  = QLabel("45")
        self._theta_slider.valueChanged.connect(
            lambda v: self._theta_label.setText(str(v)))
        pv.addWidget(self._theta_slider)
        pv.addWidget(self._theta_label)

        self._run_btn = QPushButton("실행")
        self._run_btn.clicked.connect(self._on_run_stop)
        pv.addWidget(self._run_btn)

        root.addWidget(param_panel)

        chart_panel = QWidget()
        cv = QVBoxLayout(chart_panel)
        self._loss_canvas = make_canvas()
        self._pred_canvas = make_canvas()
        cv.addWidget(self._loss_canvas)
        cv.addWidget(self._pred_canvas)
        root.addWidget(chart_panel, stretch=1)

    def _on_run_stop(self):
        if self._thread and self._thread.isRunning():
            self._thread.stop()
            self._run_btn.setText("실행")
            self._set_params_enabled(True)
            return

        try:
            layers = parse_layer_string(self._layer_edit.text())
        except ValueError as e:
            self._layer_edit.setStyleSheet("border: 2px solid red;")
            self._status.showMessage(f"레이어 입력 오류: {e}")
            return
        self._layer_edit.setStyleSheet("")

        self._history = {'loss': [], 'val_loss': []}
        for canvas in [self._loss_canvas, self._pred_canvas]:
            canvas.figure.clear()
            canvas.draw()

        epochs = self._epoch_slider.value()
        lr     = float(self._lr_combo.currentText())
        self._test_v0    = self._v0_slider.value()
        self._test_theta = self._theta_slider.value()

        self._thread = Lab2TrainingThread(layers, epochs, lr)
        self._thread.epoch_done.connect(self._on_epoch)
        self._thread.train_done.connect(self._on_done)
        self._thread.train_error.connect(
            lambda msg: self._status.showMessage(f"오류: {msg}"))
        self._thread.finished.connect(
            lambda: (self._run_btn.setText("실행"),
                     self._set_params_enabled(True)))
        self._thread.start()

        self._run_btn.setText("중지")
        self._set_params_enabled(False)
        self._status.showMessage("Lab2 학습 중…")

    def _set_params_enabled(self, enabled: bool):
        for w in [self._layer_edit, self._epoch_slider, self._lr_combo,
                  self._v0_slider, self._theta_slider]:
            w.setEnabled(enabled)

    def _on_epoch(self, epoch: int, logs: dict):
        self._history['loss'].append(logs.get('loss', 0))
        self._history['val_loss'].append(logs.get('val_loss', 0))
        self._status.showMessage(
            f"Lab2 | epoch {epoch+1} | loss={logs.get('loss', 0):.5f}")

        fig = self._loss_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(self._history['loss'],     label='Train Loss')
        ax.plot(self._history['val_loss'], label='Val Loss', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.set_title('Loss 곡선')
        ax.legend(fontsize=8)
        ax.set_yscale('log')
        self._loss_canvas.draw()

    def _on_done(self, model):
        try:
            v0        = self._test_v0
            theta_deg = self._test_theta

            tr    = np.deg2rad(theta_deg)
            t_max = 2 * v0 * np.sin(tr) / _G
            t     = np.linspace(0, t_max, 50)
            X_in  = np.column_stack([
                np.full(50, v0), np.full(50, theta_deg), t
            ])
            pred      = model.predict(X_in, verbose=0)
            x_pred, y_pred = pred[:, 0], pred[:, 1]
            x_true, y_true = make_lab2_trajectory_physics(v0, theta_deg)

            fig = self._pred_canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.plot(x_true, y_true, color='blue',   label='물리 공식')
            ax.plot(x_pred, y_pred, color='orange', linestyle='--', label='예측값')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title(f'포물선 궤적 비교 (v₀={v0} m/s, θ={theta_deg}°)')
            ax.legend(fontsize=8)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            self._pred_canvas.draw()

            self._status.showMessage("Lab2 학습 완료")
        except Exception as e:
            self._status.showMessage(f"결과 표시 오류: {e}")

    def stop_training(self):
        if self._thread and self._thread.isRunning():
            self._thread.stop()
            self._thread.wait()
