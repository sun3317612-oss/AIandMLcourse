"""
Week3 신경망 기초 탐색기 (Neural Networks Interactive Explorer)
PySide6 GUI 구현

실행: uv run week3/week3_neural_explorer.py
"""

import sys
import numpy as np

# Matplotlib 백엔드는 pyplot import 전에 설정
import matplotlib
matplotlib.use('QtAgg')

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QGroupBox, QCheckBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


# ─────────────────────────────────────────────
# 한글 폰트 설정
# ─────────────────────────────────────────────

def set_korean_font():
    font_list = [f.name for f in fm.fontManager.ttflist]
    for font in ['Malgun Gothic', 'Gulim', 'Batang', 'Dotum', 'NanumGothic', 'AppleGothic']:
        if font in font_list:
            plt.rcParams['font.family'] = font
            break
    plt.rcParams['axes.unicode_minus'] = False


set_korean_font()


# ─────────────────────────────────────────────
# 활성화 함수
# ─────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


# ─────────────────────────────────────────────
# 신경망 구현
# ─────────────────────────────────────────────

class Perceptron:
    """단일 퍼셉트론 (계단 함수 활성화)"""

    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size) * 0.1
        self.bias = np.random.randn() * 0.1
        self.lr = learning_rate

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        return self.activation(np.dot(inputs, self.weights) + self.bias)

    def train(self, X, y, epochs):
        for _ in range(epochs):
            for inputs, label in zip(X, y):
                error = label - self.predict(inputs)
                self.weights += self.lr * error * inputs
                self.bias += self.lr * error


class MLP:
    """2층 다층 퍼셉트론 (Sigmoid 활성화 + MSE 손실)"""

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate
        self.loss_history = []
        self.a1 = None

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]
        dz2 = output - y
        dW2 = (1 / m) * self.a1.T @ dz2
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * sigmoid_derivative(self.z1)
        dW1 = (1 / m) * X.T @ dz1
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs):
        self.loss_history = []
        for _ in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)
            self.loss_history.append(loss)
            self.backward(X, y, output)

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


# ─────────────────────────────────────────────
# 탭 1: 퍼셉트론
# ─────────────────────────────────────────────

class PerceptronTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)

        # 제어 패널
        panel = QGroupBox("설정")
        panel.setMaximumWidth(260)
        pl = QVBoxLayout(panel)

        pl.addWidget(QLabel("게이트 선택:"))
        self.gate_combo = QComboBox()
        self.gate_combo.addItems(["AND", "OR", "XOR", "모두 보기"])
        pl.addWidget(self.gate_combo)

        pl.addWidget(QLabel("학습률:"))
        self.lr_box = QDoubleSpinBox()
        self.lr_box.setRange(0.001, 1.0)
        self.lr_box.setValue(0.1)
        self.lr_box.setSingleStep(0.05)
        pl.addWidget(self.lr_box)

        pl.addWidget(QLabel("에포크:"))
        self.epoch_box = QSpinBox()
        self.epoch_box.setRange(10, 10000)
        self.epoch_box.setValue(100)
        self.epoch_box.setSingleStep(100)
        pl.addWidget(self.epoch_box)

        btn = QPushButton("학습 시작")
        btn.clicked.connect(self.run)
        pl.addWidget(btn)

        pl.addWidget(QLabel("결과:"))
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        pl.addWidget(self.result)
        pl.addStretch()
        layout.addWidget(panel)

        # Matplotlib 캔버스
        self.fig = Figure(figsize=(11, 4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.run()

    def run(self):
        gate = self.gate_combo.currentText()
        lr = self.lr_box.value()
        epochs = self.epoch_box.value()

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        Y = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "XOR": [0, 1, 1, 0]}

        self.fig.clear()
        text = ""

        if gate == "모두 보기":
            axes = self.fig.subplots(1, 3)
            for ax, (name, y) in zip(axes, Y.items()):
                y = np.array(y)
                p = Perceptron(2, lr)
                p.train(X, y, epochs)
                self._draw_boundary(ax, p, X, y, name)
                errs = sum(p.predict(x) != lbl for x, lbl in zip(X, y))
                text += f"{name}: 오류 {errs}/4\n"
        else:
            ax = self.fig.add_subplot(111)
            y = np.array(Y[gate])
            p = Perceptron(2, lr)
            p.train(X, y, epochs)
            self._draw_boundary(ax, p, X, y, gate)
            text = "입력  → 예측 (정답)\n"
            errs = 0
            for x, lbl in zip(X, y):
                pred = p.predict(x)
                ok = "✓" if pred == lbl else "✗"
                text += f"[{int(x[0])},{int(x[1])}] → {pred}  ({lbl})  {ok}\n"
                if pred != lbl:
                    errs += 1
            text += f"\n오류: {errs}/4\n"
            if errs == 0:
                text += "→ 학습 성공!"
            elif gate == "XOR":
                text += "→ XOR은 단일 퍼셉트론으로 불가능!"

        self.fig.tight_layout()
        self.canvas.draw()
        self.result.setText(text)

    def _draw_boundary(self, ax, p, X, y, title):
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 120), np.linspace(-0.5, 1.5, 120))
        Z = np.array([p.predict(np.array([xi, yi])) for xi, yi in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=['royalblue', 'tomato'])
        for pt, lbl in zip(X, y):
            c = 'tomato' if lbl == 1 else 'royalblue'
            m = 'o' if lbl == 1 else 's'
            ax.scatter(pt[0], pt[1], c=c, marker=m, s=220, edgecolors='black', linewidth=1.5, zorder=5)
        ax.set_title(f"{title} Gate")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)


# ─────────────────────────────────────────────
# 탭 2: 활성화 함수
# ─────────────────────────────────────────────

class ActivationFunctionsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)

        panel = QGroupBox("표시 설정")
        panel.setMaximumWidth(260)
        pl = QVBoxLayout(panel)

        pl.addWidget(QLabel("표시할 함수:"))
        self.cb_sigmoid = QCheckBox("Sigmoid")
        self.cb_sigmoid.setChecked(True)
        self.cb_tanh = QCheckBox("Tanh")
        self.cb_tanh.setChecked(True)
        self.cb_relu = QCheckBox("ReLU")
        self.cb_relu.setChecked(True)
        self.cb_lrelu = QCheckBox("Leaky ReLU")
        self.cb_lrelu.setChecked(True)

        for cb in [self.cb_sigmoid, self.cb_tanh, self.cb_relu, self.cb_lrelu]:
            pl.addWidget(cb)
            cb.toggled.connect(self.update_plot)

        pl.addWidget(QLabel("\n표시 모드:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["함수 + 미분", "함수만", "미분만"])
        self.mode_combo.currentTextChanged.connect(self.update_plot)
        pl.addWidget(self.mode_combo)

        pl.addWidget(QLabel("\n특성 요약:"))
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(180)
        pl.addWidget(self.info)
        pl.addStretch()
        layout.addWidget(panel)

        self.fig = Figure(figsize=(11, 5))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.update_plot()

    def update_plot(self):
        x = np.linspace(-5, 5, 300)

        funcs = []
        if self.cb_sigmoid.isChecked():
            funcs.append(("Sigmoid", sigmoid(x), sigmoid(x) * (1 - sigmoid(x)), 'royalblue'))
        if self.cb_tanh.isChecked():
            funcs.append(("Tanh", np.tanh(x), 1 - np.tanh(x) ** 2, 'green'))
        if self.cb_relu.isChecked():
            funcs.append(("ReLU", relu(x), np.where(x > 0, 1.0, 0.0), 'tomato'))
        if self.cb_lrelu.isChecked():
            funcs.append(("Leaky ReLU", leaky_relu(x), np.where(x > 0, 1.0, 0.01), 'orange'))

        mode = self.mode_combo.currentText()
        self.fig.clear()

        def style_ax(ax, title):
            ax.set_title(title)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='k', lw=0.6)
            ax.axvline(0, color='k', lw=0.6)

        if mode == "함수 + 미분":
            ax1, ax2 = self.fig.subplots(1, 2)
            for name, f, df, c in funcs:
                ax1.plot(x, f, label=name, color=c, lw=2)
                ax2.plot(x, df, label=f"{name}'", color=c, lw=2)
            style_ax(ax1, "활성화 함수")
            style_ax(ax2, "미분 (Gradient)")
        elif mode == "함수만":
            ax = self.fig.add_subplot(111)
            for name, f, _, c in funcs:
                ax.plot(x, f, label=name, color=c, lw=2)
            style_ax(ax, "활성화 함수 비교")
        else:
            ax = self.fig.add_subplot(111)
            for name, _, df, c in funcs:
                ax.plot(x, df, label=f"{name}'", color=c, lw=2)
            style_ax(ax, "Gradient 비교")

        self.fig.tight_layout()
        self.canvas.draw()

        info = (
            "Sigmoid: 출력 (0,1)  이진 분류 출력층\n"
            "Tanh   : 출력 (-1,1) 0 중심  RNN에 사용\n"
            "ReLU   : x>0이면 x   현대 신경망 표준\n"
            "Leaky  : Dying ReLU 문제 완화\n\n"
            "Vanishing Gradient:\n"
            "  Sigmoid/Tanh는 x가 크면 기울기≈0\n"
            "  → 깊은 망에서 학습 불안정\n"
            "  ReLU는 x>0이면 기울기=1 → 해결!"
        )
        self.info.setText(info)


# ─────────────────────────────────────────────
# 탭 3: 순전파
# ─────────────────────────────────────────────

class ForwardPropTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)

        panel = QGroupBox("입력 설정")
        panel.setMaximumWidth(260)
        pl = QVBoxLayout(panel)

        pl.addWidget(QLabel("입력 x1:"))
        self.x1_box = QDoubleSpinBox()
        self.x1_box.setRange(-2.0, 2.0)
        self.x1_box.setValue(0.5)
        self.x1_box.setSingleStep(0.1)
        pl.addWidget(self.x1_box)

        pl.addWidget(QLabel("입력 x2:"))
        self.x2_box = QDoubleSpinBox()
        self.x2_box.setRange(-2.0, 2.0)
        self.x2_box.setValue(0.8)
        self.x2_box.setSingleStep(0.1)
        pl.addWidget(self.x2_box)

        pl.addWidget(QLabel("은닉층 크기:"))
        self.hidden_box = QSpinBox()
        self.hidden_box.setRange(2, 8)
        self.hidden_box.setValue(3)
        pl.addWidget(self.hidden_box)

        pl.addWidget(QLabel("활성화 함수:"))
        self.act_combo = QComboBox()
        self.act_combo.addItems(["ReLU", "Sigmoid", "Tanh"])
        pl.addWidget(self.act_combo)

        btn = QPushButton("순전파 실행")
        btn.clicked.connect(self.run)
        pl.addWidget(btn)

        pl.addWidget(QLabel("\n단계별 계산:"))
        self.step_text = QTextEdit()
        self.step_text.setReadOnly(True)
        pl.addWidget(self.step_text)

        layout.addWidget(panel)

        self.fig = Figure(figsize=(11, 5))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.run()

    def run(self):
        x1, x2 = self.x1_box.value(), self.x2_box.value()
        h = self.hidden_box.value()
        act_name = self.act_combo.currentText()

        np.random.seed(42)
        W1 = np.random.randn(2, h) * 0.5
        b1 = np.random.randn(h) * 0.5
        W2 = np.random.randn(h, 1) * 0.5
        b2 = np.random.randn(1) * 0.5

        X = np.array([[x1, x2]])
        act_fn = {"ReLU": relu, "Sigmoid": sigmoid, "Tanh": np.tanh}[act_name]

        z1 = (X @ W1 + b1)[0]
        a1 = act_fn(z1)
        z2 = (a1 @ W2 + b2)[0]
        a2 = sigmoid(z2)[0]

        # 단계별 텍스트
        text = f"입력: x = [{x1:.2f}, {x2:.2f}]\n\n"
        text += f"── Layer 1 (입력 → 은닉층) ──\n"
        text += f"z1 = X @ W1 + b1\n"
        for i in range(h):
            text += f"  뉴런{i+1}: z={z1[i]:.4f}  a={act_name}={a1[i]:.4f}\n"
        text += f"\n── Layer 2 (은닉층 → 출력) ──\n"
        text += f"z2 = a1 @ W2 + b2 = {z2[0]:.4f}\n"
        text += f"a2 = Sigmoid(z2) = {a2:.4f}\n"
        text += f"\n최종 출력: {a2:.4f}"
        self.step_text.setText(text)

        # 플롯
        self.fig.clear()
        ax1, ax2, ax3 = self.fig.subplots(1, 3)

        # 네트워크 구조 다이어그램
        ax1.set_xlim(0, 4)
        ax1.set_ylim(-1, max(h, 2) + 0.5)
        ax1.axis('off')
        ax1.set_title('네트워크 구조')
        y_in = [max(h / 2 - 0.5, 0), min(h / 2 + 0.5, h - 1)]
        if h <= 2:
            y_in = [0, h - 1]
        for i, (xi, lbl) in enumerate([(x1, f'x1={x1:.1f}'), (x2, f'x2={x2:.1f}')]):
            yp = y_in[i]
            circ = plt.Circle((0.5, yp), 0.28, color='lightblue', zorder=5)
            ax1.add_patch(circ)
            ax1.text(0.5, yp, lbl, ha='center', va='center', fontsize=7)
            for j in range(h):
                ax1.plot([0.78, 1.72], [yp, j], 'gray', lw=0.5, alpha=0.4)
        for j in range(h):
            circ = plt.Circle((2, j), 0.28, color='lightyellow', zorder=5)
            ax1.add_patch(circ)
            ax1.text(2, j, f'{a1[j]:.2f}', ha='center', va='center', fontsize=7)
            ax1.plot([2.28, 3.22], [j, (h - 1) / 2], 'gray', lw=0.5, alpha=0.4)
        circ = plt.Circle((3.5, (h - 1) / 2), 0.28, color='lightgreen', zorder=5)
        ax1.add_patch(circ)
        ax1.text(3.5, (h - 1) / 2, f'{a2:.3f}', ha='center', va='center', fontsize=8)
        ax1.text(0.5, -0.7, '입력층', ha='center', fontsize=8, color='steelblue')
        ax1.text(2, -0.7, '은닉층', ha='center', fontsize=8, color='goldenrod')
        ax1.text(3.5, -0.7, '출력층', ha='center', fontsize=8, color='seagreen')

        # Layer 1 값
        idx = np.arange(h)
        ax2.bar(idx - 0.2, z1, width=0.35, alpha=0.8, label='z1 (pre)', color='orange')
        ax2.bar(idx + 0.2, a1, width=0.35, alpha=0.8, label=f'a1 ({act_name})', color='royalblue')
        ax2.set_title('Layer 1 값')
        ax2.set_xlabel('뉴런')
        ax2.set_xticks(idx)
        ax2.set_xticklabels([f'N{i+1}' for i in range(h)])
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='k', lw=0.6)

        # Layer 2 값
        ax3.bar(['z2', 'a2 (출력)'], [z2[0], a2], color=['orange', 'seagreen'], alpha=0.85)
        ax3.set_title('Layer 2 값')
        ax3.set_ylabel('값')
        ax3.grid(True, alpha=0.3)
        for bar, val in zip(ax3.patches, [z2[0], a2]):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        self.fig.tight_layout()
        self.canvas.draw()


# ─────────────────────────────────────────────
# 탭 4: MLP (XOR 문제)
# ─────────────────────────────────────────────

class MLPTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)

        panel = QGroupBox("MLP 설정")
        panel.setMaximumWidth(260)
        pl = QVBoxLayout(panel)

        pl.addWidget(QLabel("은닉 뉴런 수:"))
        self.hidden_box = QSpinBox()
        self.hidden_box.setRange(2, 20)
        self.hidden_box.setValue(4)
        pl.addWidget(self.hidden_box)

        pl.addWidget(QLabel("학습률:"))
        self.lr_box = QDoubleSpinBox()
        self.lr_box.setRange(0.01, 2.0)
        self.lr_box.setValue(0.5)
        self.lr_box.setSingleStep(0.1)
        pl.addWidget(self.lr_box)

        pl.addWidget(QLabel("에포크:"))
        self.epoch_box = QSpinBox()
        self.epoch_box.setRange(100, 50000)
        self.epoch_box.setValue(10000)
        self.epoch_box.setSingleStep(1000)
        pl.addWidget(self.epoch_box)

        btn = QPushButton("XOR 학습")
        btn.clicked.connect(self.run)
        pl.addWidget(btn)

        pl.addWidget(QLabel("\n결과:"))
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        pl.addWidget(self.result)
        pl.addStretch()
        layout.addWidget(panel)

        self.fig = Figure(figsize=(13, 4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def run(self):
        h = self.hidden_box.value()
        lr = self.lr_box.value()
        epochs = self.epoch_box.value()

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)

        mlp = MLP(2, h, 1, lr)
        mlp.train(X, y, epochs)

        preds = mlp.predict(X)
        acc = np.mean(preds == y.astype(int)) * 100
        final_loss = mlp.loss_history[-1]

        text = f"은닉 뉴런: {h}개\n"
        text += f"최종 Loss: {final_loss:.6f}\n"
        text += f"정확도: {acc:.1f}%\n\n"
        text += "입력  → 예측 (정답)\n"
        for xi, pred, lbl in zip(X, preds, y):
            ok = "✓" if pred[0] == lbl[0] else "✗"
            text += f"[{int(xi[0])},{int(xi[1])}] → {pred[0]}  ({int(lbl[0])})  {ok}\n"
        text += "\n→ XOR 해결 성공!" if acc == 100 else "\n→ 학습 부족 (에포크 늘려보기)"
        self.result.setText(text)

        self.fig.clear()
        ax1, ax2, ax3 = self.fig.subplots(1, 3)

        # Loss 곡선
        ax1.plot(mlp.loss_history, lw=1.5, color='royalblue')
        ax1.set_title('Training Loss (MSE)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # 결정 경계
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 250), np.linspace(-0.5, 1.5, 250))
        Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        cf = ax2.contourf(xx, yy, Z, levels=25, cmap='RdYlBu', alpha=0.85)
        self.fig.colorbar(cf, ax=ax2, label='출력 확률')
        colors = ['tomato' if lbl[0] == 1 else 'royalblue' for lbl in y]
        markers = ['o' if lbl[0] == 1 else 's' for lbl in y]
        for xi, c, m in zip(X, colors, markers):
            ax2.scatter(xi[0], xi[1], c=c, marker=m, s=280, edgecolors='black', lw=2, zorder=5)
        ax2.set_title('XOR 결정 경계')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.grid(True, alpha=0.3)

        # 은닉층 활성화 히트맵
        _ = mlp.forward(X)
        im = ax3.imshow(mlp.a1.T, cmap='viridis', aspect='auto')
        ax3.set_yticks(range(h))
        ax3.set_yticklabels([f'H{i+1}' for i in range(h)])
        ax3.set_xticks(range(4))
        ax3.set_xticklabels(['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
        ax3.set_title('은닉층 활성화')
        ax3.set_xlabel('입력 패턴')
        self.fig.colorbar(im, ax=ax3)
        for i in range(h):
            for j in range(4):
                ax3.text(j, i, f'{mlp.a1[j, i]:.2f}',
                         ha='center', va='center', color='white', fontsize=7, fontweight='bold')

        self.fig.tight_layout()
        self.canvas.draw()


# ─────────────────────────────────────────────
# 탭 5: 만능 근사 (Universal Approximation)
# ─────────────────────────────────────────────

class UniversalApproxTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)

        panel = QGroupBox("설정")
        panel.setMaximumWidth(260)
        pl = QVBoxLayout(panel)

        pl.addWidget(QLabel("근사할 함수:"))
        self.func_combo = QComboBox()
        self.func_combo.addItems(["Sine Wave", "Step Function", "Complex Function"])
        pl.addWidget(self.func_combo)

        pl.addWidget(QLabel("에포크:"))
        self.epoch_box = QSpinBox()
        self.epoch_box.setRange(1000, 30000)
        self.epoch_box.setValue(8000)
        self.epoch_box.setSingleStep(1000)
        pl.addWidget(self.epoch_box)

        pl.addWidget(QLabel("학습률:"))
        self.lr_box = QDoubleSpinBox()
        self.lr_box.setRange(0.001, 0.1)
        self.lr_box.setValue(0.01)
        self.lr_box.setSingleStep(0.005)
        pl.addWidget(self.lr_box)

        pl.addWidget(QLabel("(뉴런: 3, 10, 50개 자동 비교)"))

        btn = QPushButton("근사 시작")
        btn.clicked.connect(self.run)
        pl.addWidget(btn)

        pl.addWidget(QLabel("\n결과 (MSE):"))
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setMaximumHeight(160)
        pl.addWidget(self.result)

        pl.addWidget(QLabel("\nUniversal Approximation Theorem\n(Cybenko, 1989):"))
        info = QLabel(
            "단일 은닉층 + 충분한 뉴런\n"
            "→ 임의의 연속 함수 근사 가능\n\n"
            "뉴런 많을수록 MSE↓\n"
            "→ 폭(width) vs 깊이(depth)"
        )
        info.setWordWrap(True)
        pl.addWidget(info)
        pl.addStretch()
        layout.addWidget(panel)

        self.fig = Figure(figsize=(13, 4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def _target(self):
        x = np.linspace(0, 1, 200)
        name = self.func_combo.currentText()
        if name == "Sine Wave":
            y = np.sin(2 * np.pi * x)
        elif name == "Step Function":
            y = np.where(x < 0.5, 0.0, 1.0)
        else:
            y = np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x) + 0.25 * np.cos(6 * np.pi * x)
            y = (y - y.min()) / (y.max() - y.min())
        return x, y.astype(float)

    def _train_network(self, X_tr, y_tr, n, epochs, lr):
        np.random.seed(42)
        W1 = np.random.randn(1, n) * 0.5
        b1 = np.random.randn(1, n) * 0.5
        W2 = np.random.randn(n, 1) * 0.5
        b2 = np.zeros((1, 1))
        m = len(X_tr)
        for _ in range(epochs):
            z1 = X_tr @ W1 + b1
            a1 = sigmoid(z1)
            y_pred = a1 @ W2 + b2
            dz2 = (y_pred - y_tr) / m
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, keepdims=True)
            da1 = dz2 @ W2.T
            dz1 = da1 * sigmoid_derivative(z1)
            dW1 = X_tr.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1
        z1 = X_tr @ W1 + b1
        a1 = sigmoid(z1)
        return (a1 @ W2 + b2).ravel()

    def run(self):
        x, y_target = self._target()
        epochs = self.epoch_box.value()
        lr = self.lr_box.value()
        X_tr = x.reshape(-1, 1)
        y_tr = y_target.reshape(-1, 1)

        self.fig.clear()
        axes = self.fig.subplots(1, 3)
        neuron_counts = [3, 10, 50]
        text = f"함수: {self.func_combo.currentText()}\n\n"

        for ax, n in zip(axes, neuron_counts):
            y_pred = self._train_network(X_tr, y_tr, n, epochs, lr)
            mse = np.mean((y_pred - y_target) ** 2)
            ax.plot(x, y_target, 'b-', lw=2.5, label='목표 함수')
            ax.plot(x, y_pred, 'r--', lw=2, label=f'신경망')
            ax.set_title(f'{n}개 뉴런\nMSE: {mse:.5f}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            text += f"{n:>3}개 뉴런: MSE = {mse:.6f}\n"

        self.fig.suptitle(
            f"Universal Approximation — {self.func_combo.currentText()}",
            fontsize=13, fontweight='bold'
        )
        self.fig.tight_layout()
        self.canvas.draw()
        self.result.setText(text)


# ─────────────────────────────────────────────
# 메인 윈도우
# ─────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week3: 신경망 기초 탐색기  |  PySide6 + NumPy + Matplotlib")
        self.setMinimumSize(1200, 720)

        font = QFont()
        font.setPointSize(10)
        QApplication.setFont(font)

        tabs = QTabWidget()
        tabs.addTab(PerceptronTab(),         "1. 퍼셉트론")
        tabs.addTab(ActivationFunctionsTab(), "2. 활성화 함수")
        tabs.addTab(ForwardPropTab(),         "3. 순전파")
        tabs.addTab(MLPTab(),                 "4. MLP (XOR)")
        tabs.addTab(UniversalApproxTab(),     "5. 만능 근사")
        self.setCentralWidget(tabs)

        self.statusBar().showMessage(
            "Week3 신경망 기초 탐색기  |  부산대 AI와인공지능  |  이태영"
        )


# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
