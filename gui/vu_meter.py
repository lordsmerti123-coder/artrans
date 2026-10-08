# gui/vu_meter.py
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QLinearGradient

class VUMeter(QWidget):
    """Кастомный VU-метр с отображением текущего уровня и пика."""
    def __init__(self, parent=None, orientation=Qt.Orientation.Vertical):
        super().__init__(parent)
        self._level = 0.0          # 0..1
        self._peak = 0.0
        self._peak_decay = 0.995   # коэффициент затухания пика
        self._orientation = orientation
        self.setMinimumSize(20 if orientation == Qt.Orientation.Vertical else 100,
                            100 if orientation == Qt.Orientation.Vertical else 20)
        self.setMaximumSize(30 if orientation == Qt.Orientation.Vertical else 150,
                            150 if orientation == Qt.Orientation.Vertical else 30)

    def set_level(self, level: float):
        """Установить текущий уровень (0..1)."""
        self._level = max(0.0, min(1.0, level))
        if self._level > self._peak:
            self._peak = self._level
        else:
            self._peak *= self._peak_decay
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        if self._orientation == Qt.Orientation.Vertical:
            # Рисуем снизу вверх
            bar_height = int(self._level * rect.height())
            peak_y = int((1 - self._peak) * rect.height())
            # Градиент: зелёный -> жёлтый -> красный
            gradient = QLinearGradient(0, rect.bottom(), 0, rect.top())
            gradient.setColorAt(0.0, QColor(0, 255, 0))
            gradient.setColorAt(0.7, QColor(255, 255, 0))
            gradient.setColorAt(1.0, QColor(255, 0, 0))

            # Заливка фона (тёмный)
            painter.fillRect(rect, QColor(20, 20, 25))
            # Заливка уровня
            painter.fillRect(0, rect.bottom() - bar_height, rect.width(), bar_height, gradient)
            # Пик
            painter.setPen(QColor(255, 255, 255))
            painter.drawLine(0, peak_y, rect.width(), peak_y)
            # Обводка
            painter.setPen(QColor(60, 60, 70))
            painter.drawRect(rect)
        else:
            # Горизонтальный (не используется, но для полноты)
            bar_width = int(self._level * rect.width())
            peak_x = int(self._peak * rect.width())
            gradient = QLinearGradient(0, 0, rect.right(), 0)
            gradient.setColorAt(0.0, QColor(0, 255, 0))
            gradient.setColorAt(0.7, QColor(255, 255, 0))
            gradient.setColorAt(1.0, QColor(255, 0, 0))

            painter.fillRect(rect, QColor(20, 20, 25))
            painter.fillRect(0, 0, bar_width, rect.height(), gradient)
            painter.setPen(QColor(255, 255, 255))
            painter.drawLine(peak_x, 0, peak_x, rect.height())
            painter.setPen(QColor(60, 60, 70))
            painter.drawRect(rect)