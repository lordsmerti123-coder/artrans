# gui/main_window.py
import sys
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QStatusBar
from PyQt6.QtCore import Qt
from gui.main_tab import MainTab
from gui.settings_tab import SettingsTab
from gui.log_tab import LogTab
from gui.controller import Controller

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BabelDuo – двусторонний переводчик")
        self.setGeometry(100, 100, 1200, 800)

        self.controller = Controller()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.main_tab = MainTab(self.controller)
        self.tabs.addTab(self.main_tab, "Основная")

        self.settings_tab = SettingsTab(self.controller)
        self.tabs.addTab(self.settings_tab, "Настройки")

        self.log_tab = LogTab(self.controller)
        self.tabs.addTab(self.log_tab, "Логи")

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе")

        self.controller.log_signal.connect(self.status_bar.showMessage)
        self.controller.status_signal.connect(self.status_bar.showMessage)

    def closeEvent(self, event):
        self.main_tab.stop_all_meters()
        self.controller.stop_all()
        event.accept()