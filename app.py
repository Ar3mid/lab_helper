from PySide6.QtGui import QRegularExpressionValidator, QColor
from PySide6.QtCore import Qt, QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator, QColor, QBrush
import os
from PySide6.QtCore import Qt, QRegularExpression, QStandardPaths
from PySide6.QtWidgets import (
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QApplication, 
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QFileDialog,
    QTextEdit,
)
import sys
import numpy as np
from abc import ABC, abstractmethod
import matplotlib
matplotlib.use('QtAgg') # использование движка qt для отрисовки графика
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
from PySide6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox


class NumDelegate(QStyledItemDelegate):
    """
    менеджер для ячеек таблицы, центрирует пользовательский ввод и не дает вводить что то, кроме цифр,
    точки, запятой, знака минуса
    """

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setAlignment(Qt.AlignCenter)
        # центровка во время ввода
        regex = QRegularExpression(r"^-?\d+([.,]\d+)?$")
        # разрешение регулярным выражением
        validator = QRegularExpressionValidator(regex)
        editor.setValidator(validator)
        return editor

    def initStyleOption(self, option, index):
        """настройка внешнего вида ячейки"""
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter
        # центровка введенного текста


class InputTable(QTableWidget):
    """
    кастомная таблица для ввода данных
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # super используется, чтобы не переписывать код в случае смены класса
        # parent=None для возможности существования таблицы как самостоятельного объекта,
        # так и части дерева наследования редактора ui
        self._setup_base()

    def _setup_base(self):
        """
        создание таблицы c первым столбцом по умолчанию
        """
        self.setColumnCount(1)
        self.setHorizontalHeaderLabels(["№"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # растяжение таблицы на все возможное пространство
        self.setItemDelegate(NumDelegate(self))
    
    def _create_number_item(self, number_str: str):
        """
        вспомогательный метод для избегания повторов
        """
        item = QTableWidgetItem(number_str)
        item.setTextAlignment(Qt.AlignCenter)
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        return item

    def set_columns(self, column_names: list[str]):
        """
        пользовательская настройка столбцов
        """
        self.setRowCount(0)
        # сброс таблицы
        self.setColumnCount(len(column_names) + 1)
        headers = ["№"] + column_names
        self.setHorizontalHeaderLabels(headers)

    def add_data_row(self):
        """
        добавление строки, нумерация строк в первом столбце
        """
        row_index = self.rowCount()
        self.insertRow(row_index)
        num_item = QTableWidgetItem(str(row_index + 1))
        # создание ячейки с номером измерения
        self.setItem(row_index, 0, num_item)
        # установка ячейки в 1 столбец

    def remove_data_row(self):
        """
        удаление выделенной строки, если нет выделения, удаляет последнюю
        """
        if self.rowCount() == 0:
            return
        current_row = self.currentRow()
        if current_row >= 0:
            self.removeRow(current_row)
        else:
            self.removeRow(self.rowCount() - 1)
        self._update_numbering()

    def keyPressEvent(self, event):
        """
        удалениеданных из ячейки
        """
        if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            selected_items = self.selectedItems()
            for item in selected_items:
                if item.column() != 0:
                    # проверка, чтобы не стереть нумерацию
                    item.setText("")
        else:
            # чтобы не ломать остальные клавишы, передаем их родительскому классу
            super().keyPressEvent(event)

    def _update_numbering(self):
        """
        пересчет нумерации
        """
        for row in range(self.rowCount()):
            num_item = QTableWidgetItem(str(row + 1))
            self.setItem(row, 0, num_item)

    def get_column_data(self, col_index):
        """
        получение данных из столбца с игнорированием пустых ячеек
        """
        data = []
        for row in range(self.rowCount()):
            item = self.item(row, col_index)
            if item and item.text().strip():
                val_str = item.text().replace(",", ".")
                data.append(float(val_str))
        return data

class DataError(Exception):
    """недостаточно данных"""
    # кастомная ошибка для недостаточного количества данных
    pass

class BaseApprox(ABC):
    """Абстрактный класс для методов аппроксимации"""
    def __init__(self, x, y):
        if len(x) != len(y):
            raise ValueError("разное количество x и y")
        if len(x) < 3:
            raise DataError("необходимо минимум 3 точки")
        self._x = np.array(x, dtype=float)
        self._y = np.array(y, dtype=float)
        self._n = len(x)
        #преобразование массивов в numpy, проверка количсетва данных

    @abstractmethod #для каждого наследника свой расчет
    def calculate(self):
        """метод для переопределения"""
        pass

    def __len__(self):
        """для применения len к объекту"""
        return self._n

class LSM(BaseApprox):
    """метод наименьших квадратов"""
    def __init__(self, x: list[float], y: list[float]):
        super().__init__(x, y)
        self.a = 0.0
        self.b = 0.0
        self.sigma_a = 0.0
        self.sigma_b = 0.0
        #резервация места под коэффициенты прямой и их погрешности
    def calculate(self):
        """расчет коэффициентов a и b и их погрешностей"""
        x_mean = np.mean(self._x)
        y_mean = np.mean(self._y)
        #среднее значение
        x2_mean = np.mean(self._x**2)
        y2_mean = np.mean(self._y**2)
        xy_mean = np.mean(self._x * self._y)
        #среднее квадратов и произведения
        sigma2_x = x2_mean - x_mean**2
        sigma2_y = y2_mean - y_mean**2
        #дисперсия
        if sigma2_x == 0:
            raise ValueError("все x одинаковы, невозможно построить прямую")
        #исключение для деления на 0
        self.a = (xy_mean - x_mean * y_mean) / sigma2_x
        self.b = y_mean - self.a * x_mean
        #расчет коэффициентов
        val = max(0, (sigma2_y / sigma2_x) - self.a**2)
        self.sigma_a = np.sqrt((1 / (self._n - 2)) * val)
        self.sigma_b = self.sigma_a * np.sqrt(x2_mean)
        return self.a, self.b, self.sigma_a, self.sigma_b
        #расчет погрешностей
        #max для исключения ошибки, когда значение должно быть равно 0,
        #но вывод -0.0...01

    def out_table_data(self):
        """вывод данных"""
        x_mean = np.mean(self._x)
        y_mean = np.mean(self._y)
        #вычисление столбцов
        dx = self._x - x_mean            # (x_i - <x>)
        dy = self._y - y_mean            # (y_i - <y>)
        dx_sq = dx**2                    # (x_i - <x>)^2
        dy_sq = dy**2                    # (y_i - <y>)^2
        xy = self._x * self._y           # x_i * y_i
        return {
            "columns": {
                "x": self._x,
                "y": self._y,
                "x - <x>": dx,
                "y - <y>": dy,
                "(x - <x>)^2": dx_sq,
                "(y - <y>)^2": dy_sq,
                "x*y": xy
            },
            "sums": {
                "x": np.sum(self._x),
                "y": np.sum(self._y),
                "x - <x>": np.sum(dx),
                "y - <y>": np.sum(dy),
                "(x - <x>)^2": np.sum(dx_sq),
                "(y - <y>)^2": np.sum(dy_sq),
                "x*y": np.sum(xy)
            },
            "means": {
                "x": x_mean,
                "y": y_mean
            }
            #словари для ui
        }

class ResultTable(QTableWidget):
    """таблица результатов"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        # запрет на редактирование
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setSelectionMode(QTableWidget.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def display_results(self, data_dict: dict):
        """строит таблицу"""
        col_data = data_dict["columns"]
        col_names = list(col_data.keys())
        n_points = len(col_data[col_names[0]])
        
        self.setRowCount(n_points + 2)
        self.setColumnCount(len(col_names))
        self.setHorizontalHeaderLabels(col_names)

        for row in range(n_points):
            for col_idx, col_name in enumerate(col_names):
                val = col_data[col_name][row]
                item = QTableWidgetItem(f"{val:.4f}")
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(row, col_idx, item)

        sum_row = n_points
        for col_idx, col_name in enumerate(col_names):
            val = data_dict["sums"][col_name]
            item = QTableWidgetItem(f"∑={val:.4f}")
            item.setTextAlignment(Qt.AlignCenter)
            item.setBackground(QColor(240, 240, 240))
            item.setForeground(QBrush(QColor(0, 0, 0)))
            self.setItem(sum_row, col_idx, item)
            
        mean_row = n_points + 1
        for col_idx, col_name in enumerate(col_names):
            if col_name in data_dict["means"]:
                val = data_dict["means"][col_name]
                item = QTableWidgetItem(f"<{col_name}>={val:.4f}")
            else:
                item = QTableWidgetItem("-") 
                
            item.setTextAlignment(Qt.AlignCenter)
            item.setBackground(QColor(255, 255, 200))
            item.setForeground(QBrush(QColor(0, 0, 0)))
            
            self.setItem(mean_row, col_idx, item)

class AxisSettingsDialog(QDialog):
    """окно для переименования осей"""
    
    def __init__(self, current_x: str, current_y: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("названия осей")
        layout = QFormLayout(self)
        self.x_edit = QLineEdit(current_x)
        self.y_edit = QLineEdit(current_y)
        layout.addRow("Название оси X:", self.x_edit)
        layout.addRow("Название оси Y:", self.y_edit)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        
        layout.addWidget(self.buttons)

    def get_labels(self):
        """извлечение текста после закрытия окна"""
        return self.x_edit.text(), self.y_edit.text()


class LogDialog(QDialog):
    """вывод расчетов и ошибок"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Журнал расчетов (Консоль)")
        self.resize(450, 300)
        
        layout = QVBoxLayout(self)
        
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("font-family: Consolas, Courier New; font-size: 11pt;")
        layout.addWidget(self.text_area)
        self.btn_close = QPushButton("закрыть")
        self.btn_close.clicked.connect(self.accept)
        layout.addWidget(self.btn_close)

    def print_msg(self, message: str, is_error: bool = False):
        """вывод"""
        if is_error:
            self.text_area.append(f"<span style='color: #FF5555;'><b>[ОШИБКА]</b> {message}</span>")
        else:
            self.text_area.append(f"<span style='color: white;'>{message}</span>")
        scrollbar = self.text_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class PlotWindow(QWidget):
    """окно графика"""

    def __init__(self, x: list[float], y: list[float], a: float = None, b: float = None):
        super().__init__()
        self.setWindowTitle("график экспериментальной зависимости")
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.x_label = "Ось X"
        self.y_label = "Ось Y"
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        
        self.plot_data(x, y, a, b)
        
        self.btn_settings = QPushButton("переименовать оси")
        self.btn_settings.clicked.connect(self.open_settings)
        layout.addWidget(self.btn_settings)
        self.btn_save = QPushButton("сохранить график в .png")
        self.btn_save.clicked.connect(self.save_plot)
        layout.addWidget(self.btn_save)

    def plot_data(self, x, y, a, b):
        """метод построения точек, сетки и прямой"""
        self.ax.clear()
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        self.ax.grid(which='major', color="#333131", linewidth=0.8)
        self.ax.grid(which='minor', color="#BDBDBD", linestyle='-', linewidth=0.5)
        self.ax.scatter(x, y, color='red', marker='o', s=50, label='Эксперимент', zorder=5)
        
        if a is not None and b is not None:
            x_min, x_max = min(x), max(x)
            margin = (x_max - x_min) * 0.1
            if margin == 0: margin = 1
            x_line = np.array([x_min - margin, x_max + margin])
            y_line = a * x_line + b
            sign = "+" if b >= 0 else "-"
            label_text = f'Тренд: y = {a:.4f}x {sign} {abs(b):.4f}'
            self.ax.plot(x_line, y_line, color='blue', linestyle='--', linewidth=2, label=label_text, zorder=4)
            

        self.ax.set_xlabel(self.x_label, fontsize=12)
        self.ax.set_ylabel(self.y_label, fontsize=12)
        self.ax.set_title("График МНК", fontsize=14, fontweight='bold')
        self.ax.legend()
        self.canvas.draw()
        
    def open_settings(self):
        """вызов диалогового окна и перерисовка графика"""
        dialog = AxisSettingsDialog(self.x_label, self.y_label, self)
        if dialog.exec() == QDialog.Accepted:
            new_x, new_y = dialog.get_labels()
            self.x_label = new_x
            self.y_label = new_y
            self.ax.set_xlabel(self.x_label, fontsize=12)
            self.ax.set_ylabel(self.y_label, fontsize=12)
            self.canvas.draw()
    def save_plot(self):
        """сохранение графика"""
        downloads_folder = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
        
        # если нет пути к загрузке, сохранение в домашнюю папку
        if not downloads_folder:
            downloads_folder = os.path.expanduser("~")
        default_path = os.path.join(downloads_folder, "mnk_plot.png")
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Сохранить график", 
            default_path, 
            "PNG Картинки (*.png);;JPEG Картинки (*.jpg);;Все файлы (*)"
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=600, bbox_inches='tight')
                QMessageBox.information(self, "успех", f"график сохранен:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "ошибка", f"график не сохранен:\n{e}")


class MainWindow(QMainWindow):
    """главное окно"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("анализ данных методом наименьших квадратов")
        self.resize(900, 700) # Сделали чуть просторнее
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.table = InputTable()
        self.table.set_columns(["x", "y"])
        main_layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("добавить строку")
        self.btn_add.clicked.connect(self.table.add_data_row)
        self.btn_remove = QPushButton("ндалить строку")
        self.btn_remove.clicked.connect(self.table.remove_data_row)
        self.btn_calc = QPushButton("таблица МНК")
        self.btn_calc.clicked.connect(self.calculate_mnk)
        self.btn_plot = QPushButton("построить график")
        self.btn_plot.clicked.connect(self.show_plot)
        self.btn_plot.setEnabled(False)
        self.log_window = LogDialog(self)
        self.btn_log = QPushButton("Открыть консоль")
        self.btn_log.clicked.connect(self.log_window.show)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_calc)
        btn_layout.addWidget(self.btn_plot)
        btn_layout.addWidget(self.btn_log)
        main_layout.addLayout(btn_layout)

        self.result_table = ResultTable()
        main_layout.addWidget(self.result_table)
        
        self.table.add_data_row()
        self.table.add_data_row()
        self.table.add_data_row()

    def calculate_mnk(self):
        """сбор, расчет, вывод"""
        self.log_window.print_msg("-" * 30)
        self.log_window.print_msg("расчет...")
        try:
            x_data = self.table.get_column_data(1)
            y_data = self.table.get_column_data(2)
            lsm_solver = LSM(x_data, y_data)
            a, b, sigma_a, sigma_b = lsm_solver.calculate()
            table_dict = lsm_solver.out_table_data()
            
            self.result_table.display_results(table_dict)

            self.plot_data_x = x_data
            self.plot_data_y = y_data
            self.plot_a = a
            self.plot_b = b
            
            self.btn_plot.setEnabled(True)

            sign = "+" if b >= 0 else "-"
            self.log_window.print_msg("успех! уравнение прямой:")
            self.log_window.print_msg(f"y = {a:.4f}x {sign} {abs(b):.4f}")
            self.log_window.print_msg(f"погрешность σa = {sigma_a:.4f}")
            self.log_window.print_msg(f"погрешность σb = {sigma_b:.4f}")
            
            self.log_window.show()

        except DataError as e:
            self.log_window.print_msg(str(e), is_error=True)
            self.log_window.show()
        except ValueError as e:
            self.log_window.print_msg(str(e), is_error=True)
            self.log_window.show()
            
    def show_plot(self):
        """окно с графиком"""
        self.plot_window = PlotWindow(
            self.plot_data_x, 
            self.plot_data_y, 
            self.plot_a, 
            self.plot_b
        )
        self.plot_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = MainWindow()
    window.show()
    sys.exit(app.exec())