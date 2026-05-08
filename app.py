from PySide6.QtGui import QRegularExpressionValidator, QColor
from PySide6.QtCore import Qt, QRegularExpression
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
)
import sys
import numpy as np
from abc import ABC, abstractmethod


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
    """таблица результатов (для чтения)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        # запрет на редактирование
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def display_results(self, data_dict: dict):
        """
        принимает словарь из out_table_data и сторит таблицу
        """
        col_data = data_dict["columns"]
        col_names = list(col_data.keys())
        # названия стлбцов
        n_points = len(col_data[col_names[0]])
        
        self.setRowCount(n_points + 2)
        self.setColumnCount(len(col_names))
        self.setHorizontalHeaderLabels(col_names)
        # настройка сетки таблицы
        
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
            self.setItem(sum_row, col_idx, item)
            
        mean_row = n_points + 1
        for col_idx, col_name in enumerate(col_names):
            if col_name in data_dict["means"]:
                val = data_dict["means"][col_name]
                item = QTableWidgetItem(f"<{col_name}>={val:.4f}")
            else:
                item = QTableWidgetItem("-") # при остуствии средних
                
            item.setTextAlignment(Qt.AlignCenter)
            item.setBackground(QColor(255, 255, 200))
            self.setItem(mean_row, col_idx, item)

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("lab_helper")
        self.resize(800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        self.table = InputTable()
        self.table.set_columns(["x", "y"])
        main_layout.addWidget(self.table)
        
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("добавить строку")
        self.btn_add.clicked.connect(self.table.add_data_row)
        
        self.btn_remove = QPushButton("удалить строку")
        self.btn_remove.clicked.connect(self.table.remove_data_row)
        
        self.btn_calc = QPushButton("рассчитать МНК")
        self.btn_calc.clicked.connect(self.calculate_mnk)
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_calc)
        main_layout.addLayout(btn_layout)

        self.result_table = ResultTable()
        main_layout.addWidget(self.result_table)

        self.table.add_data_row()
        self.table.add_data_row()
        self.table.add_data_row()

    def calculate_mnk(self):
        try:
            x_data = self.table.get_column_data(1)
            y_data = self.table.get_column_data(2)
            
            lsm_solver = LSM(x_data, y_data)
            a, b, sigma_a, sigma_b = lsm_solver.calculate()
            table_dict = lsm_solver.out_table_data()
            
            self.result_table.display_results(table_dict)

        except DataError as e:
            print(f"ошибка данных: {e}")
        except ValueError as e:
            print(f"ошибка ввода: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())