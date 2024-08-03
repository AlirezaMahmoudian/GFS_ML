#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import random
from random import shuffle, randint as rnd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGroupBox, QVBoxLayout, QHBoxLayout
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from io import BytesIO
from PIL import Image, ImageQt
from matplotlib.figure import Figure
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import shap

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1283, 842)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1283, 842))
        self.tabWidget.setObjectName("tabWidget")

        # Tab 1: Import Dataset
        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")
        self.setupTab1(self.tab1)
        self.tabWidget.addTab(self.tab1, "Import Dataset")

        # Tab 2: Train
        self.tab2 = QtWidgets.QWidget()
        self.tab2.setObjectName("tab2")
        self.setupTab2(self.tab2)
        self.tabWidget.addTab(self.tab2, "Train")

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

    def setupTab1(self, tab):
        self.toolButton = QtWidgets.QToolButton(tab)
        self.toolButton.setGeometry(QtCore.QRect(50, 30, 201, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.toolButton.setFont(font)
        self.toolButton.setObjectName("toolButton")

        self.additionalButton1 = QtWidgets.QToolButton(tab)
        self.additionalButton1.setGeometry(QtCore.QRect(261, 30, 201, 51))
        self.additionalButton1.setFont(font)
        self.additionalButton1.setObjectName("additionalButton1")

        self.additionalButton2 = QtWidgets.QToolButton(tab)
        self.additionalButton2.setGeometry(QtCore.QRect(472, 30, 201, 51))
        self.additionalButton2.setFont(font)
        self.additionalButton2.setObjectName("additionalButton2")

        self.additionalButton3 = QtWidgets.QToolButton(tab)
        self.additionalButton3.setGeometry(QtCore.QRect(683, 30, 201, 51))
        self.additionalButton3.setFont(font)
        self.additionalButton3.setObjectName("additionalButton3")

        self.tableWidget = QtWidgets.QTableWidget(tab)
        self.tableWidget.setGeometry(QtCore.QRect(50, 100, 1211, 731))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)

        self.toolButton.clicked.connect(self.loadFile)
        self.additionalButton1.clicked.connect(self.preprocessData)
        self.additionalButton2.clicked.connect(self.showScatterPlot)
        self.additionalButton3.clicked.connect(self.showHeatmap)

        self.retranslateTab1(tab)

    def loadFile(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "", "All Files (*);;CSV Files (*.csv);;Excel Files (*.xlsx)", options=options)
        if fileName:
            try:
                if fileName.endswith('.csv'):
                    self.df = pd.read_csv(fileName)
                elif fileName.endswith('.xlsx'):
                    self.df = pd.read_excel(fileName)
                else:
                    raise ValueError("Unsupported file format")

                self.displayDataFrame(self.df)
                self.lineEdit_2.setText(str(self.df.shape[1] - 1))  # Update NumberOfQueens (features) in lineEdit_2
                self.createFeatureInputs()  # Create input fields and prediction button
            except Exception as e:
                self.showErrorMessage(str(e))

    def preprocessData(self):
        print("Preprocess button clicked")  # Debug statement
        if hasattr(self, 'df'):
            try:
                df = self.df.copy()

                # Identify text features
                text_features = df.select_dtypes(include=['object']).columns.tolist()
                if text_features:
                    label_encoder = LabelEncoder()
                    for feature in text_features:
                        df[feature] = label_encoder.fit_transform(df[feature])
            
                # Debug statement to confirm text feature conversion
                print("Text feature conversion done")
                print(df.head())  # Print the dataframe to see changes

                # Normalize the data if the radio button is checked
                if self.radioButton.isChecked():
                    scaler = StandardScaler()
                    df[df.columns] = scaler.fit_transform(df)
                
                    # Debug statement to confirm normalization
                    print("Normalization done")
                    print(df.head())  # Print the dataframe to see changes

                # Clear the table and display the preprocessed DataFrame
                self.tableWidget.clear()
                self.displayDataFrame(df)
                self.df = df  # Update the stored DataFrame
                print("Preprocessing completed and table updated")  # Debug statement
            except Exception as e:
                self.showErrorMessage(str(e))
        else:
            self.showErrorMessage("No dataset loaded. Please import a dataset first.")

    def displayDataFrame(self, df):
        self.tableWidget.clear()  # Clear existing data and headers
        self.tableWidget.setRowCount(df.shape[0])
        self.tableWidget.setColumnCount(df.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(df.columns)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.tableWidget.setItem(i, j, QtWidgets.QTableWidgetItem(str(df.iat[i, j])))

        self.tableWidget.viewport().update()  # Force the table to update
        print("Table displayed with updated data")  # Debug statement

    def showErrorMessage(self, message):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText("Error")
        msgBox.setInformativeText(message)
        msgBox.setWindowTitle("Error")
        msgBox.exec_()

    def showScatterPlot(self):
        if hasattr(self, 'df'):
            try:
                sns.pairplot(self.df,
                             plot_kws={'color': 'green', 'marker': 's'},
                             diag_kws={'color': 'red'})

                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                img = img.resize((800, 600), Image.LANCZOS)
                imgQt = ImageQt.ImageQt(img)

                msgBox = QMessageBox()
                msgBox.setIconPixmap(QtGui.QPixmap.fromImage(imgQt))
                msgBox.setWindowTitle("Scatter Plot")
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.exec_()

                plt.close()

            except Exception as e:
                self.showErrorMessage(str(e))

    def showHeatmap(self):
        if hasattr(self, 'df'):
            try:
                corr = self.df.corr()

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, cmap='RdPu', annot=True, fmt=".2f", annot_kws={"size": 8})
                fig.set_dpi(1000)

                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                img = img.resize((800, 600), Image.LANCZOS)
                imgQt = ImageQt.ImageQt(img)

                msgBox = QMessageBox()
                msgBox.setIconPixmap(QtGui.QPixmap.fromImage(imgQt))
                msgBox.setWindowTitle("Heatmap Correlation")
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.exec_()

                plt.close()

            except Exception as e:
                self.showErrorMessage(str(e))

    def retranslateTab1(self, tab):
        _translate = QtCore.QCoreApplication.translate
        tab.setWindowTitle(_translate("tab", "Import Dataset"))
        self.toolButton.setText(_translate("tab", "Import Dataset"))
        self.additionalButton1.setText(_translate("tab", "Preprocessing"))
        self.additionalButton2.setText(_translate("tab", "Scatter Plot"))
        self.additionalButton3.setText(_translate("tab", "Heatmap Correlation"))

    def setupTab2(self, tab):
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        tab.setFont(font)

        # Create a button group for the primary set of radio buttons
        self.buttonGroup = QtWidgets.QButtonGroup(tab)

        self.radioButton = QtWidgets.QRadioButton(tab)
        self.radioButton.setGeometry(QtCore.QRect(38, 70, 231, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton.setFont(font)
        self.radioButton.setObjectName("radioButton")
        self.buttonGroup.addButton(self.radioButton)  # Add to button group

        self.label = QtWidgets.QLabel(tab)
        self.label.setGeometry(QtCore.QRect(60, 20, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(tab)
        self.label_2.setGeometry(QtCore.QRect(38, 210, 351, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(tab)
        self.label_3.setGeometry(QtCore.QRect(48, 260, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        self.radioButton_2 = QtWidgets.QRadioButton(tab)
        self.radioButton_2.setGeometry(QtCore.QRect(338, 70, 261, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setObjectName("radioButton_2")
        self.buttonGroup.addButton(self.radioButton_2)  # Add to button group

        self.radioButton_4 = QtWidgets.QRadioButton(tab)
        self.radioButton_4.setGeometry(QtCore.QRect(668, 70, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton_4.setFont(font)
        self.radioButton_4.setObjectName("radioButton_4")
        self.buttonGroup.addButton(self.radioButton_4)  # Add to button group

        self.radioButton_5 = QtWidgets.QRadioButton(tab)
        self.radioButton_5.setGeometry(QtCore.QRect(958, 70, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton_5.setFont(font)
        self.radioButton_5.setObjectName("radioButton_5")
        self.buttonGroup.addButton(self.radioButton_5)  # Add to button group

        self.radioButton_6 = QtWidgets.QRadioButton(tab)
        self.radioButton_6.setGeometry(QtCore.QRect(958, 120, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton_6.setFont(font)
        self.radioButton_6.setObjectName("radioButton_6")
        self.buttonGroup.addButton(self.radioButton_6)  # Add to button group

        self.radioButton_3 = QtWidgets.QRadioButton(tab)
        self.radioButton_3.setGeometry(QtCore.QRect(338, 120, 261, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton_3.setFont(font)
        self.radioButton_3.setObjectName("radioButton_3")
        self.buttonGroup.addButton(self.radioButton_3)  # Add to button group

        self.radioButton_7 = QtWidgets.QRadioButton(tab)
        self.radioButton_7.setGeometry(QtCore.QRect(668, 120, 231, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton_7.setFont(font)
        self.radioButton_7.setObjectName("radioButton_7")
        self.buttonGroup.addButton(self.radioButton_7)  # Add to button group

        self.radioButton_8 = QtWidgets.QRadioButton(tab)
        self.radioButton_8.setGeometry(QtCore.QRect(38, 120, 231, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton_8.setFont(font)
        self.radioButton_8.setObjectName("radioButton_8")
        self.buttonGroup.addButton(self.radioButton_8)  # Add to button group

        self.line = QtWidgets.QFrame(tab)
        self.line.setGeometry(QtCore.QRect(-12, 170, 1271, 16))
        self.line.setLineWidth(3)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.label_4 = QtWidgets.QLabel(tab)
        self.label_4.setGeometry(QtCore.QRect(48, 310, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        self.label_5 = QtWidgets.QLabel(tab)
        self.label_5.setGeometry(QtCore.QRect(48, 360, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")

        self.horizontalSlider = QtWidgets.QSlider(tab)
        self.horizontalSlider.setGeometry(QtCore.QRect(218, 370, 160, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setRange(0, 100)  # Set the range from 0 to 100
        self.horizontalSlider.setSingleStep(1)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.valueChanged.connect(self.updateSliderValue)

        self.label_slider_value = QtWidgets.QLabel(tab)
        self.label_slider_value.setGeometry(QtCore.QRect(390, 370, 60, 22))
        self.label_slider_value.setText("0.00")
        self.label_slider_value.setObjectName("label_slider_value")

        self.lineEdit_3 = QtWidgets.QLineEdit(tab)
        self.lineEdit_3.setGeometry(QtCore.QRect(218, 410, 61, 31))
        self.lineEdit_3.setText("")
        self.lineEdit_3.setObjectName("lineEdit_3")

        self.label_6 = QtWidgets.QLabel(tab)
        self.label_6.setGeometry(QtCore.QRect(48, 410, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")

        self.line_2 = QtWidgets.QFrame(tab)
        self.line_2.setGeometry(QtCore.QRect(478, 180, 20, 661))
        self.line_2.setLineWidth(3)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")

        self.label_12 = QtWidgets.QLabel(tab)
        self.label_12.setGeometry(QtCore.QRect(810, 190, 71, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")

        self.pushButton_2 = QtWidgets.QPushButton(tab)
        self.pushButton_2.setGeometry(QtCore.QRect(40, 470, 241, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")

        self.textEdit = QtWidgets.QTextEdit(tab)
        self.textEdit.setGeometry(QtCore.QRect(40, 590, 401, 101))
        self.textEdit.setObjectName("textEdit")

        self.label_14 = QtWidgets.QLabel(tab)
        self.label_14.setGeometry(QtCore.QRect(40, 540, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")

        self.radioButton_9 = QtWidgets.QRadioButton(tab)
        self.radioButton_9.setGeometry(QtCore.QRect(50, 710, 231, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.radioButton_9.setFont(font)
        self.radioButton_9.setObjectName("radioButton_9")

        self.lineEdit_3 = QtWidgets.QLineEdit(tab)
        self.lineEdit_3.setGeometry(QtCore.QRect(220, 410, 61, 31))
        self.lineEdit_3.setObjectName("lineEdit_3")

        self.pushButton_3 = QtWidgets.QPushButton(tab)
        self.pushButton_3.setGeometry(QtCore.QRect(50, 770, 241, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")

        self.pushButton_4 = QtWidgets.QPushButton(tab)
        self.pushButton_4.setGeometry(QtCore.QRect(320, 770, 241, 41))
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")

        self.pushButton_reset = QtWidgets.QPushButton(tab)
        self.pushButton_reset.setGeometry(QtCore.QRect(590, 770, 241, 41))
        self.pushButton_reset.setFont(font)
        self.pushButton_reset.setObjectName("pushButton_reset")

        self.lineEdit = QtWidgets.QLineEdit(tab)
        self.lineEdit.setGeometry(QtCore.QRect(220, 270, 71, 31))
        self.lineEdit.setObjectName("lineEdit")

        self.lineEdit_2 = QtWidgets.QLineEdit(tab)
        self.lineEdit_2.setGeometry(QtCore.QRect(220, 310, 71, 31))
        self.lineEdit_2.setObjectName("lineEdit_2")

        self.groupBox = QGroupBox("Model Input and Prediction", tab)
        self.groupBox.setGeometry(QtCore.QRect(510, 250, 571, 461))
        self.groupBox.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        layout = QVBoxLayout()
        self.groupBox.setLayout(layout)

        self.retranslateTab2(tab)
        QtCore.QMetaObject.connectSlotsByName(tab)

        self.pushButton_2.clicked.connect(self.trainModel)
        self.pushButton_3.clicked.connect(self.showResults)
        self.pushButton_4.clicked.connect(self.showShapleyValues)
        self.pushButton_reset.clicked.connect(self.resetAll)

    def updateSliderValue(self, value):
        scaled_value = value / 100.0
        self.label_slider_value.setText(f"{scaled_value:.2f}")

    def trainModel(self):
        if hasattr(self, 'df'):
            try:
                df = self.df.copy()
                y = df.iloc[:, -1].values
                X = df.iloc[:, :-1].values

                # Get selected model
                model = None
                if self.radioButton.isChecked():
                    model = DecisionTreeRegressor
                elif self.radioButton_2.isChecked():
                    model = RandomForestRegressor
                elif self.radioButton_3.isChecked():
                    model = CatBoostRegressor
                elif self.radioButton_4.isChecked():
                    model = ExtraTreesRegressor
                elif self.radioButton_5.isChecked():
                    model = AdaBoostRegressor
                elif self.radioButton_6.isChecked():
                    model = XGBRegressor
                elif self.radioButton_7.isChecked():
                    model = GradientBoostingRegressor
                elif self.radioButton_8.isChecked():
                    model = LGBMRegressor

                if model is None:
                    self.showErrorMessage("Please select a model.")
                    return

                # Get parameters from UI and validate them
                NumberOfRows_str = self.lineEdit_2.text()
                NumberOfQueens_str = self.lineEdit_2.text()
                epochs_str = self.lineEdit_3.text()

                # Debugging statements
                print("Raw Initial Population input:", repr(NumberOfRows_str))
                print("Raw Number of genes input:", repr(NumberOfQueens_str))
                print("Raw Number of epochs input:", repr(epochs_str))

                if not NumberOfRows_str or not NumberOfQueens_str or not epochs_str:
                    self.showErrorMessage("Please ensure all input fields are filled with valid numbers.")
                    return

                try:
                    if NumberOfRows_str:
                        NumberOfRows = int(NumberOfRows_str)
                    else:
                        NumberOfRows = 20
                    NumberOfRows = int(NumberOfRows_str)
                    NumberOfQueens = int(NumberOfQueens_str)
                    epochs = int(epochs_str)
                    mr = self.horizontalSlider.value() / 100.0
                except ValueError as ve:
                    self.showErrorMessage(f"Please ensure all input fields are filled with valid numbers. Error: {ve}")
                    return

                # Genetic Algorithm for Feature Selection
                def randomGeneration(NumberOfRows, NumberOfQueens, m):
                    generation_list = []
                    # Ensure at least some features are selected
                    for i in range(NumberOfRows - 1):  # Generate one less to add the all-1s member later
                        gene = [1] * NumberOfQueens
                        zero_indices = random.sample(range(NumberOfQueens), NumberOfQueens // 2)  # Randomly select half indices to set to 0
                        for j in zero_indices:
                            gene[j] = 0
                        generation_list.append(gene)

                    # Add one member with all features selected
                    all_features_selected = [1] * NumberOfQueens
                    generation_list.append(all_features_selected)

                    return generation_list


                def cross_over(generation_list, p, n):
                    for i in range(0, p, 2):
                        child1 = generation_list[i][:n // 2] + generation_list[i + 1][n // 2:n]
                        child2 = generation_list[i + 1][:n // 2] + generation_list[i][n // 2:n]
                        generation_list.append(child1)
                        generation_list.append(child2)
                    return generation_list

                def mutation(generation_list, p, n, mr):
                    chosen_ones = list(range(p, p * 2))
                    shuffle(chosen_ones)
                    chosen_ones = chosen_ones[:int(p * mr)]

                    for i in chosen_ones:
                        cell = rnd(0, n - 1)
                        val = rnd(0, 1)
                        generation_list[i][cell] = val
                    return generation_list

                def fitness(population_list):
                    fitness = []
                    for i in population_list:
                        member = []
                        selected_features = [df.columns[j] for j in range(len(i)) if i[j] == 1]  # Select features where the value is 1
                        if not selected_features:
                            selected_features = df.columns.tolist()  # If no features selected, use all features to avoid errors
                        X = df[selected_features].to_numpy()
                        Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=10)
                        model_instance = model(random_state=0)
                        model_instance.fit(Xtr, ytr)
                        yprte = model_instance.predict(Xte)
                        r2te = round(r2_score(yte, yprte), 2)
                        member.extend(i)
                        member.append(r2te)
                        fitness.append(member)
                    return fitness

                def hazf(result):
                    for i in result:
                        i.pop()
                    return result

                # Initial generation
                generation = randomGeneration(NumberOfRows, NumberOfQueens, len(df.columns) - 1)
                for epoch in range(epochs):
                    generation = cross_over(generation, NumberOfRows, NumberOfQueens)
                    generation = mutation(generation, NumberOfRows, NumberOfQueens, mr)
                    fit = fitness(generation)
                    fit = sorted(fit, key=lambda x: x[-1], reverse=True)
                    generation = fit[:NumberOfRows]
                    print(f"Epoch {epoch + 1}/{epochs} - Best R2: {generation[0][-1]}")  # Print best R2 score in each epoch

                self.best_features = [df.columns[j] for j in range(len(generation[0]) - 1) if generation[0][j] == 1]
                print("Best features:", self.best_features)  # Print best features

                # Display the best features in the textEdit
                self.textEdit.setPlainText(", ".join(self.best_features))

                # Create input fields for the best features
                self.createFeatureInputs()

            except Exception as e:
                self.showErrorMessage(str(e))
        else:
            self.showErrorMessage("No dataset loaded. Please import a dataset first.")

    def createFeatureInputs(self):
        if hasattr(self, 'best_features'):
            # Remove previous input fields if any
            for i in reversed(range(self.groupBox.layout().count())): 
                widget = self.groupBox.layout().itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()

            # Add new input fields for best features
            for feature in self.best_features:
                label = QtWidgets.QLabel(feature, self.groupBox)
                label.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
                line_edit = QtWidgets.QLineEdit(self.groupBox)
                line_edit.setObjectName(f"input_{feature}")

                hbox = QtWidgets.QHBoxLayout()
                hbox.addWidget(label)
                hbox.addWidget(line_edit)
                self.groupBox.layout().addLayout(hbox)

            # Add Predict button and result display
            self.predictButton = QtWidgets.QPushButton("Predict", self.groupBox)
            self.predictButton.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
            self.predictButton.clicked.connect(self.predict)

            self.resultLineEdit = QtWidgets.QLineEdit(self.groupBox)
            self.resultLineEdit.setReadOnly(True)

            hbox = QtWidgets.QHBoxLayout()
            hbox.addWidget(self.predictButton)
            hbox.addWidget(self.resultLineEdit)
            self.groupBox.layout().addLayout(hbox)



    def predict(self):
        if hasattr(self, 'best_features'):
            try:
                input_data = []
                for feature in self.best_features:
                    line_edit = self.groupBox.findChild(QtWidgets.QLineEdit, f"input_{feature}")
                    if line_edit:
                        value = float(line_edit.text())
                        input_data.append(value)
                    else:
                        self.showErrorMessage(f"Input for feature '{feature}' not found.")
                        return

                input_data = pd.DataFrame([input_data], columns=self.best_features)

                # Get selected model
                model = None
                if self.radioButton.isChecked():
                    model = DecisionTreeRegressor
                elif self.radioButton_2.isChecked():
                    model = RandomForestRegressor
                elif self.radioButton_3.isChecked():
                    model = CatBoostRegressor
                elif self.radioButton_4.isChecked():
                    model = ExtraTreesRegressor
                elif self.radioButton_5.isChecked():
                    model = AdaBoostRegressor
                elif self.radioButton_6.isChecked():
                    model = XGBRegressor
                elif self.radioButton_7.isChecked():
                    model = GradientBoostingRegressor
                elif self.radioButton_8.isChecked():
                    model = LGBMRegressor

                if model is None:
                    self.showErrorMessage("Please select a model.")
                    return

                # Train model with best features
                y = self.df.iloc[:, -1].values
                X = self.df[self.best_features].values
                Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=10)
                model_instance = model(random_state=0)
                model_instance.fit(Xtr, ytr)

                # Predict
                prediction = model_instance.predict(input_data)
                self.resultLineEdit.setText(str(prediction[0]))

            except Exception as e:
                self.showErrorMessage(str(e))

    def showShapleyValues(self):
        if hasattr(self, 'df') and hasattr(self, 'best_features'):
            try:
                df = self.df.copy()
                best_features = self.best_features
                y = df.iloc[:, -1].values
                X = df[best_features].values

                # Train-test split
                Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=10)

                # Get selected model
                model = None
                if self.radioButton.isChecked():
                    model = DecisionTreeRegressor(random_state=0)
                elif self.radioButton_2.isChecked():
                    model = RandomForestRegressor(random_state=0)
                elif self.radioButton_3.isChecked():
                    model = CatBoostRegressor(random_state=0)
                elif self.radioButton_4.isChecked():
                    model = ExtraTreesRegressor(random_state=0)
                elif self.radioButton_5.isChecked():
                    model = AdaBoostRegressor(random_state=0)
                elif self.radioButton_6.isChecked():
                    model = XGBRegressor(random_state=0)
                elif self.radioButton_7.isChecked():
                    model = GradientBoostingRegressor(random_state=0)
                elif self.radioButton_8.isChecked():
                    model = LGBMRegressor(random_state=0)

                if model is None:
                    self.showErrorMessage("Please select a model.")
                    return

                # Train the model
                model.fit(Xtr, ytr)

                # Calculate SHAP values
                explainer = shap.Explainer(model, Xtr, feature_names=best_features)
                shap_values = explainer(Xte)

                fig, ax = plt.subplots(figsize=(12, 8))
                shap.summary_plot(shap_values, Xte, color='coolwarm', show=False)
                ax.set_xlabel('Shapley values: impact on model output', fontsize=12, fontname='Times New Roman', fontweight='bold')
                ax.set_ylabel('Features', fontsize=14, fontname='Times New Roman', fontweight='bold')

                handles, labels = ax.get_legend_handles_labels()
                legend = ax.legend(handles, labels, loc='upper right', fontsize=12)
                for text in legend.texts:
                    text.set_fontname('Times New Roman')

                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontname('Times New Roman')
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontname('Times New Roman')

                # Convert plot to image
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                img = img.resize((800, 600), Image.LANCZOS)
                imgQt = ImageQt.ImageQt(img)

                # Display image in message box
                msgBox = QMessageBox()
                msgBox.setIconPixmap(QtGui.QPixmap.fromImage(imgQt))
                msgBox.setWindowTitle("Shapley Values")
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.exec_()

                plt.close()

            except Exception as e:
                self.showErrorMessage(str(e))

    def showResults(self):
        if not self.radioButton_9.isChecked():  # Check if radioButton_9 is off
            if hasattr(self, 'df') and hasattr(self, 'best_features'):
                try:
                    df = self.df.copy()
                    best_features = self.best_features
                    y = df.iloc[:, -1].values
                    X = df[best_features].values

                    # Train-test split
                    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=10)

                    # Get selected model
                    model = None
                    if self.radioButton.isChecked():
                        model = DecisionTreeRegressor(random_state=0)
                    elif self.radioButton_2.isChecked():
                        model = RandomForestRegressor(random_state=0)
                    elif self.radioButton_3.isChecked():
                        model = CatBoostRegressor(random_state=0)
                    elif self.radioButton_4.isChecked():
                        model = ExtraTreesRegressor(random_state=0)
                    elif self.radioButton_5.isChecked():
                        model = AdaBoostRegressor(random_state=0)
                    elif self.radioButton_6.isChecked():
                        model = XGBRegressor(random_state=0)
                    elif self.radioButton_7.isChecked():
                        model = GradientBoostingRegressor(random_state=0)
                    elif self.radioButton_8.isChecked():
                        model = LGBMRegressor(random_state=0)

                    if model is None:
                        self.showErrorMessage("Please select a model.")
                        return

                    # Train the model
                    model.fit(Xtr, ytr)
                    yprtr = model.predict(Xtr)
                    yprte = model.predict(Xte)
                    r2tr = round(r2_score(ytr, yprtr), 2)
                    r2te = round(r2_score(yte, yprte), 2)
                    msetr = round(mean_squared_error(ytr, yprtr)**0.5, 2)
                    msete = round(mean_squared_error(yte, yprte)**0.5, 2)
                    maetr = round(mean_absolute_error(ytr, yprtr), 2)
                    maete = round(mean_absolute_error(yte, yprte), 2)

                    # Plotting the figures
                    plt.figure(figsize=(8, 6))
                    font = {'family': 'Times New Roman', 'size': 14}
                    plt.rc('font', **font)
                    plt.scatter(ytr, yprtr, s=80, marker='o', facecolors='blue', edgecolors='black',
                                label=f'\n Train \n R2 = {r2tr}  \nRMSE = {msetr}\nMAE = {maetr}')
                    plt.scatter(yte, yprte, s=80, marker='o', facecolors='pink', edgecolors='black',
                                label=f'\n Test \n R2 = {r2te} \nRMSE = {msete}\nMAE = {maete}')
                    plt.plot([min(ytr.min(), yte.min()), max(ytr.max(), yte.max())],
                             [min(ytr.min(), yte.min()), max(ytr.max(), yte.max())], c='black', lw=1.4, label='y = x')
                    plt.title(f'Results', fontsize=14)
                    plt.xlabel('Vu (kN)_Experimental', fontsize=15)
                    plt.ylabel('Vu (kN)_Predicted', fontsize=15)
                    plt.legend(loc=4)

                    # Convert plot to image
                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    img = Image.open(buf)
                    img = img.resize((800, 600), Image.LANCZOS)
                    imgQt = ImageQt.ImageQt(img)

                    # Display image in message box
                    msgBox = QMessageBox()
                    msgBox.setIconPixmap(QtGui.QPixmap.fromImage(imgQt))
                    msgBox.setWindowTitle("Model Results")
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    msgBox.exec_()

                except Exception as e:
                    self.showErrorMessage(str(e))
            else:
                self.showErrorMessage("No dataset or best features found. Please import a dataset and find best features first.")
        else:
             if hasattr(self, 'df') and hasattr(self, 'best_features'):
                try:
                    df = self.df.copy()
                    best_features = self.best_features
                    y = df.iloc[:, -1].values
                    X = df[best_features].values

                    # Train-test split
                    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=10)

                    # Get selected model
                    model = None
                    if self.radioButton.isChecked():
                        max_depth = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, None]
                        min_samples_leaf = [1, 2, 4, 8]
                        min_samples_split = [2, 5, 10, 20]
                        max_features = [ 'sqrt', 'log2', None]
                        result = []
                        for i in max_depth:
                            for j in min_samples_leaf:
                                for k in min_samples_split:
                                    for m in max_features:
                                        model = DecisionTreeRegressor(random_state=0, max_depth=i, min_samples_leaf=j, min_samples_split=k, max_features=m)
                                        model.fit(Xtr, ytr)
                                        yprtr = model.predict(Xtr)
                                        yprte = model.predict(Xte)
                                        r2tr = round(r2_score(ytr, yprtr), 2)
                                        r2te = round(r2_score(yte, yprte), 2)
                                        result.append((i, j, k, m, r2tr, r2te))
                        best_params = sorted(result, key=lambda x: x[-1], reverse=True)[0]
                        model = DecisionTreeRegressor(random_state=0, max_depth=best_params[0], min_samples_leaf=best_params[1], min_samples_split=best_params[2], max_features=best_params[3])

                    elif self.radioButton_2.isChecked():
                        n_estimators = [100, 200,300,400,500]
                        max_depth = [2, 3, 4, 8, 10,15,20, None]
                        max_features = ['sqrt', 'log2', None]
                        result = []
                        for i in n_estimators:
                            for j in max_depth:
                                for m in max_features:
                                    model = RandomForestRegressor(random_state=0, n_estimators=i, max_depth=j, max_features=m)
                                    model.fit(Xtr, ytr)
                                    yprtr = model.predict(Xtr)
                                    yprte = model.predict(Xte)
                                    r2tr = round(r2_score(ytr, yprtr), 2)
                                    r2te = round(r2_score(yte, yprte), 2)
                                    result.append((i, j, m, r2tr, r2te))

                        best_params = sorted(result, key=lambda x: x[-1], reverse=True)[0]
                        model = RandomForestRegressor(random_state=0, n_estimators=best_params[0],
                                                      max_depth=best_params[1], max_features=best_params[2])
                    elif self.radioButton_3.isChecked():
                        learning_rate=[0.01, 0.05,0.1,0.2,0.3,0.4]
                        iteration= [30,50, 100, 200, 500]
                        depth= [2,3, 5, 7, 9, 12]
                        result=[]
                        for i in learning_rate:
                            for j in iteration:
                                for k in depth:
                                            model=XGBRegressor(learning_rate=i, iteration=j,
                                                                            depth=k,random_state=0)
                                            model.fit(Xtr , ytr)
                                            yprtr = model.predict(Xtr)
                                            yprte = model.predict(Xte)
                                            r2tr=round(r2_score(ytr , yprtr),2)
                                            r2te=round(r2_score(yte , yprte),2)
                                            result.append((i,j,k,r2tr,r2te))
                        best_params = sorted(result, key=lambda x: x[-1], reverse=True)[0]
                        model = XGBRegressor(learning_rate=best_params[0], iteration=best_params[1],
                                            depth=best_params[2],random_state=0)
                        model = CatBoostRegressor(random_state=0)
                    elif self.radioButton_4.isChecked():
                        n_estimators = [50, 100, 150, 200, 250, 300]
                        max_depth = [None, 5, 10, 15, 20, 25]
                        min_samples_split = [2, 5, 10, 20]
                        min_samples_leaf = [1, 2, 4, 8]
                        max_features = ['sqrt', 'log2', None]
                        result = []
                        for i in n_estimators:
                            for j in max_depth:
                                for k in min_samples_split:
                                    for m in min_samples_leaf:
                                        for n in max_features:
                                            model = ExtraTreesRegressor(n_estimators=i, max_depth=j, min_samples_split=k, min_samples_leaf=m, max_features=n, random_state=0)
                                            model.fit(Xtr, ytr)
                                            yprtr = model.predict(Xtr)
                                            yprte = model.predict(Xte)
                                            r2tr = round(r2_score(ytr, yprtr), 2)
                                            r2te = round(r2_score(yte, yprte), 2)
                                            result.append((i, j, k, m, n, r2tr, r2te))
                        best_params = sorted(result, key=lambda x: x[-1], reverse=True)[0]
                        model = ExtraTreesRegressor(n_estimators=best_params[0], max_depth=best_params[1], min_samples_split=best_params[2], min_samples_leaf=best_params[3], max_features=best_params[4], random_state=0)

                    elif self.radioButton_5.isChecked():
                        learning_rate = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
                        n_estimators = [30, 50, 100, 200, 300]
                        loss = ['linear', 'square', 'exponential']  # corrected loss parameter values
                        result = []
                        for i in learning_rate:
                            for j in n_estimators:
                                for k in loss:
                                    model = AdaBoostRegressor(learning_rate=i, n_estimators=j, loss=k, random_state=0)
                                    model.fit(Xtr, ytr)
                                    yprtr = model.predict(Xtr)
                                    yprte = model.predict(Xte)
                                    r2tr = round(r2_score(ytr, yprtr), 2)
                                    r2te = round(r2_score(yte, yprte), 2)
                                    result.append((i, j, k, r2tr, r2te))
                        best_params = sorted(result, key=lambda x: x[-1], reverse=True)[0]
                        model = AdaBoostRegressor(learning_rate=best_params[0], n_estimators=best_params[1], loss=best_params[2], random_state=0)

                    elif self.radioButton_6.isChecked():
                        learning_rate=[0.01, 0.05,0.1,0.2,0.3,0.4]
                        n_estimators= [30,50, 100, 200, 300]
                        max_depth= [3, 5, 7, 9, 12]
                        result=[]
                        for i in learning_rate:
                            for j in n_estimators:
                                for k in max_depth:
                                            model=XGBRegressor(learning_rate=i, n_estimators=j,
                                                                            max_depth=k,random_state=0)
                                            model.fit(Xtr , ytr)
                                            yprtr = model.predict(Xtr)
                                            yprte = model.predict(Xte)
                                            r2tr=round(r2_score(ytr , yprtr),2)
                                            r2te=round(r2_score(yte , yprte),2)
                                            result.append((i,j,k,r2tr,r2te))
                        best_params = sorted(result, key=lambda x: x[-1], reverse=True)[0]
                        model = XGBRegressor(learning_rate=best_params[0], n_estimators=best_params[1],
                                            max_depth=best_params[2],random_state=0)
                    elif self.radioButton_7.isChecked():
                        learning_rate=[0.01, 0.05,0.1,0.2,0.3,0.4]
                        n_estimators= [30,50, 100, 200, 300]
                        max_depth= [3, 5, 7, 9, 12]
                        result=[]
                        for i in learning_rate:
                            for j in n_estimators:
                                for k in max_depth:
                                            model=GradientBoostingRegressor(learning_rate=i, n_estimators=j,
                                                                            max_depth=k,random_state=0)
                                            model.fit(Xtr , ytr)
                                            yprtr = model.predict(Xtr)
                                            yprte = model.predict(Xte)
                                            r2tr=round(r2_score(ytr , yprtr),2)
                                            r2te=round(r2_score(yte , yprte),2)
                                            result.append((i,j,k,r2tr,r2te))
                        best_params = sorted(result, key=lambda x: x[-1], reverse=True)[0]
                        model = GradientBoostingRegressor(learning_rate=best_params[0], n_estimators=best_params[1],
                                            max_depth=best_params[2],random_state=0)
                    elif self.radioButton_8.isChecked():
                        model = LGBMRegressor(random_state=0)

                    if model is None:
                        self.showErrorMessage("Please select a model.")
                        return

                    # Train the model
                    model.fit(Xtr, ytr)
                    yprtr = model.predict(Xtr)
                    yprte = model.predict(Xte)
                    r2tr = round(r2_score(ytr, yprtr), 2)
                    r2te = round(r2_score(yte, yprte), 2)
                    msetr = round(mean_squared_error(ytr, yprtr)**0.5, 2)
                    msete = round(mean_squared_error(yte, yprte)**0.5, 2)
                    maetr = round(mean_absolute_error(ytr, yprtr), 2)
                    maete = round(mean_absolute_error(yte, yprte), 2)

                    # Plotting the figures
                    plt.figure(figsize=(8, 6))
                    font = {'family': 'Times New Roman', 'size': 14}
                    plt.rc('font', **font)
                    plt.scatter(ytr, yprtr, s=80, marker='o', facecolors='blue', edgecolors='black',
                                label=f'\n Train \n R2 = {r2tr}  \nRMSE = {msetr}\nMAE = {maetr}')
                    plt.scatter(yte, yprte, s=80, marker='o', facecolors='pink', edgecolors='black',
                                label=f'\n Test \n R2 = {r2te} \nRMSE = {msete}\nMAE = {maete}')
                    plt.plot([min(ytr.min(), yte.min()), max(ytr.max(), yte.max())],
                             [min(ytr.min(), yte.min()), max(ytr.max(), yte.max())], c='black', lw=1.4, label='y = x')
                    plt.title(f'Results', fontsize=14)
                    plt.xlabel('Vu (kN)_Experimental', fontsize=15)
                    plt.ylabel('Vu (kN)_Predicted', fontsize=15)
                    plt.legend(loc=4)

                    # Convert plot to image
                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    img = Image.open(buf)
                    img = img.resize((800, 600), Image.LANCZOS)
                    imgQt = ImageQt.ImageQt(img)

                    # Display image in message box
                    msgBox = QMessageBox()
                    msgBox.setIconPixmap(QtGui.QPixmap.fromImage(imgQt))
                    msgBox.setWindowTitle("Model Results")
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    msgBox.exec_()

                except Exception as e:
                    self.showErrorMessage(str(e))

    def resetAll(self):
        try:
            # Clear the Model Input and Prediction box
            for widget in self.groupBox.findChildren(QtWidgets.QWidget):
                widget.deleteLater()

            # Reset the layout of groupBox
            self.groupBox.setLayout(QVBoxLayout())

            # Reset all other variables and UI elements
            self.df = None
            self.best_features = None

            # Clear the table
            self.tableWidget.clear()
            self.tableWidget.setRowCount(0)
            self.tableWidget.setColumnCount(0)

            # Clear text edits
            self.textEdit.clear()
            self.resultLineEdit.clear()

            # Reset input fields
            self.lineEdit.clear()
            self.lineEdit_2.clear()
            self.lineEdit_3.clear()
            self.label_slider_value.setText("0.00")
            self.horizontalSlider.setValue(0)

            # Deselect radio buttons
            self.buttonGroup.setExclusive(False)
            for button in self.buttonGroup.buttons():
                button.setChecked(False)
            self.buttonGroup.setExclusive(True)

            print("All settings reset successfully")  # Debug statement

        except Exception as e:
            self.showErrorMessage(str(e))




    def retranslateTab2(self, tab):
        _translate = QtCore.QCoreApplication.translate
        tab.setWindowTitle(_translate("tab", "Train"))
        self.radioButton.setText(_translate("tab", "Decision Tree regressor"))
        self.label.setText(_translate("tab", "Model selection:"))
        self.label_2.setText(_translate("tab", "Genetic operators for Feature Selection:"))
        self.label_3.setText(_translate("tab", "Initial Population:"))
        self.radioButton_2.setText(_translate("tab", "Random Forest regressor"))
        self.radioButton_4.setText(_translate("tab", "Extra tree regressor"))
        self.radioButton_5.setText(_translate("tab", "ADAboost regressor"))
        self.radioButton_6.setText(_translate("tab", "XGBoost regressor"))
        self.radioButton_3.setText(_translate("tab", "Catboost regressor"))
        self.radioButton_7.setText(_translate("tab", "Gradientboost regressor"))
        self.radioButton_8.setText(_translate("tab", "LGBM regressor"))
        self.label_4.setText(_translate("tab", "Number of genes:"))
        self.label_5.setText(_translate("tab", "Mutation Rate:"))
        self.label_6.setText(_translate("tab", "Number of epochs:"))
        self.label_12.setText(_translate("tab", "Results:"))
        self.pushButton_2.setText(_translate("tab", "Find the best features"))
        self.label_14.setText(_translate("tab", "The selected features:"))
        self.radioButton_9.setText(_translate("tab", "Tune Hyperparameters"))
        self.pushButton_3.setText(_translate("tab", "Show results"))
        self.pushButton_4.setText(_translate("tab", "Show Shapley Values"))
        self.pushButton_reset.setText(_translate("tab", "Reset"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


# In[ ]:




