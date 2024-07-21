import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showgui.ui', self)
        self.Image = None
        self.button_loadCitra.clicked.connect(self.fungsi)
        self.button_prosesCitra.clicked.connect(self.grayscale)

        #operasi titik
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Stretching.triggered.connect(self.contrastStretching)
        self.actionNegative_Image.triggered.connect(self.negativeImage)
        self.actionBiner_Image.triggered.connect(self.binaryImage)

        #operasi histogram
        self.actionHistogram_Gray.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.rgbHistogram)
        self.actionHistogram_Equal.triggered.connect(self.equalHistogram)

        #operasi geometri
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action90_Derajat.triggered.connect(self.rotasi90derajat)
        self.action45_Derajat.triggered.connect(self.rotasi45derajat)
        self.action180_Derajat.triggered.connect(self.rotasi180derajat)
        self.action_45_Derajat.triggered.connect(self.rotasimin45derajat)
        self.action_90_Derajat.triggered.connect(self.rotasimin90derajat)
        self.actionZoom_In.triggered.connect(self.zoomIn)
        self.actionZoom_Out.triggered.connect(self.zoomOut)
        self.actionDimensi.triggered.connect(self.dimensi)
        self.actionCrop.triggered.connect(self.cropImage)

        #operasi aritmatika
        self.actionAdd.triggered.connect(self.aritmatikaAdd)
        self.actionSubtract.triggered.connect(self.aritmatikaSubtract)
        self.actionAnd.triggered.connect(self.booleanAnd)
        self.actionOr.triggered.connect(self.booleanOr)
        self.actionNot.triggered.connect(self.booleanNot)
        self.actionXor.triggered.connect(self.booleanXor)

        #operasi spasial
        self.actionKonvolusi_2D.triggered.connect(self.konvolusi2D)
        self.actionMean_Filter.triggered.connect(self.meanFilter)
        self.actionGauss_Filter.triggered.connect(self.gaussFilter)
        self.actioni.triggered.connect(self.sharpeningI)
        self.actionii.triggered.connect(self.sharpeningII)
        self.actioniii.triggered.connect(self.sharpeningIII)
        self.actioniv.triggered.connect(self.sharpeningIV)
        self.actionv.triggered.connect(self.sharpeningV)
        self.actionvi.triggered.connect(self.sharpeningVI)
        self.actionMedian_Filter.triggered.connect(self.medianFilter)
        self.actionMax_Filter.triggered.connect(self.maxFilter)
        self.actionMin_Filter.triggered.connect(self.minFilter)

        #Operasi Deteksi Tepi
        self.actionSobel.triggered.connect(self.deteksiTepi)
        self.actionCanny.triggered.connect(self.deteksiCanny)
        self.actionDilasi.triggered.connect(self.dilasi)
        self.actionErosi.triggered.connect(self.erosi)
        self.actionOpening.triggered.connect(self.opening)
        self.actionClosing.triggered.connect(self.closing)

        #Segmentasi Citra
        self.actionBinary.triggered.connect(self.binary)
        self.actionBinary_Invers.triggered.connect(self.binaryInvers)
        self.actionTrunc.triggered.connect(self.trunc)
        self.actionTo_Zero.triggered.connect(self.zero)
        self.actionTo_Zero_Invers.triggered.connect(self.zeroInvers)
        self.actionMean_Thresholding.triggered.connect(self.meanThresholding)
        self.actionGaussian_Thresholding.triggered.connect(self.gaussThresholding)
        self.actionOtsu_Thresholding.triggered.connect(self.otsuThresholding)
        self.actionContour_Image.triggered.connect(self.contourImage)

        #Color Processing
        self.actionColor_Tracking.triggered.connect(self.colorTracking)
        self.actionColor_Picker.triggered.connect(self.colorPicker)
        self.actionObject_Detection.triggered.connect(self.objectDetection)
    def deteksiCanny(self):
        # Input image
        img_input = cv2.imread('BANG.jpeg')

        if img_input is None:
            print("Image not found")
            return

        # Image convert RGB to grayscale
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

        gauss = (1.0 / 57) * np.array(
            [[0, 1, 2, 1, 0],
             [1, 3, 5, 3, 1],
             [2, 5, 9, 5, 2],
             [1, 3, 5, 3, 1],
             [0, 1, 2, 1, 0]]
        )

        # Convolution of image with Gaussian kernel
        smoothed_image = cv2.filter2D(gray_image, -1, gauss)

        # Initialize Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        Gx = cv2.filter2D(smoothed_image, cv2.CV_64F, sobel_x)
        Gy = cv2.filter2D(smoothed_image, cv2.CV_64F, sobel_y)

        gradient = np.sqrt(Gx ** 2 + Gy ** 2)
        gradient = (gradient / np.max(gradient)) * 255
        gradient = gradient.astype(np.uint8)
        theta = np.arctan2(Gy, Gx)

        # Non-Maximum Suppression
        H, W = gradient.shape
        Z = np.zeros((H, W), dtype=np.uint8)
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255

                    # Angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = gradient[i, j + 1]
                        r = gradient[i, j - 1]
                    # Angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = gradient[i + 1, j - 1]
                        r = gradient[i - 1, j + 1]
                    # Angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = gradient[i + 1, j]
                        r = gradient[i - 1, j]
                    # Angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = gradient[i - 1, j - 1]
                        r = gradient[i + 1, j + 1]

                    if (gradient[i, j] >= q) and (gradient[i, j] >= r):
                        Z[i, j] = gradient[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError as e:
                    pass

        img_N = Z.astype("uint8")
        weak = 100
        strong = 150
        for i in range(H):
            for j in range(W):
                a = img_N.item(i, j)
                if a > strong:
                    b = 255
                elif a < weak:
                    b = 0
                else:
                    b = a
                img_N.itemset((i, j), b)

        img_H1 = img_N.astype("uint8")
        cv2.imshow("Hysteresis I", img_H1)

        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if img_H1[i, j] == weak:
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or (img_H1[i + 1, j] == strong) or
                                (img_H1[i + 1, j + 1] == strong) or (img_H1[i, j - 1] == strong) or
                                (img_H1[i, j + 1] == strong) or (img_H1[i - 1, j - 1] == strong) or
                                (img_H1[i - 1, j] == strong) or (img_H1[i - 1, j + 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass

        img_H2 = img_H1.astype("uint8")
        cv2.imshow("Hysteresis II", img_H2)
        cv2.imshow('Image asli', img_input)
        cv2.waitKey()

    def dilasi(self):
        img_input = cv2.imread('BANG.jpeg', cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(img_input, 127, 255, cv2.THRESH_BINARY)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        dilasi = cv2.dilate(binary_image, strel, iterations=1)
        cv2.imshow('Image asli', img_input)
        cv2.imshow('Dilated Image', dilasi)
        cv2.waitKey()

    def erosi(self):
        img_input = cv2.imread('BANG.jpeg', cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(img_input, 127, 255, cv2.THRESH_BINARY)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        erosi = cv2.erode(binary_image, strel, iterations=1)
        cv2.imshow('Image asli', img_input)
        cv2.imshow('Eroded Image', erosi)
        cv2.waitKey()

    def opening(self):
        img_input = cv2.imread('BANG.jpeg', cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(img_input, 127, 255, cv2.THRESH_BINARY)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, strel)
        cv2.imshow('Image asli', img_input)
        cv2.imshow('Opened Image', opening)
        cv2.waitKey()

    def closing(self):
        img_input = cv2.imread('BANG.jpeg', cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(img_input, 127, 255, cv2.THRESH_BINARY)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, strel)
        cv2.imshow('Image asli', img_input)
        cv2.imshow('Closed Image', closing)
        cv2.waitKey()

    def binary(self):
        img_input = cv2.imread('BANG.jpeg', cv2.IMREAD_GRAYSCALE)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img_input, T, max, cv2.THRESH_BINARY)
        cv2.imshow("Binary", thresh)
        print(thresh)
        cv2.waitKey()

    def binaryInvers(self):
        img_input = cv2.imread('BANG.jpeg', cv2.IMREAD_GRAYSCALE)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img_input, T, max, cv2.THRESH_BINARY_INV)
        cv2.imshow("Binary Invers", thresh)
        print(thresh)
        cv2.waitKey()

    def trunc(self):
        img_input = cv2.imread('BANG.jpeg', cv2.IMREAD_GRAYSCALE)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img_input, T, max, cv2.THRESH_TRUNC)
        cv2.imshow("Trunc Image", thresh)
        print(thresh)
        cv2.waitKey()

    def zero(self):
        img_input = cv2.imread('BANG.jpeg', cv2.IMREAD_GRAYSCALE)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img_input, T, max, cv2.THRESH_TOZERO)
        cv2.imshow("To Zero", thresh)
        print(thresh)
        cv2.waitKey()

    def zeroInvers(self):
        img_input = cv2.imread('BANG.jpeg', cv2.IMREAD_GRAYSCALE)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img_input, T, max, cv2.THRESH_TOZERO_INV)
        cv2.imshow("To Zero Invers", thresh)
        print(thresh)
        cv2.waitKey()

    def meanThresholding(self):
        img_input = cv2.imread('BANG.jpeg', cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(img_input, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow("Mean Thresholding", thresh)
        print(thresh)
        cv2.waitKey()

    def gaussThresholding(self):
        img_input = cv2.imread('BANG.jpeg', cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(img_input, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow("Gaussian Thresholding", thresh)
        print(thresh)
        cv2.waitKey()

    def otsuThresholding(self):
        img_input = cv2.imread('BANG.jpeg', cv2.IMREAD_GRAYSCALE)
        T = 130
        ret, thresh = cv2.threshold(img_input, T, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("Otsu Thresholding", thresh)
        print(thresh)
        cv2.waitKey()
    def contourImage(self):
        # Membaca gambar input
        img_input = cv2.imread('BANG.jpeg')
        if img_input is None:
            print("Error: Gambar tidak ditemukan.")
            return

        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        T = 127
        ret, thresh = cv2.threshold(gray_image, T, 255, cv2.THRESH_BINARY)

        # Menampilkan hasil thresholding untuk debugging
        cv2.imshow('Thresholded Image', thresh)

        # Menemukan kontur
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Menampilkan gambar dengan semua kontur yang ditemukan
        img_contours = img_input.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        cv2.imshow('All Contours', img_contours)

        # Warna untuk masing-masing bentuk
        shape_colors = {
            "Triangle": (0, 255, 0),
            "Square": (255, 0, 0),
            "Rectangle": (0, 0, 255),
            "Star": (255, 255, 0),
            "Circle": (0, 255, 255)
        }

        # Membuat gambar berwarna untuk mewarnai bagian dalam
        colored_image = np.zeros_like(img_input)

        for contour in contours:
            # Mengaproksimasi kontur menjadi poligon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Mengidentifikasi bentuk berdasarkan jumlah sisi
            num_vertices = len(approx)
            shape = "Unknown"

            if num_vertices == 3:
                shape = "Triangle"
            elif num_vertices == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                # Debug print removed
                if 0.9 <= aspect_ratio <= 1.1:
                    shape = "Square"
                else:
                    shape = "Rectangle"
            elif num_vertices == 10:  # Typically, stars have more vertices
                shape = "Star"
            else:
                # Menggunakan kebundaran untuk membedakan lingkaran dari bintang
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter != 0:  # Menghindari pembagian dengan nol
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    print(f"Circularity: {circularity} for contour with {num_vertices} vertices")
                    if 0.5 <= circularity <= 1.5:  # Perluasan rentang circularity
                        shape = "Circle"
                    else:
                        shape = "Unknown"

            print(f"Detected shape: {shape} with {num_vertices} vertices")

            if shape != "Unknown":
                # Menemukan pusat kontur
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                # Mendapatkan warna untuk bentuk
                color = shape_colors.get(shape, (255, 255, 255))

                # Menggambar kontur dan nama bentuk
                cv2.drawContours(colored_image, [contour], -1, color, -1)  # Mengisi bagian dalam bentuk
                cv2.putText(colored_image, shape, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Menampilkan kontur dan bentuk yang terdeteksi
            cv2.imshow('Contours and Shapes', colored_image)
            cv2.waitKey()  # Penundaan untuk melihat hasil setiap langkah

        # Menampilkan hasil akhir
        cv2.imshow('Contours and Shapes', colored_image)
        cv2.waitKey(0)

    def colorTracking(self):
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Error: Could not open camera.")
            return

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_color = np.array([66, 98, 100])
            upper_color = np.array([156, 232, 255])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            cv2.imshow("result", result)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc key to stop
                break

        cam.release()


    def colorPicker(self):
        def nothing(x):
            pass
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("L-H", "Trackbars", 0, 179, nothing)
        cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("U-H", "Trackbars", 179, 179, nothing)
        cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
        cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

        while True:
            _, frame = cam.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            l_h = cv2.getTrackbarPos("L-H", "Trackbars")
            l_s = cv2.getTrackbarPos("L-S", "Trackbars")
            l_v = cv2.getTrackbarPos("L-V", "Trackbars")
            u_h = cv2.getTrackbarPos("U-H", "Trackbars")
            u_s = cv2.getTrackbarPos("U-S", "Trackbars")
            u_v = cv2.getTrackbarPos("U-V", "Trackbars")

            lower_color = np.array([l_h, l_s, l_v])
            upper_color = np.array([u_h, u_s, u_v])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            cv2.imshow("result", result)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc key to stop
                break

        cam.release()

    def objectDetection(self):
        cam = cv2.VideoCapture('cars.mp4')
        car_cascade = cv2.CascadeClassifier('cars.xml')

        while True:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect cars
            cars = car_cascade.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('video', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cam.release()

    def displayImage(self, window=1):
        if self.Image is not None:
            qformat = QImage.Format_Indexed8

            if len(self.Image.shape) == 3:
                if self.Image.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888

            img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)
            img = img.rgbSwapped()

            if window == 1:
                self.label.setPixmap(QPixmap.fromImage(img))
            elif window == 2:
                self.label_2.setPixmap(QPixmap.fromImage(img))


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('PRAKTEK PCD')
window.show()
sys.exit(app.exec_())
