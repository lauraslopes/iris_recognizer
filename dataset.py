import os
import numpy as np
from PIL import Image
import cv2
import math

class Dataset:
    def __init__(self, path):
        self.path = path
        self.images = []
        self.labels = []
        self.codes = []
        self.iris = []

    #funções comuns entre os datasets
    def load_images_and_labels(self):

        for d in os.listdir(self.path):
            directory = os.path.join(self.path, d)
            for sd in os.listdir(directory): #L ou R
                subdirectory = os.path.join(directory,sd)
                for f in os.listdir(subdirectory):
                    if f.endswith('.jpg'): #é uma imagem
                        image_path = os.path.join(subdirectory, f)
                        image_pil = Image.open(image_path)
                        # Convert the image format into numpy array
                        image = np.array(image_pil, 'uint8')
                        self.images.append(image)
                        # Get the label of the image (de qual pessoa pertence)
                        if subdirectory.endswith('L'):
                            self.labels.append(int(os.path.split(image_path)[1].split("L")[0].replace("S", "")))
                        else:
                            self.labels.append(int(os.path.split(image_path)[1].split("R")[0].replace("S", "")))


def preprocess(image, threshold=35, lamp=True):

    smoothed = cv2.medianBlur(image,5)
    num, mask = cv2.threshold(smoothed, threshold, 255, cv2.THRESH_BINARY_INV)

    if lamp:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=5)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.erode(mask, kernel, iterations=4)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=9)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=7)
        mask = cv2.erode(mask, kernel, iterations=4)

    mask = cv2.Canny(mask,100,200)

    circles = cv2.HoughCircles(mask,method=cv2.HOUGH_GRADIENT,dp=2,minDist=220,param1=500,param2=50,minRadius=20,maxRadius=70)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            centerx = i[0]
            centery = i[1]
            radius = i[2]

        circles = circles[0][0]

    return circles

def get_mean(image, mask):
    image_iris = cv2.bitwise_and(image, image, mask=mask)
    mean = np.mean(image_iris)
    
    return mean

def get_iris_radius(image, pupil):
    mask = np.zeros(image.shape, dtype='uint8')
    mean_ant = get_mean(image, mask)
    max_mean = -1
    max_radius = 0
    for r in range((pupil[2]+16), 120):
        mean = get_mean(image, mask)
        diff = mean - mean_ant
        mean_ant = mean
        if diff > max_mean:
            max_mean = diff
            max_radius = r

    return max_radius

def normalize_iris(image, pupil, iris_radius):

    total_grades = 2*math.pi
    num_pixels = iris_radius - pupil[2] #raio da iris menos raio pupila
    interval = np.linspace(0, total_grades, num=360)
    polar = np.zeros((num_pixels, len(interval)), dtype='uint8') #matriz para a iris normalizada
    for pixel in range(0, num_pixels):
        offset = 0
        ant = 0
        for i in interval:
            grade = i
            x = int((pupil[2] + pixel)*math.cos(grade) + pupil[0]) # centerx
            y = int((pupil[2] + pixel)*math.sin(grade) + pupil[1]) # centery
            # print pixel, int(grade*total_grades+offset), offset
            if (int(grade*total_grades+offset) == (ant+2)):
                offset-=1
            if y < image.shape[0] and x < image.shape[1]:
                polar[pixel, int(grade*total_grades+offset)] = image[y,x] #pq a imagem é deitada
            else:
                polar[pixel, int(grade*total_grades+offset)] = 0 #iris fora da imagem
            ant = int(grade*total_grades+offset)
            offset+=1
            if (grade*total_grades+offset) >= 360:
                break

    return polar