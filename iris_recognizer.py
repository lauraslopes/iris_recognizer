import cv2, os
import numpy as np
from PIL import Image
import math
import pywt
from skimage import feature
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from dataset import Dataset

#criar mascaras
def preprocess_lamp(image):

    smoothed = cv2.medianBlur(image,5)
    num, mask = cv2.threshold(smoothed, 35, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel, iterations=4)
    # cv2.imshow("Preprocesdsdsdsdsing...", mask)
    # cv2.waitKey(0)
    #aumenta um pouco o círculo
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=9)
    # cv2.imshow("Preprocesdsdsdsdsing...", mask)
    # cv2.waitKey(0)

    mask = cv2.Canny(mask,100,200)

    circles = cv2.HoughCircles(mask,method=cv2.HOUGH_GRADIENT,dp=2,minDist=220,param1=500,param2=50,minRadius=20,maxRadius=70)
    #dependendo da versão do python, descomentar a linha abaixo e comentar a anterior
    # circles = cv2.HoughCircles(mask,method=cv2.cv.CV_HOUGH_GRADIENT,dp=2,minDist=220,param1=500,param2=50,minRadius=20,maxRadius=70)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            centerx = i[0]
            centery = i[1]
            radius = i[2]
            # cv2.circle(smoothed, (centerx,centery), radius, (255, 0, 255), 1)
            # cv2.imshow("Image", smoothed)
            # cv2.waitKey(50)

        circles = circles[0][0]

    return circles

def preprocess_interval(image):

    smoothed = cv2.medianBlur(image,5)
    # cv2.imshow('smoothed', smoothed)
    num, mask = cv2.threshold(smoothed, 100, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('mask', mask)
    # cv2.waitKey()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=7)
    # cv2.imshow('dilate', mask)
    # cv2.waitKey()

    mask = cv2.erode(mask, kernel, iterations=4)
    # cv2.imshow("erode", mask)
    # cv2.waitKey(0)

    mask = cv2.Canny(mask,100,200)

    circles = cv2.HoughCircles(mask,method=cv2.HOUGH_GRADIENT,dp=2,minDist=220,param1=500,param2=50,minRadius=20,maxRadius=70)
    #dependendo da versão do python, descomentar a linha abaixo e comentar a anterior
    #circles = cv2.HoughCircles(mask,method=cv2.cv.CV_HOUGH_GRADIENT,dp=2,minDist=220,param1=500,param2=50,minRadius=20,maxRadius=70)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            centerx = i[0]
            centery = i[1]
            radius = i[2]
            # cv2.circle(smoothed, (centerx,centery), radius, (255, 0, 255), 1)
            # cv2.imshow("Image", smoothed)
            # cv2.waitKey(50)

        circles = circles[0][0]

    return circles

def get_iris (image, circle):
    centerx, centery, radius = circle

    #mascara com o raio da pupila
    pupil = np.zeros(image.shape, dtype = 'uint8')
    cv2.circle(pupil, (centerx,centery), radius, 255, 1)    # draw the outer circle
    # cv2.imshow("Preprocesdsdsdsdsing...", pupil)
    # cv2.waitKey(0)

    image_pupil = cv2.bitwise_and(image, image, mask=pupil)
    # cv2.imshow("Preprocesdsdsdsdsing...", img1)
    # cv2.waitKey(0)
    masks = np.zeros(image.shape, dtype='uint8')
    cv2.circle(masks, (centerx,centery), radius + 15, 255, 3) #aumenta raio em 2 pixels
    image_iris = cv2.bitwise_and(image, image, mask=masks)
    mean_ant = np.mean(image_iris)

    max_mean = -1
    max_radius = 0
    for r in range((radius+16), 120):
        masks = np.zeros(image.shape, dtype='uint8')
        cv2.circle(masks, (centerx,centery), r, 255, 3) #aumenta raio em 1 pixel
        image_iris = cv2.bitwise_and(image, image, mask=masks)
        mean = np.mean(image_iris)
        outraimg = image.copy()
        # cv2.circle(outraimg, (centerx,centery), r, 255, 2)
        # cv2.imshow("...oi", outraimg)
        # cv2.waitKey()
        diff = mean - mean_ant
        mean_ant = mean
        if diff > max_mean:
            # img = image.copy()
            # cv2.circle(img, (centerx,centery), r, 255, 2)
            # cv2.imshow("...", img)
            # cv2.waitKey()
            max_mean = diff
            max_radius = r

    # cv2.circle(image,(centerx,centery), max_radius, 255, 2)
    # cv2.imshow("Preprocesdsdsdsdsing...", image)
    # cv2.waitKey()
    return max_radius

def normalize_iris(image, pupil, radius_iris):

    total_grades = 2*math.pi
    num_pixels = radius_iris - pupil[2] #raio da iris menos raio pupila
    interval = np.linspace(0, total_grades, num=360)
    polar = np.zeros((num_pixels, len(interval)), dtype='uint8') #matriz para a iris normalizada
    for pixel in xrange(0, num_pixels):
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

    # cv2.imshow("Preprocesdsdsdsdsing...", polar)
    # cv2.waitKey()
    # cv2.imwrite("oioi.png", polar)
    image = polar
    return image

def hamming_distance(code_query, code_database):
    nm = code_query.shape[0] * code_query.shape[1]
    xor = np.logical_xor(code_query, code_database).sum()
    return xor / float(nm)


## CASIA IRIS LAMP ##
np.set_printoptions(threshold=np.nan)
# Path to the Lamp-100 Dataset
path_lamp = './CASIA-Iris-Lamp-100'
casia_lamp = Dataset(path_lamp)
print("Lendo dataset CASIA Lamp")
casia_lamp.load_images_and_labels()
print (str(len(casia_lamp.images))+" imagens")
c = centerx = centery = radius_iris = radius_pupil = 0
for image in casia_lamp.images:
    print("Pré-processando a imagem "+str(c+1)+" e detectando a pupila")
    pupil = preprocess_lamp(image)
    if pupil is not None: #achou a pupila
        print("Detectando a iris")
        radius_iris = get_iris(image, pupil)
        print("Normalizando a iris")
        image_normalized = normalize_iris(image, pupil, radius_iris)
        casia_lamp.iris.append(image_normalized)
        #binariza a imagem com Haar Wavelet
        print("Realizando Haar Wavelet")
        coefficients = pywt.wavedec2(image_normalized, 'haar', level=4)
        cA = coefficients[0] #Approximation
        (cH, cV, cD) = coefficients[1] #horizontal detail, vertical detail and diagonal detail
        image_binar = np.zeros((cA.shape[0], cA.shape[1]), dtype='uint8')
        for i in range(cA.shape[0]):
            for j in xrange(cA.shape[1]):
                if ((cH[i,j]>=0) and (cV[i,j]>=0) and (cD[i,j]>=0)):
                    image_binar[i,j] = 1
                else:
                    image_binar[i,j] = 0
        image_binar = np.resize(image_binar, (5,23)) #deixar todos do mesmo tamanho
        # img = image_binar.copy()
        # for i in xrange(img.shape[0]):
        #     for j in xrange(img.shape[1]):
        #         if img[i,j] == 1:
        #             img[i,j] = 255
        # cv2.imshow('binar',img)
        # cv2.waitKey()
        casia_lamp.codes.append(image_binar) #salvo na ordem das labels tbm

    else: #não achou a pupila, remove a imagem
        np.delete(casia_lamp.images, i, 0)
        c-=1
    c+=1

if (len(casia_lamp.images)%2 != 0): #se não for número par de imagens
    np.delete(casia_lamp.images, len(casia_lamp.images)-1, 0)

print("Hamming")
#percorrer como uma matriz de iris codes triangular (para n repetir as distancias)
distance = []
true_or_false = []
for i in range(0, len(casia_lamp.codes)): #3954 imagens e códigos
    code_query = casia_lamp.codes[i]
    for j in range(i+1, len(casia_lamp.codes)):
        code_database = casia_lamp.codes[j]
        distance.append(hamming_distance(code_query, code_database))
        true_or_false.append((casia_lamp.labels[i] == casia_lamp.labels[j]))

limits = np.sort(distance)
limits = np.unique(limits)
positives_total = 0
negatives_total = 0
for aux in true_or_false:
    if aux == True: #mesma label
        positives_total += 1
    else:
        negatives_total += 1

false_positives = []
false_negatives = []
print("Detectando falsos positivos e falsos negativos")
for limit in limits:
    fp = 0
    fn = 0
    for j in range(len(distance)):
        database = distance[j]
        if database <= limit and true_or_false[j] == False:
            fp += 1
        elif database > limit and true_or_false[j] == True:
            fn += 1

    fp = fp / float(negatives_total)
    fn = fn / float(positives_total)
    false_positives.append(fp)
    false_negatives.append(fn)

print("Plotando gráficos FARxFRR")
false_positives = np.array(false_positives)
false_negatives = np.array(false_negatives)
index_EER = np.argmin(np.absolute(false_positives - false_negatives)) #indice para a diferença mínima entre falso positivos e falso negativos
plt.axis([0, 1.0, 0, 1.0])
plt.title('FAR x FRR graph')
plt.xlabel('False Positives')
plt.ylabel('False Negatives')
plt.plot(false_positives, false_negatives, linestyle='--', color='black')
plt.plot(false_positives[index_EER], false_negatives[index_EER], 'o', color='green', label='EER')
plt.legend()
plt.savefig('DETcurve.png')

lbps = []
major = 0
for img in casia_lamp.iris:
    if img.shape[0] > major:
        major = img.shape[0]

c = 0
for img in casia_lamp.iris:
    img = np.resize(img, (major,img.shape[1]))
    print ("LBP da imagem "+str(c+1))
    lbps.append(feature.local_binary_pattern(img, img.shape[1], img.shape[0], method="uniform"))
    c+=1

#dividir imagens de treino e teste 10 vezes
kfold = KFold(n_splits=10, shuffle=True)
accuracy = 0
for train_index, test_index in kfold.split(lbps):
    data_train = []
    data_test = []
    label_train = []
    label_test = []
    # print("TRAIN:", train_index, "TEST:", test_index)
    #labels referenciada pelo indice da imagem
    for train in train_index:
        data_train.append(lbps[train])
        label_train.append(casia_lamp.labels[train])

    print("Treinando SVM")
    model = svm.SVC(kernel='linear', C = 1.0)
    data_train = np.reshape(data_train, (len(data_train), data_train[0].shape[0]*data_train[0].shape[1]))
    model.fit(data_train, label_train)

    for test in test_index:
        data_test.append(lbps[test])
        label_test.append(casia_lamp.labels[test])

    data_test = np.reshape(data_test, (len(data_test), data_test[0].shape[0]*data_test[0].shape[1]))
    accuracy += model.score(data_test, label_test)

print ("Mean accuracy for CASIA Lamp: "+str(accuracy/10))






### CASIA INTERVAL ###
path_interval = './CASIA-IrisV4-Interval'
casia_interval = Dataset(path_interval)
print("\n\nLendo dataset CASIA Interval")
casia_interval.load_images_and_labels()
print(str(len(casia_interval.images))+" imagens")
c = centerx = centery = radius_iris = radius_pupil = 0
for image in casia_interval.images:
    print("Pré-processando a imagem "+str(c+1)+" e detectando a pupila")
    pupil = preprocess_interval(image)
    if pupil is not None: #achou a pupila
        print("Detectando a iris")
        radius_iris = get_iris(image, pupil)
        print("Normalizando a iris")
        image_normalized = normalize_iris(image, pupil, radius_iris)
        casia_interval.iris.append(image_normalized)
        #binariza a imagem com Haar Wavelet
        print("Realizando Haar Wavelet")
        coefficients = pywt.wavedec2(image_normalized, 'haar', level=4)
        cA = coefficients[0] #Approximation
        (cH, cV, cD) = coefficients[1] #horizontal detail, vertical detail and diagonal detail
        image_binar = np.zeros((cA.shape[0], cA.shape[1]), dtype='uint8')
        for i in range(cA.shape[0]):
            for j in range(cA.shape[1]):
                if ((cH[i,j]>=0) and (cV[i,j]>=0) and (cD[i,j]>=0)):
                    image_binar[i,j] = 1
                else:
                    image_binar[i,j] = 0
        image_binar = np.resize(image_binar, (5,23)) #deixar todos do mesmo tamanho
        # img = image_binar.copy()
        # for i in xrange(img.shape[0]):
        #     for j in xrange(img.shape[1]):
        #         if img[i,j] == 1:
        #             img[i,j] = 255
        # cv2.imshow('binar',img)
        # cv2.waitKey()
        casia_interval.codes.append(image_binar) #salvo na ordem das labels tbm

    else: #não achou a pupila, remove a imagem
        np.delete(casia_interval.images, i, 0)
        c-=1
    c+=1

if (len(casia_interval.images)%2 != 0): #se não for número par de imagens
    np.delete(casia_interval.images, len(casia_interval.images)-1, 0)

print("Hamming")
#percorrer como uma matriz de iris codes triangular (para n repetir as distancias)
distance = []
true_or_false = []
for i in range(0, len(casia_interval.codes)): #3954 imagens e códigos
    code_query = casia_interval.codes[i]
    for j in range(i+1, len(casia_interval.codes)):
        code_database = casia_interval.codes[j]
        distance.append(hamming_distance(code_query, code_database))
        true_or_false.append((casia_interval.labels[i] == casia_interval.labels[j]))

limits = np.sort(distance)
limits = np.unique(limits)
positives_total = 0
negatives_total = 0
for aux in true_or_false:
    if aux == True: #mesma label
        positives_total += 1
    else:
        negatives_total += 1

false_positives = []
false_negatives = []
print("Detectando falsos positivos e falsos negativos")
for limit in limits:
    fp = 0
    fn = 0
    for j in range(len(distance)):
        database = distance[j]
        if database <= limit and true_or_false[j] == False:
            fp += 1
        elif database > limit and true_or_false[j] == True:
            fn += 1

    fp = fp / float(negatives_total)
    fn = fn / float(positives_total)
    false_positives.append(fp)
    false_negatives.append(fn)

print("Plotando gráficos FARxFRR")
false_positives = np.array(false_positives)
false_negatives = np.array(false_negatives)
index_EER = np.argmin(np.absolute(false_positives - false_negatives)) #indice para a diferença mínima entre falso positivos e falso negativos
plt.axis([0, 1.0, 0, 1.0])
plt.title('FAR x FRR graph')
plt.xlabel('False Positives')
plt.ylabel('False Negatives')
plt.plot(false_positives, false_negatives, linestyle='--', color='black')
plt.plot(false_positives[index_EER], false_negatives[index_EER], 'o', color='green', label='EER')
plt.legend()
plt.savefig('DETcurve.png')

lbps = []
major = 0
for img in casia_interval.iris:
    if img.shape[0] > major:
        major = img.shape[0]

c = 0
for img in casia_interval.iris:
    img = np.resize(img, (major,img.shape[1]))
    print ("LBP da imagem "+str(c+1))
    lbps.append(feature.local_binary_pattern(img, img.shape[1], img.shape[0], method="uniform"))
    c+=1

#dividir imagens de treino e teste 10 vezes
kfold = KFold(n_splits=10, shuffle=True)
accuracy = 0
for train_index, test_index in kfold.split(lbps):
    data_train = []
    data_test = []
    label_train = []
    label_test = []
    # print("TRAIN:", train_index, "TEST:", test_index)
    #labels referenciada pelo indice da imagem
    for train in train_index:
        data_train.append(lbps[train])
        label_train.append(casia_interval.labels[train])

    print("Treinando SVM")
    model = svm.SVC(kernel='linear', C = 1.0)
    data_train = np.reshape(data_train, (len(data_train), data_train[0].shape[0]*data_train[0].shape[1]))
    model.fit(data_train, label_train)

    for test in test_index:
        data_test.append(lbps[test])
        label_test.append(casia_interval.labels[test])

    data_test = np.reshape(data_test, (len(data_test), data_test[0].shape[0]*data_test[0].shape[1]))
    accuracy += model.score(data_test, label_test)

print ("Mean accuracy for CASIA Interval: "+str(accuracy/10))
