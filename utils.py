import numpy as np
import cv2, os
import numpy as np
from PIL import Image
import math
import pywt
import sys
import progressbar
from skimage import feature
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from dataset import Dataset, preprocess, get_iris_radius, normalize_iris

def hamming_distance(code_query, code_database):
    nm = code_query.shape[0] * code_query.shape[1]
    xor = np.logical_xor(code_query, code_database).sum()
    return xor / float(nm)

def iris_detection(dataset):
    print("Detect, segment and normalize iris region")
    bar = progressbar.ProgressBar(maxval=len(dataset.images) - 1)
    bar.start()

    c = radius_iris = 0
    for index, image in enumerate(dataset.images):
        bar.update(index)
        pupil = preprocess(image)
        if pupil is not None: #achou a pupila
            radius_iris = get_iris_radius(image, pupil)
            image_normalized = normalize_iris(image, pupil, radius_iris)
            dataset.iris.append(image_normalized)
            #binariza a imagem com Haar Wavelet
            coefficients = pywt.wavedec2(image_normalized, 'haar', level=4)
            cA = coefficients[0]
            #horizontal detail, vertical detail and diagonal detail
            (cH, cV, cD) = coefficients[1]
            image_binar = np.zeros((cA.shape[0], cA.shape[1]), dtype='uint8')
            for i in range(cA.shape[0]):
                for j in range(cA.shape[1]):
                    if ((cH[i,j]>=0) and (cV[i,j]>=0) and (cD[i,j]>=0)):
                        image_binar[i,j] = 1
                    else:
                        image_binar[i,j] = 0
            #deixar todos do mesmo tamanho
            image_binar = np.resize(image_binar, (5,23)) 
            #salvo na ordem das labels tbm
            dataset.codes.append(image_binar)
        else: #não achou a pupila, remove a imagem
            np.delete(dataset.images, i, 0)
            c-=1
        c+=1
    bar.finish()

def iris_verification(dataset):
    distance = []
    true_or_false = []
    for i in range(0, len(dataset.codes)):
        code_query = dataset.codes[i]
        for j in range(i+1, len(dataset.codes)):
            code_database = dataset.codes[j]
            distance.append(hamming_distance(code_query, code_database))
            true_or_false.append((dataset.labels[i] == dataset.labels[j]))

    limits = np.sort(distance)
    limits = np.unique(limits)
    positives_total = 0
    negatives_total = 0
    for aux in true_or_false:
        if aux == True: # mesma label
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

    #indice para a diferença mínima entre falso positivos e falso negativos
    index_EER = np.argmin(np.absolute(false_positives - false_negatives)) 
    plt.axis([0, 1.0, 0, 1.0])
    plt.title('FAR x FRR graph')
    plt.xlabel('False Positives')
    plt.ylabel('False Negatives')
    plt.plot(false_positives, false_negatives, linestyle='--', color='black')
    plt.plot(false_positives[index_EER], false_negatives[index_EER], 'o', color='green', label='EER')
    plt.legend()
    plt.savefig('DETcurve.png')

def lbp(iris):
    lbps = []
    major = 0
    for img in iris:
        if img.shape[0] > major:
            major = img.shape[0]

    c = 0
    for img in iris:
        img = np.resize(img, (major,img.shape[1]))
        print ("LBP da imagem "+str(c+1))
        lbps.append(feature.local_binary_pattern(img, img.shape[1], img.shape[0], method="uniform"))
        c+=1

    return lbps

def iris_identification(dataset):
    lbps = lbp(dataset.iris)
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

    print ("Mean accuracy: "+str(accuracy/10))