import numpy as np
from dataset import Dataset
from utils import iris_detection, iris_verification, iris_identification

def execute(path):
    dataset = Dataset(path)
    print("Reading dataset " + path)
    dataset.load_images_and_labels()
    print (str(len(dataset.images))+" images")
    iris_detection(dataset)
    if (len(dataset.images)%2 != 0): #se não for número par de imagens
        np.delete(dataset.images, len(dataset.images)-1, 0)
    iris_verification(dataset)
    iris_identification(dataset)

if __name__ == '__main__':
    execute('data/CASIA-Iris-Lamp-100')
    execute('data/CASIA-IrisV4-Interval')

