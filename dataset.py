import os
import numpy as np
from PIL import Image

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
