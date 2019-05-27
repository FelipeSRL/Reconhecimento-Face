# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:34:45 2019

@author: Felipe Lelis
"""

import cv2
import os
import numpy as np
from PIL import Image

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def getImagemComId():
    caminhos = [os.path.join('yalefaces/treinamento', f) for f in os.listdir('yalefaces/treinamento')]
    print(caminhos)
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = Image.open(caminhoImagem).convert('L')
        imagemNP = np.array(imagemFace, 'uint8')
        id = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
        ids.append(id)
        faces.append(imagemNP)
        # cv2.imshow("Face", imagemFace)
        # cv2.waitKey(10)
    return np.array(ids), faces


ids, faces = getImagemComId()
print("Treinando ...")

eigenface.train(faces, ids)
eigenface.write('classificadorEigenYale.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisherYale.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPHYale.yml')

print('Treinamento Realizado')
