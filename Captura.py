# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:34:45 2019

@author: Felipe Lelis
"""
import cv2

classificador = cv2.CascadeClassifier("haarcascade-frontalface.xml")
camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostras = 25
id = input('Digite seu identificador: ')
largura, altura = 220, 220
print("Capturando as fazes ...")

while (True):
	conectado, imagem = camera.read()

	imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
	facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor =1.5,
		minSize=(150,150))
	for (x, y, l, a) in facesDetectadas:
		cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 0, 255), 2)
		if cv2.waitKey(1) & 0xFF ==ord('q'):
			imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
			cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) +".jpg", imagemFace)
			print("[foto" + str(amostra) + "capturada com sucesso")
			amostra += 1
			if(amostra >= numeroAmostras +1):
				break
				print("Faces capturadas com Sucesso")
	cv2.imshow("face", imagem)
	cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()