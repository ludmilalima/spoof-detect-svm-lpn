'''
Programa: Reconhecimento de Spoofing em Impressões digitais através de um
          modelo de inteligência artificial SVM Linear, com base nas texturas
          da imagem pelo mecanismo de Local Binary Pattern do respectivo.

Nome: Bruno Sampaio Leite
      Edson Junior Fontes Carvalho
      Talita Ludmila Lima

Matéria: Segurança da Informação
'''
# Bibliotecas auxiliares
import os
import cv2
import numpy as np
from imutils import paths as direc
from skimage import feature as ft

# Algoritmo de SVM
from sklearn.svm import LinearSVC

# Bibliotecas para análise dos resultados
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# ------------------ Local Binary Pattern --------------------------
def LBP(nPoints, rad, image, eps=1e-7):
    
    lbp = ft.local_binary_pattern(image, nPoints, rad, method="uniform")
    (histogram, _) = np.histogram(lbp.ravel(), bins=np.arange(0, nPoints + 3), range=(0, nPoints + 2))
    histogram = histogram.astype("float")
    histogram /= (histogram.sum() + eps)
    
    return histogram


sep = os.path.sep
# ------------------------ Treino --------------------------------

# Amostra do treinamento
database_train = []
# Classe do treinamento
label_train = []

for image in direc.list_images("Base_Imagens" + sep + "Treino"):
    
     # Leitura da imagem na escala de cinza
    fp_bw = cv2.imread(image, 0)

    # Extrai o histograma do local binary pattern da imagem
    
    hist = LBP(32,8, fp_bw)
    #hist = LBP(8,4, fp_bw)
    #hist = LBP(16,4, fp_bw)
    #hist = LBP(24,4, fp_bw)
    #hist = LBP(32,4, fp_bw)
    #hist = LBP(24,8, fp_bw)
    #hist = LBP(64,8, fp_bw)
    #hist = LBP(32,12, fp_bw)

    # Pega o nome da pasta da imagem para identificar a classe do respectivo
    label_train.append(image.split(sep)[-2])

    # Insere os valores do histograma
    database_train.append(hist)
    

# ----------------------- Modelo SVC ------------------------------
#model = LinearSVC(C=1, random_state=42, tol = 1e-2, max_iter= 4000)
#model = LinearSVC(C=10, random_state=42, tol = 1e-2, max_iter= 4000)
model = LinearSVC(C=100, random_state=42, tol = 1e-2, max_iter= 4000)
#model = LinearSVC(C=1000, random_state=42, tol = 1e-2, max_iter= 4000)
model.fit(database_train, label_train)


# ---------------------- Teste da base -----------------------------
database_test = [] 
label_test = []
SVC_label = []

for image in direc.list_images("Base_Imagens" + sep + "Teste"):

    # Leitura da imagem na escala de cinza
    fp_bw = cv2.imread(image, 0)

    # Extrai o histograma do local binary pattern da imagem
    
    hist = LBP(32,8, fp_bw)
    #hist = LBP(8,4, fp_bw)
    #hist = LBP(16,4, fp_bw)
    #hist = LBP(24,4, fp_bw)
    #hist = LBP(32,4, fp_bw)
    #hist = LBP(24,8, fp_bw)
    #hist = LBP(64,8, fp_bw)
    #hist = LBP(32,12, fp_bw)
    
    # Pega o nome da pasta da imagem para identificar a classe do respectivo
    label_test.append(image.split(sep)[-2])
    
    # Insere os valores do histograma
    database_test.append(hist)

    # Predição dos valores
    predict = model.predict(hist.reshape(1, -1))
    SVC_label.append(predict[0])

print("\nModelo previsionado com sucesso!\n")


# ------------------- Avaliação dos resultados -------------------------
confusion_matrix = confusion_matrix(label_test,SVC_label,labels=["Verdadeiro","Falso"])

print(" --------------------")
print(" |Matriz de confusão|")
print(" --------------------")
print(f" |TP = {str(confusion_matrix[0][0]).zfill(3)}| FN = {str(confusion_matrix[0][1]).zfill(3)}|")
print(f" |FP = {str(confusion_matrix[1][0]).zfill(3)}| TN = {str(confusion_matrix[1][1]).zfill(3)}|")
print(" --------------------")


print("\n-> Precisão: {:.2%}".format(precision_score(label_test,SVC_label,labels=["Verdadeiro","Falso"],pos_label="Verdadeiro")))
print("-> Recall: {:.2%}".format(recall_score(label_test,SVC_label,labels=["Verdadeiro","Falso"],pos_label="Verdadeiro")))
print("-> Acurácia: {:.2%}".format(accuracy_score(label_test,SVC_label)))
