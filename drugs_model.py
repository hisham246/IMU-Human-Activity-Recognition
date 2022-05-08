import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import tkinter as tk

# Importing the dataset
dataset = pd.read_csv("basse.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



#converte ver string
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1497])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3022])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3826])], remainder='passthrough')
x = np.array(ct.fit_transform(x))



le = LabelEncoder()
y = le.fit_transform(y)


#from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs
#x , y = make_blobs(n_samples=125 , centers=2 , cluster_std=0.60 , random_state=0)

#splite data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


#plot svm
#plt.scatter(x_train[:,0] ,x_train[:,1], c=y_train, cmap='winter')


#Classifier training using Support Vector Machine 
model = SVC()
model.fit(x_train,y_train)

# sparse=False Check classifier accuracy on test data and see result 
predict_medic = model.predict(x_test)
print("Accuracy: ",accuracy_score(y_test, predict_medic))


#application 
from tkinter import*
root=Tk() 
root.geometry("600x600") 
root.configure (background="light green")
 
Label(root, text="application pour classification des medicaments", 
       font=('Helvetica', 15, 'bold'), bg="light green", relief="solid").pack()
#Label(root, text="application version 1.1", relief="solid",bg="light green").pack(side=BOTTOM)



Label (root, text="ID de medicaments", font=('Helvetica', 10, 'bold'), 
         bg="white",relief="solid").place(x=40, y=70)
Label (root, text="medicaments actif", font=('Helvetica', 10, 'bold'), 
         bg="white",relief="solid").place(x=40, y=120) 
Label (root, text="class medicaments", font=('Helvetica', 10, 'bold'), 
         bg="white",relief="solid").place(x=40, y=180)


#Label (root, text="resultat_categorie ", font=('Helvetica', 10, 'bold'),bg="white",relief="solid").place(x=40, y=260)


sl=tk.StringVar()
sm=tk.StringVar()
sn=tk.StringVar()

Entry(root, text=sl,width=25).place(x=200, y=70)
Entry(root, text=sm,width=25).place(x=200, y=120)
Entry(root, text=sn,width=25).place(x=200, y=180)



#modele app

def modele():
    dataset = pd.read_csv('basse.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values




#converte ver string
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1497])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3022])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3826])], remainder='passthrough')
x = np.array(ct.fit_transform(x))



le = LabelEncoder()
y = le.fit_transform(y)


#Classifier training using Support Vector Machine 
model = SVC()
model.fit(x_train,y_train)



# sparse=False Check classifier accuracy on test data and see result 

x_test= [str(sl.get()),str(sm.get()),str(sn.get())]
predict_medic = model.predict([x_test,])

Label (root, text=str(accuracy_score(y_test, predict_medic)),font=('Helvetica', 10, 'bold'), 
         bg="white",relief="solid").place(x=200, y=320) 


    
Button(root, text="resultat",width=18, command=modele).place(x=40,y=409)

root.resizable(0,0)
root.mainloop() 