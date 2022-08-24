from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score


main = tkinter.Tk()
main.title("OthoPedic Disease")
main.geometry("1300x1200")

def upload():
    global filename
    global data
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = ".")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")

def importdata():
    global filename
    global df
    df = pd.read_csv(filename,encoding = 'latin1')
    text.insert(END,"Data Information:\n"+str(df.head())+"\n")
    text.insert(END,"Columns Information:\n"+str(df.columns)+"\n")
    df.plot.bar()
    plt.show()
    
    
def preprocess():
    global df
    global x,y
    x= df.iloc[:,: -1]
    y=df.iloc[:,-1]
    color_list = ['red' if i=='Hernia' else ('purple' if i=='Spondylolisthesis' else 'green' ) for i in df.loc[:,'class']]
    pd.plotting.scatter_matrix(df.loc[:, df.columns != 'class'],
                               c=color_list,# c - color
                               figsize= [15,15],# figure size
                               diagonal='hist',# histohram of each features
                               alpha=0.5,# opacity
                               s = 150, # size of marker
                               marker = 'o',# marker type
                               edgecolor= "black")
    plt.show()
def ttmodel():
    global x,y
    global x_train,x_test,y_train,y_test
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)
    text.insert(END,"Train Shape: "+str(x_train.shape)+"\n")
    text.insert(END,"Test Shape: "+str(x_test.shape)+"\n")

def mlmodels():
    global df
    global clf_lr_acc,clf_rfc_acc,clf_ada_acc,clf_knn_acc
    global x_train,x_test,y_train,y_test
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(x_train,y_train)
    pred = knn.predict(x_test)
    clf_knn_acc=knn.score(x_test,y_test)
    print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test))
    neighbor = np.arange(1,25)
    train_accuracy = []
    test_accuracy = []

    for i,k in enumerate(neighbor):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train,y_train)
        train_accuracy.append(knn.score(x_train, y_train))
        test_accuracy.append(knn.score(x_test, y_test))

    
    print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
    clf_lr = LogisticRegression(random_state=0)
    clf_lr.fit(x_train,y_train)
    clf_lr.score(x_test,y_test)
    clf_lr_acc=clf_lr.score(x_test, y_test)
    text.insert(END,"Logit Accuracy: "+str(clf_lr.score(x_test, y_test))+"\n")
    text.insert(END,"Logit recall_score: "+str(recall_score(y_test,pred,average='micro'))+"\n")
    text.insert(END,"Logit precision_score: "+str(precision_score(y_test,pred,average='micro'))+"\n")
    text.insert(END,"Logit f1_score: "+str(f1_score(y_test,pred,average='micro'))+"\n")
    
    clf_rfc = RandomForestClassifier(random_state=0)
    clf_rfc.fit(x_train,y_train)
    clf_rfc.score(x_test,y_test)
    clf_rfc_acc=clf_rfc.score(x_test, y_test)
    text.insert(END,"RFC Accuracy: "+str(clf_rfc.score(x_test, y_test))+"\n")
    text.insert(END,"RFC recall_score: "+str(recall_score(y_test,pred,average='micro'))+"\n")
    text.insert(END,"RFC precision_score: "+str(precision_score(y_test,pred,average='micro'))+"\n")
    text.insert(END,"RFC f1_score: "+str(f1_score(y_test,pred,average='micro'))+"\n")
    

    x= df.iloc[:,: -1]
    y=df.iloc[:,-1]

    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)
    lb= LabelEncoder()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    clf_rfc= RandomForestClassifier(random_state=0)

    clf_ada_rf= AdaBoostClassifier(clf_rfc,
                              n_estimators=300, random_state=0)

    clf_ada_rf.fit(x_train,y_train)

    clf_ada_acc=clf_ada_rf.score(x_test, y_test)
    text.insert(END,"ADA Accuracy: "+str(clf_ada_rf.score(x_test, y_test))+"\n")
   

    # Plot
    plt.figure(figsize=[13,8])
    plt.plot(neighbor, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neighbor, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(neighbor)
    plt.savefig('graph.png')
    plt.show()

def graph():
    global clf_lr_acc,clf_rfc_acc,clf_knn_acc,clf_ada_acc
    
    height = [clf_lr_acc,clf_rfc_acc,clf_knn_acc,clf_ada_acc]
    bars = ('Logit','RFC','KNN','ADA')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()   
                
font = ('times', 16, 'bold')
title = Label(main, text='Classification and prediction of Orthopedic disease based on lumber and pelvic state of patients')
title.config(bg='dark salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700,y=150)

ip = Button(main, text="Data Import", command=importdata)
ip.place(x=700,y=200)
ip.config(font=font1)

pp = Button(main, text="Data Preprocessing", command=preprocess)
pp.place(x=700,y=250)
pp.config(font=font1)

tt = Button(main, text="Train and Test Model", command=ttmodel)
tt.place(x=700,y=300)
tt.config(font=font1)

ml = Button(main, text="Run Algorithms", command=mlmodels)
ml.place(x=700,y=350)
ml.config(font=font1)

gph = Button(main, text="Accuracy Graph", command=graph)
gph.place(x=700,y=400)
gph.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='mint cream')
main.mainloop()




