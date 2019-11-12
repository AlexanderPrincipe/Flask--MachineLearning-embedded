from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import time 
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as py
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,roc_curve
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_curve
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/tasks.db'
db = SQLAlchemy(app)

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200))
    done = db.Column(db.Boolean)

@app.route('/')
def home():
    tasks = Task.query.all()
    return render_template('index.html', tasks = tasks)

@app.route('/create-task', methods=['POST'])
def create():
    task = Task(content=request.form['content'], done=False)
    db.session.add(task)
    db.session.commit()
    return redirect(url_for('home'))

@app.route('/done/<id>')
def done(id):
    task = Task.query.filter_by(id=int(id)).first()
    task.done = not(task.done)
    db.session.commit()
    return  redirect(url_for('home'))

@app.route('/delete/<id>')
def delete(id):
    task = Task.query.filter_by(id=int(id)).delete()
    db.session.commit()
    return redirect(url_for('home'))
    
@app.route('/svm')
def svm():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import numpy as np
    import pandas as pd
    import time 
    import matplotlib.pyplot as plt
    import seaborn as sns

    import numpy as py
    from sklearn import svm
    from sklearn import metrics
    from sklearn.metrics import classification_report,confusion_matrix,roc_curve
    import csv
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report,confusion_matrix,roc_curve
    from sklearn import metrics
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_selection import SelectKBest,f_regression
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import RFE
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import Imputer


    ################# dataset antiguo #################################################################################
    '''
    cancer_df=pd.read_csv("base23.csv",names=["Edad","Grupo de edad","Genero","Etnia","Zona","Regimen de Afiliación",
                                        "Fumador Activo","Hipertensión Arterial sistemica","Nivel de Glicemia",
                                        "Complicaciones  y Lesiones en Organo Blanco","Antecedentes Fliar  Enfermedad Coronaria",
                                        "Tension SISTOLICA","Tension DIASTOLICA","HTA COMPENSADOS"	,"Colesterol Total","Trigliceridos",
                                        "Clasificación de RCV Global","Glicemia de ayuno","Perimetro Abdominal","Clasificación perímetro abdominal",
                                        "Peso","Talla","IMC","CLAIFICACION IMC","Creatinina","Factor de corrección de la formula","Proteinuria",
                                        "Farmacos Antihipertensivos","Estatina","Antidiabeticos","Adherencia al tratamiento","Diabetes"])

    data=np.genfromtxt("base23.csv",delimiter=",")
    cancer_df.head(4)
        
        
    #idx= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] 
    #este y 12 componentes principales 72.6%
    idx= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]    

    '''
    ######################################################################################################################
    ################# dataset nuevo

    cancer_df=pd.read_csv("tesis2222.csv",names=["Edad","Genero","Etnia","Zona",
                                        "Fumador Activo","Hipertensión Arterial Sistemica","Nivel de Glicemia",
                                        "Complicaciones  y Lesiones en Organo Blanco","Antecedentes Fliar  Enfermedad Coronaria",
                                        "Tension SISTOLICA","Tension DIASTOLICA","HTA COMPENSADOS"	,"Colesterol Total","Colesterol HDL","Trigliceridos","Colesterol LDL",
                                        "Clasificación de RCV Global","Glicemia de ayuno","Perimetro Abdominal","Clasificación perímetro abdominal",
                                        "Peso","Talla","CLAIFICACION IMC","Creatinina","Factor de corrección de la formula","Microalbuminuria","Proteinuria",
                                        "Calculo de  TFG corregida (Cockcroft-Gault)","Estadio IRC","Farmacos Antihipertensivos","Estatina","Antidiabeticos","Adherencia al tratamiento","Diabetes"])

    data=np.genfromtxt("tesis2222.csv",delimiter=",")
    #numfields = 39
    #fieldwidth = 5894
    #data = np.genfromtxt("tesis2222.csv", dtype='S%d' % fieldwidth, delimiter=(fieldwidth,)*numfields)
    #cancer_df.head(4)

    #para PCA
    idx= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    #idx= [0,1,2,3,4,6,7,9,12,13,14,15,16,17,18,20,21,23,24,25,26,27,28,29,30,31,32] 
    #correlacionadas
    #idx= [20,21,22,2,0,24,18,16,9] 
    #63% con 3
    #idx= [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,24,25,26,27,28,29,30,31,32] 
    #idx= [1,2,3,4,6,7,9,12,13,14,15,16,17,20,21,23,25,26,27,29,30,31,32] 
    #no ultimo(no sirve)
    #idx= [0,2,5,8,9,10,11,16,18,19,20,21,22,24,25,28,30,32] 

    ##chi-square
    #idx= [0,3,6,9]

    ###############################################
    #dataset 2224
    #################
    '''
    cancer_df=pd.read_csv("tesis2224.csv",names=["Edad","Genero","Etnia","Zona",
                                        "Fumador Activo","Nivel de Glicemia",
                                        "Complicaciones  y Lesiones en Organo Blanco","Antecedentes Fliar  Enfermedad Coronaria",
                                        "Tension SISTOLICA","Tension DIASTOLICA","Colesterol Total","Colesterol HDL","Trigliceridos","Colesterol LDL",
                                        "Clasificación de RCV Global","Glicemia de ayuno",
                                        "Peso","Talla","Creatinina","Factor de corrección de la formula","Microalbuminuria","Proteinuria",
                                        "Calculo de  TFG corregida (Cockcroft-Gault)","Estadio IRC","Farmacos Antihipertensivos","Estatina","Antidiabeticos","Adherencia al tratamiento","Diabetes"])

    data=np.genfromtxt("tesis2224.csv",delimiter=",")
    idx= [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,18,19,20,21,22,24,25,26,27]

    '''
    
    #############################################################################################

    
    cancer_data=data[:,idx]
        ##X=data[:,:-1]
    imputer = Imputer()
    cancer_data = imputer.fit_transform(cancer_data)
    cancer_target=data[:,-1]


    print ("Shape of DataFrame: ", cancer_df.shape)
    benign = len(cancer_data[cancer_target==1])
    print ("number of benign samples: ", benign)
    print (cancer_df.columns)
    # select the columns with names mean, error and worst 
    feature_mean  = list(cancer_df.columns[0:10])
    feature_error = list(cancer_df.columns[11:20])
    feature_worst = list(cancer_df.columns[20:29])


    mean_corr = cancer_df[feature_mean].corr()
    print (mean_corr)
    #fig = plt.figure()
    #########################################################
    sample_size=1174
    random_seed=6
    uniq_levels = np.unique(cancer_target) 
    uniq_counts = {level: sum(cancer_target == level) for level in uniq_levels} 
    if not random_seed is None: 
        np.random.seed(random_seed) 

        
    groupby_levels = {} 
        
    for ii, level in enumerate(uniq_levels): 
        obs_idx = [idx for idx, val in enumerate(cancer_target) if val == level] 
        groupby_levels[level] = obs_idx 
    balanced_copy_idx = [] 

    c=balanced_copy_idx 
    ##############################################
    minmax=preprocessing.MinMaxScaler(feature_range=(0, 1))
    cancer_data=minmax.fit_transform(cancer_data)
    np.savetxt("importado1.csv", cancer_data)
    np.savetxt("target_importado1.csv", cancer_target)
    #X_train,X_test,y_train,y_test=train_test_split(cancer_data,cancer_target,random_state=3)

    LDA_tr=np.genfromtxt("lda1.csv",delimiter=",")
    LDA_te=np.genfromtxt("lda2.csv",delimiter=",")
    LDA_tr.shape
    X_train = LDA_tr[:,0]
    X_test = LDA_te[:,0]
    Y_train = LDA_tr[:,-1]
    Y_test = LDA_te[:,-1]

    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    #Y_train = Y_train.reshape(-1,1)
    #Y_test = Y_test.reshape(-1,1)


    # Pipeline Steps are StandardScaler, PCA and SVM 
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV



    clf=svm.SVC(gamma= 0.6 , kernel='rbf', C=10)
    #clf=svm.LinearSVC()
    clf.fit(X_train,Y_train)

        

    Y_pred = clf.predict(X_test)
        
        
    print(confusion_matrix(Y_test,Y_pred))
    score = clf.score(X_test,Y_test)
    print ("Accuracy(single): %0.2f" % (clf.score(X_test,Y_test)))
    #print(score)    
    print(classification_report(Y_test,Y_pred))
    clasificacion = classification_report(Y_test,Y_pred)

    ###############################################################
    return render_template('svm.html', value = clasificacion.split())



if __name__ == '__main__':
    app.run(debug=True)