
import streamlit as st
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score, ShuffleSplit
# Confusion Matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Linear Algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# Non-Linear Algorithm
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as CART
from sklearn.svm import SVC  # Support Vector Classifier

from sklearn.model_selection import LeaveOneOut

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
##regression 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut
from sklearn.model_selection import ShuffleSplit


st.sidebar.header('User Input parameters')
selection = st.sidebar.selectbox('Choose an option', ["pima", "housing"])

def metrics(conf_matrix):
    TN = matrix[0, 0]
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    TP = matrix[1, 1]
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2 * ((recall * precision) / (precision + recall))
    spec = (TN / (TN + FP))
    roc = (TP / (TP + FP))
    return accuracy, recall, precision, f1, spec,roc

    
# define metrics function


if selection == "pima" :   
    filename = 'pima-indians-diabetes.data.csv'
    attributs = ['preg','plas','pres','skin','test','mass','pedi','age','class']
    dataframe = read_csv(filename,names=attributs)
    values = dataframe.values
    X= values[  :  , 0: -1] # Toutes les lignes et toutes les colonnes (sauf la derniere)
    Y =  values[ : , -1]
algo=st.selectbox('Choisir un algorithm :',('Train-test split','K-folds cross validation','Leave one out cross validation','Repeated random train-test split'))
metric=st.selectbox('Choisir une metric :',('accuracy','recall','precision','f1','specificity','roc_auc'))


mod=st.selectbox('Choisir un model :',('LR','LDA','KNN','CART','NB','SVM'))
if mod=='LR':
     model=LogisticRegression()
elif mod =='LDA'   :
    model=LDA()
elif mod=='KNN'   :
    model=KNN()
elif mod=='CART'   :
    model=CART()
elif mod == 'NB':
    model = GaussianNB()
elif mod == 'SVM':
    model = SVC(probability=True)


# Import dataset
filename = 'pima-indians-diabetes.data.csv'
attributes = ['preg', 'plas', 'pres', 'skin', 'test',
              'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=attributes)
values = dataframe.values  # Retrieve values from the dataframe
X = values[:, 0:-1]
Y = values[:, -1]

seed = 7
test = 0.30

tran=st.selectbox('Choisir une transformation :',('None','Rescale','Standardization','Normalization','Binarization'))
if tran=='Rescale':
     transform=MinMaxScaler()
     scaler = transform
     transformedX = scaler.fit_transform(X)
     st.write("%s Transform \n-------------" % transform)
elif tran=='Standardization'   :
    transform=StandardScaler()
    scaler = transform
    transformedX = scaler.fit_transform(X)
    st.write("%s Transform \n-------------" % transform)
elif tran=='Normalization'   :
    transform = Normalizer()
    scaler = transform
    transformedX = scaler.fit_transform(X)
    st.write("%s Transform \n-------------" % transform)
elif tran=='Binarization'   :
    transform = Binarizer()
    scaler = transform
    transformedX = scaler.fit_transform(X)
    st.write("%s Transform \n-------------" % transform)
else:
    transform=''
    transformedX=X
    st.write("%s pas de transformation \n-------------" % transform)




if algo=='Train-test split':

 X_train, X_test, Y_train, Y_test = train_test_split(transformedX, Y, test_size=test, random_state=seed)
 model.fit(X_train, Y_train)
 predicted = model.predict(X_test)
 matrix = confusion_matrix(Y_test, predicted)
 acc, rec, prec, f1, spec, roc = metrics(matrix)
 if metric == 'accuracy':
     st.write("Accuracy: %s" % (acc * 100).round(2))
 elif metric == 'recall':
     st.write("Recall: %s" % (rec * 100).round(2))
 elif metric == 'precision':
     st.write("Precision: %s" % (prec * 100).round(2))
 elif metric == 'f1':
     st.write("F1-Score: %s" % (f1 * 100).round(2))
 elif metric == 'roc_auc':
     st.write("Area under the curve: %s" % (roc * 100).round(2))
 else:
     st.write("Specificity: %s " % (spec * 100).round(2))

elif algo=='K-folds cross validation':
    kfold = KFold(n_splits=7, random_state=7, shuffle=True)
    result=cross_val_score(model, transformedX, Y, cv=kfold, scoring=metric)
    st.write(result.mean())

elif algo=='Leave one out cross validation':
    loocv = LeaveOneOut()
    result=cross_val_score(model, transformedX, Y, cv=loocv, scoring=metric)
    st.write(result.mean())
else:
    kfold=ShuffleSplit(n_splits=7,test_size=7,random_state=7)
    result=cross_val_score(model,transformedX,Y,cv=kfold,scoring=metric)
    st.write(result.mean())









def pima():
 Preg = st.sidebar.slider('Preg', 0.0,17.0,10.0)  # (min, max, default)
 Plas = st.sidebar.slider('Plas',0,199,110)
 Pres = st.sidebar.slider('Pres',0,122,10)
 Skin = st.sidebar.slider('Skin',0,99,800)
 Test = st.sidebar.slider('Test', 0,846,70)
 Mass = st.sidebar.slider('Mass',0.0,67.100,6.0)
 Pedi = st.sidebar.slider('Pedi',0.078,2.420,0.078)
 Age = st.sidebar.slider('Age', 21,81,40)

 data = {
         'Pregt': Preg,
         'Plas': Plas,
         'Pres': Pres,
         'Skin': Skin,
         'Test ':Test ,
         'Mass ': Mass,
         'Pedi ': Pedi,
         'Age ': Age


     }
 features = pd.DataFrame(data, index=[0])
 return features


df = pima()
st.subheader("User Input parameters")
st.write(df)



model.fit(X, Y)
# Prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)
# display
st.subheader("class labels and theirs corresponding index number")

# Display Prediction
st.write('Prediction')
st.write(prediction)
# Display probability
st.write('Prediction probability')
st.write(prediction_proba)


if selection == "housing" :   
    st.sidebar.header('User Input parameters')
    mod = st.selectbox(
    'Choisir un modele',
    ( 'K-Fold Cross Validation ','Train-Test Split','Leave One Out Cross Validationn','Repeated Random Test-Train Split'))
algorithm = st.selectbox("Choisir un algorithme",
                            ('LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'KNeighborsRegressor',
                             'DecisionTreeRegressor', 'SVR'))
metric = st.selectbox(
            "Choisir une métric ",
            ('MAE', 'MSE', 'R2'))
transformation = st.selectbox(
                "Choisir une transformation ",
                ('Aucun', 'rescale', 'Normalization', 'Binarization', 'Standardization'))


def housing():
    CRIM = st.sidebar.slider('CRIM', 0.0, 88.9762, 30.0)
    ZN = st.sidebar.slider('ZN', 0.0, 100.0, 30.0)
    INDUS = st.sidebar.slider('INDUS', 0.46, 27.74, 3.0)
    CHAS = st.sidebar.slider('CHAS', 0.0, 1.0, 0.5)
    NOX = st.sidebar.slider('NOX', 0.385, 0.871, 0.666)
    RM = st.sidebar.slider('RM', 3.561, 8.78, 8.0)
    AGE = st.sidebar.slider('AGE', 2.9, 100.0, 50.0)
    DIS = st.sidebar.slider('DIS', 1.1296, 12.1265, 7.0)
    RAD = st.sidebar.slider('RAD', 1.0, 24.0, 19.0)
    TAX = st.sidebar.slider('TAX', 187.0, 711.0, 300.0)
    PTRATIO = st.sidebar.slider('PTRATIO', 12.0, 22.0, 20.0)
    B = st.sidebar.slider('B', 0.32, 396.9, 96.0)
    LSTAT = st.sidebar.slider('LSTAT', 1.73, 37.97, 33.0)
    data = {
        'CRIM': CRIM,
        'ZN': ZN,
        'INDUS': INDUS,
        'CHAS': CHAS,
        'NOX': NOX,
        'RM': RM,
        'AGE': AGE,
        'DIS': DIS,
        'RAD': RAD,
        'TAX': TAX,
        'PTRATIO': PTRATIO,
        'B': B,
        'LSTAT': LSTAT,

     }
    features = pd.DataFrame(data, index=[0])
    return features
filename = 'housing.csv'
attributs = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
             'B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, delim_whitespace=True, names=attributs)

dataframe_values = dataframe.values
X = dataframe_values[:, 0: -1]
Y = dataframe_values[:, -1]
df = housing()



if algorithm == "LinearRegression":
    algoName = LinearRegression
elif algorithm == "Ridge":
    algoName = Ridge
elif algorithm == "Lasso":
    algoName = Lasso
elif algorithm == "ElasticNet":
    algoName = ElasticNet
elif algorithm == "KNeighborsRegressor":
    algoName = KNeighborsRegressor
elif algorithm == "DecisionTreeRegressor":
    algoName = DecisionTreeRegressor
elif algorithm == "SVR":
    algoName = SVR
model = algoName()

if transformation== "rescale":
    transName = StandardScaler
    transformedX = transName().fit_transform(X)
elif transformation == "Normalization":
    transName = Normalizer

    transformedX = transName().fit_transform(X)
elif transformation == "Binarization":
    transName = Binarizer

    transformedX = transName().fit_transform(X)
elif transformation == "Standardization":
    transName = MinMaxScaler

    transformedX = transName().fit_transform(X)

elif transformation == "Aucun":
    transformedX = X

if metric == "MAE":
    metricName = "neg_mean_absolute_error"
elif metric == "MSE":
    metricName = "neg_mean_squared_error"
elif metric == "R2":
    metricName = "r2"




num_fold = 10
seed = 7
kfold = KFold(n_splits=num_fold, random_state=seed, shuffle=True)
results = cross_val_score(model, transformedX, Y, cv=kfold, scoring=metricName)

if mod == "K-Fold Cross Validation":
    divisionName = KFold
    num_fold = 10
    seed = 7
    kfold = KFold(n_splits=num_fold, random_state=seed, shuffle=True)
    results = cross_val_score(model, transformedX, Y, cv=kfold, scoring=metricName)

elif mod == "Leave One Out Cross Validationn":
    divisionName = LeaveOneOut()
    num_fold = 10
    seed = 7
    results = cross_val_score(model, transformedX, Y, cv=divisionName, scoring=metricName)
elif mod == "Repeated Random Test-Train Split":
    divisionName = ShuffleSplit
    num_fold = 10
    seed = 7
    shuffle = divisionName(n_splits=num_fold,test_size = 0.3, random_state=seed)
    results = cross_val_score(model, transformedX, Y, cv=shuffle, scoring=metricName)
model.fit(transformedX, Y)

prediction = model.predict(df)

st.subheader('Prediction :')
st.write(prediction)
st.write('-------------------------------')
st.subheader("Metric choisie : " + metric)
st.write(results.mean().round(2))
st.write('-------------------------------')
st.write("Input Values: ")
st.write(df)