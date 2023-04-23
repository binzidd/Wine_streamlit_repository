
import pandas as pd
import numpy as np
import time 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay

from matplotlib import pyplot as plt
import dtreeviz
import seaborn as sns
import base64

import streamlit as st
from io import StringIO
import streamlit.components.v1 as components
from svgelements import *


#Page Config 
st.set_page_config(
    page_title="Ex-stream-ly Cool Wine App",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state='auto'
    )

##Header 
st.title('De-code Red Wines 🍷🍷')

st.subheader('AI Experiment to determine Wine Quality')


# ### Prepare Dataset

# In[2]:



def load_data(WINE_SCORE_LIMIT):
    
    wine=pd.read_csv(uploaded_file, sep=';')
    wine['quality'] = np.where(wine['quality']>=WINE_SCORE_LIMIT, 1,0) # 1 stands for decent wine,0 - cooking wine
    return wine 



with st.container():
    
    st.subheader('Step 1: Load a CSV file')
    data_load_state = st.text('Loading data...')
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=None)
    
    for percent_complete in range(100):
        time.sleep(0.001)
        my_bar.progress(percent_complete + 1, text=None)
    
    uploaded_file=st.file_uploader("Chose a file")
    
    if uploaded_file != None:
    #Creating a sidebar to capture Wine Score Limit 
        
            st.sidebar.subheader("Filter")
            WINE_SCORE_LIMIT = st.sidebar.slider(
            "Choose a Wine Score Limit",
            0, 10,6)
            wine = load_data(WINE_SCORE_LIMIT)
            data_load_state.text("'Data Loaded and Cached'")
            st.write(wine)
    else: 
         st.error("Cannot Continue Further, Please load a CSV File", icon="🚨")
         st.stop()

    #submitted = st.form_submit_button("Explore Data")

    
    st.subheader('Step 2: Drop Non Relevant Columns')
    cols=list(wine.columns)
       
    Drop_cols = st.multiselect(
    'Select the Columns You Want to drop',
    list(wine.columns),
    default=['volatile acidity','free sulfur dioxide','pH','sulphates','density'],
    )
    wine=wine.drop(Drop_cols, axis=1)
    wine=wine.rename(columns={"fixed acidity": "Sourness", "citric acid": "Fruitiness",
                  "residual sugar":"Sweetness","chlorides":"Saltiness","total sulfur dioxide":"Preservatives",
                 "alcohol":"Alcohol%","quality":"Quality"})

    st.markdown("""---""")
    st.subheader('Step 3: Describe Current Data')
    desc=wine.describe()
    st.write(wine.head(10))

    st.markdown("""---""")
    st.subheader("Step 4: Investigate Data using a Confusion Matrix")
    corr = wine.corr()
       # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(145, 20, as_cmap=True)
    f, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
         
    st.pyplot(f)

    

st.markdown("""---""")
st.header("Lets Do Feature Engineering 😎 and Create few Classification Models")

classifier_name = st.selectbox(
    'Select Model',
    ('Decision Tree', 'SVM', 'Random Forest'))

def decisiontreeclassifier(x,y,rand_seed,v_depth):
    train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=rand_seed)
    winemodel = DecisionTreeClassifier(random_state=rand_seed, max_depth=v_depth)
    winemodel.fit(train_x, train_y)
    val_predictions1=winemodel.predict(val_x)
    model_acc=round(accuracy_score(val_y, val_predictions1)*100,2)
    model_f1=round(f1_score(val_y, val_predictions1)*100,2)
    return model_acc, train_x,train_y,winemodel

def randomforestclassifier(x,y,rand_seed,v_n_estimators):
     train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=rand_seed)
     winemodel = RandomForestClassifier(random_state=rand_seed, n_estimators = v_n_estimators) 
     winemodel.fit(train_x, train_y)
     winemodel.predict(val_x)
     val_predictions=winemodel.predict(val_x)
     model_acc=round(accuracy_score(val_y, val_predictions)*100,2)
     model_f1=round(f1_score(val_y, val_predictions)*100,2)
     return model_acc,model_f1 

    
def add_parameter_ui(classifier_name):
    if classifier_name == 'Decision Tree':
        st.header("Model 1 (DecisionTreeClassifier)")
        st.subheader("Step 1: Choose a Random Seed and Depth")
        rand_seed = st.slider("Choose Random Seed for Model",0, 1000,123)
        v_depth=st.slider("Choose Depth for Model",0,10,5)
        st.subheader('Step 2: Choose Features for Model')
        features = st.multiselect('Select the Columns You Want to select',
                                      list(wine.columns),
                                      ['Sourness','Fruitiness','Sweetness'])
        x=wine.filter(features,axis=1)
        y=wine['Quality']
        model1_acc, train_x,train_y,winemodel=decisiontreeclassifier(x,y,rand_seed,v_depth)
        st.subheader("Model's Accuracy is "+ f"{model1_acc}")
        return model1_acc
    
    elif classifier_name == 'Random Forest':
         st.header("Model 2: Random Forest")
         st.subheader("Step 1: Chose Random Seed and Estimators")
         rand_seed = st.slider("Choose Random Seed for Model",0, 1000,123,key="DF")
         v_n_estimators=st.slider("Choose number of trees in the forest Model",0,1000,250,help="Number of Trees for forest")
         x=wine.drop(['Quality'],axis=1)
         y=wine['Quality']
         model_acc, model_f1=randomforestclassifier(x,y,rand_seed,v_n_estimators)
         st.subheader("Model's Accuracy is "+ f"{model_acc}% "+ "& F1 score is "+ f"{model_f1}" )
    
    else: st.write("Model in development #to-do#") 


params = add_parameter_ui(classifier_name)
