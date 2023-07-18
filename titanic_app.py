# Libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib as jbl
# Build Pipelines
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# Empty values
from sklearn.impute import SimpleImputer
# Transform categorical data
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
#from category_encoders import MEstimateEncoder


# Import training data
DF = pd.read_csv("train.csv", header = 0)
X = DF.drop(['Survived'], axis = 1)


# Web app
st.write("""
# Titanic - Machine Learning from Disaster App

This app predicts who might **survive**.

Data obtained from [Kaggle competition](https://www.kaggle.com/c/titanic).
""")

# Sidebar input
st.sidebar.header('User Input Features')
uploaded_file = st.sidebar.file_uploader('Upload you input CSV file', type=['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_feature():
        PassengerIdV = st.sidebar.selectbox('PassengerId', X['PassengerId'].unique().tolist())
        PclassV = st.sidebar.selectbox('Pclass', X['Pclass'].unique().tolist())
        NameV = st.sidebar.selectbox('Name', X['Name'].unique().tolist())
        SexV = st.sidebar.selectbox('Sex', X['Sex'].unique().tolist())
        AgeV = st.sidebar.slider('Age', float(X['Age'].min()), float(X['Age'].max()), float(X['Age'].mean()))
        SibSpV = st.sidebar.selectbox('SibSp', X['SibSp'].unique().tolist())
        ParchV = st.sidebar.selectbox('Parch', X['Parch'].unique().tolist())
        TicketV = st.sidebar.selectbox('Ticket', X['Ticket'].unique().tolist())
        FareV = st.sidebar.slider('Fare', float(X['Fare'].min()), float(X['Fare'].max()), float(X['Fare'].mean()))
        CabinV = st.sidebar.selectbox('Cabin', X['Cabin'].unique().tolist())
        EmbarkedV = st.sidebar.selectbox('Embarked', X['Embarked'].unique().tolist())
        data = {'PassengerId': PassengerIdV,
               'Pclass': PclassV,
               'Name':NameV,
               'Sex':SexV,
               'Age':AgeV,
               'SibSp':SibSpV,
               'Parch':ParchV,
               'Ticket':TicketV,
               'Fare':FareV,
               'Cabin':CabinV,
               'Embarked':EmbarkedV}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_feature()


    
# Preview input data
st.subheader('User input features')
if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)
    
    
    
# Rebuild pipelines used in the model training
class OutliersTransformer(BaseEstimator, TransformerMixin):
    """
    Remove observations according to Fare since are possible outliers.
    Receive the whole DataFrame.
    Returns all DataFrama's columns but excluding the outliers.
    """
    def __init__(self, perc_threshold):
        self.perc_threshold = perc_threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        threshold = -1
        for i in range(100, 500, 50):
            obs_lost = X['Fare'].loc[X['Fare'] > i].count()
            tol_col = X['Fare'].count()
            perc_lost = obs_lost / tol_col

            if (perc_lost <= self.perc_threshold) and (threshold == -1):
                threshold = i
        
        X = X.loc[X['Fare'] <= threshold]
        return X



class WasMissingTransformer(BaseEstimator, TransformerMixin):
    """
    Create a class to add a column indicating which row with missing values.
    Receive a DataFrame and one string ColumnName.
    Returns the same DataFrame with an column added.
    """
    def __init__(self, ColumnName):
        self.ColumnName = ColumnName
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X[self.ColumnName + '_was_missing'] = X[self.ColumnName].isnull().astype(int)
        return X.drop([self.ColumnName], axis = 1)
    

    
class Log1pTransformer(BaseEstimator, TransformerMixin):
    """
    Receive a DataFrame and a list of ColumnNames.
    Return the same DataFrame with columns added.
    """
    def __init__(self, ColumnName):
        self.ColumnName = ColumnName
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for i in self.ColumnName:
            X[i + '_transf'] = X[i].apply(np.log1p)
        return X



from sklearn import set_config
set_config(transform_output="pandas")
#set_config(transform_output="default")

# Preprocessing for categorical data
Embarked_pipeline01 = Pipeline(steps=[
    ('simple_imputer', SimpleImputer(strategy = 'most_frequent'))
    ,('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# This Pipeline work with the Training, Test and Production data.
preprocessor01 = ColumnTransformer(
    transformers=[
        ('PClass_pipeline', 'passthrough', ['Pclass'])
        ,('Sex_pipeline', OrdinalEncoder(), ['Sex'])
        ,('Age_pipeline', WasMissingTransformer('Age'), ['Age'])
        ,('Age_pipeline01', SimpleImputer(strategy='mean'), ['Age'])
        ,('Log1p_pipeline', Log1pTransformer(['SibSp', 'Parch', 'Fare']), ['SibSp', 'Parch', 'Fare'])
        ,('Embarked_pipeline', WasMissingTransformer('Embarked'), ['Embarked'])
        ,('Embarked_pipeline01', Embarked_pipeline01, ['Embarked'])
    ]
    ,verbose_feature_names_out = False
)

# This Pipeline is just for training because it removes row. That can't be done to the Test and Production data.
preprocessor = Pipeline(steps=[
    ('Fare_pipeline', OutliersTransformer(0.05))
    ,('Preprocessor01_pipeline', preprocessor01)
])

# Data for Training
X_train01_2 = preprocessor.fit_transform(X)

# Data for Test and Production
X_train01_3 = preprocessor01.fit_transform(X)

# Transform
X_test01 = preprocessor01.transform(input_df)
X_test01.rename(columns = {'Sex':'Sex_encoded', 'Age':'Age_fill'}, inplace = True)
X_train_columns = ['Pclass', 'Sex_encoded', 'Age_fill', 'Age_was_missing', 'SibSp',
                   'SibSp_transf', 'Parch', 'Parch_transf', 'Fare', 'Fare_transf',
                   'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_was_missing']
X_test01 = X_test01[X_train_columns].copy()

# Load the model and make predictions with the new data
final_model_reloaded = jbl.load('titanic_model.pkl')
predictions = final_model_reloaded.predict(X_test01)


# Show inputs transformed
st.subheader('Input after pipeline')
st.write(X_test01)


# Show results
st.subheader('Prediction')
st.write(predictions)
st.write('0: dead, 1: survived')
