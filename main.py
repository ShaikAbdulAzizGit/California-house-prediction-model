import numpy  as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os,joblib


MODEL_FILE='model.pkl'
PIPELINE_FILE='pipeline.pkl'

def build_pipeline(num_attributes,cat_attributes):
    # Creating Pipelines

    # Pipeline for numerical columns
    num_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
    ])

    # Pipeline for numerical columns
    cat_pipeline=Pipeline([
    ('onehotencoder',OneHotEncoder(handle_unknown='ignore'))
    ])

    # Constructing the full pipeline using ColumnTransformer

    full_pipeline=ColumnTransformer([
        ('num',num_pipeline,num_attributes),
        ('cat',cat_pipeline,cat_attributes)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Loading the data 
    housing=pd.read_csv('housing.csv')
    # Creating a staratified test set based on income category
    housing['income_cat']=pd.cut(housing['median_income'],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for train_index,test_index in split.split(housing,housing['income_cat']):
        housing.loc[test_index].drop('income_cat',axis=1).to_csv('input_data.csv',index=False)
        housing=housing.loc[train_index].drop('income_cat',axis=1)
    # Seperating features and labels
    housing_labels=housing['median_house_value'].copy()
    housing_features=housing.drop('median_house_value',axis=1)

    # Seperating numerical and categorical columns
    cat_attributes=['ocean_proximity']
    num_attributes=housing_features.drop('ocean_proximity',axis=1).columns.tolist()

    pipeline=build_pipeline(num_attributes,cat_attributes)
    housing_prepared=pipeline.fit_transform(housing_features)

    # Training model 
    model=RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Your model is trained !.. Congrats ...")
else:
    pipeline=joblib.load(PIPELINE_FILE)
    model=joblib.load(MODEL_FILE)
    input_data=pd.read_csv('input_data.csv')
    transformed_input=pipeline.fit_transform(input_data)
    # transformed_input.to_csv('trans.csv')
    predictions=model.predict(transformed_input)
    input_data['median_house_value_predictions']=predictions
    input_data.to_csv('output.csv',index=False)
    print("Inference is completed . result is saved to output.csv ")