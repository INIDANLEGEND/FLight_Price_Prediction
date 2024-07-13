import pickle

import joblib

import numpy as np

import pandas as pd

import xgboost as xgb

import streamlit as st
import datetime as dt

import sklearn
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin


class CustomColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, predefined_columns, estimator, scoring, threshold):
        self.predefined_columns = predefined_columns
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.selector = None
        self.remaining_columns = None

    def fit(self, X, y=None):
        # Determine columns that need to be selected automatically
        self.remaining_columns = [col for col in X.columns if col not in self.predefined_columns]
        # Fit the selector to these remaining columns
        self.selector = SelectBySingleFeaturePerformance(
            estimator=self.estimator,
            scoring=self.scoring,
            threshold=self.threshold
        )
        self.selector.fit(X[self.remaining_columns], y)
        return self

    def transform(self, X):
        # Apply the selector to the remaining columns
        selected_indices = self.selector.get_support(indices=True)
        selected_remaining_columns = [self.remaining_columns[i] for i in selected_indices]
        
        # Combine predefined columns with selected columns
        selected_columns = self.predefined_columns + selected_remaining_columns
        return X[selected_columns]



def is_north(x):
    north_cities = ["Delhi", "Kolkata", "Mumbai", "New Delhi"]
    columns = x.columns
    return(
        x
        .assign(**{
            f"{col}_is_north" : x.loc[:,col].isin(north_cities).astype(int)
            for col in columns
        })
        .drop(columns=columns)
    )

def part_of_the_day(x, morning=4, noon=12, eve=16, night=20):
    columns = x.columns.to_list()
    x_temp = x.assign(**{
        col: pd.to_datetime(x.loc[:,col]).dt.hour  # for each columns extract hour value for all the rows
        for col in columns
    })

    return (
        x_temp
        .assign(**{
            f"{col}_part_of_the_day" : np.select(
                [x_temp.loc[:, col].between(morning, noon, inclusive="left"),
				x_temp.loc[:, col].between(noon, eve, inclusive="left"),
			    x_temp.loc[:, col].between(eve, night, inclusive="left")],
                ["morning","afternoon","evening"],
                default="night"
            )
            for col in columns
        })
        .drop(columns=columns)
    )

class RBFPercentileSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, percentile=[0.25, 0.5, 0.75],gamma=0.1):
        self.variables = variables
        self.percentile = percentile
        self.gamma = gamma 

    def fit(self,x,y=None):
        if not self.variables:
            self.variables = x.select_dtypes(include="number").columns.to_list()

        self.reference_values_ = {
            col:(
                x
                .loc[:, col]
                .quantile(self.percentile)
                .values
                .reshape(-1,1)
            )
            for col in self.variables
        }

        return self
    
    def transform(self,x):
        objects=[]
        for col in self.variables:
            columns = [f"{col}_rbf_{int(percentile*100)}" for percentile in self.percentile]
            obj = pd.DataFrame(
                data = rbf_kernel(x.loc[: ,[col]],Y=self.reference_values_[col],gamma = self.gamma), 
                columns=columns
            )
            # print(obj)
            objects.append(obj)

        return pd.concat(objects,axis=1)
    
def duration_category(x, short=180, med=500):
    return (
        x
        .assign(duration_cat=np.select([x.duration.between(0,short),
                                        x.duration.between(short,med,inclusive = "left")],
                                        ["short", "medium"],
                                        default = "long"))
        .drop(columns="duration")
    )

def is_over(x, value=1000):
    return (
        x
        .assign(**{
            f"duration_over_{value}" : x.duration.ge(value).astype(int)
        })
        .drop(columns="duration")
    )

def is_direct(x):
    return x.assign(is_direct_flight = x.total_stops.eq(0).astype(int))

def have_info(x):
    return x.assign(additional_info = x.additional_info.str.lower()!="no info").astype(int)

# functions used in the preprocessing 

sklearn.set_config(transform_output="pandas")

train = pd.read_csv("data/train_.csv")

st.set_page_config(
    page_title = "Flights Prices Prediction",
    page_icon = "✈️",
    layout = "wide"
)

st.title("Flights Prices Prediction")  # title of the app

# taking input
airline = st.selectbox(
    "Airline:",
    options = train.airline.unique()
)

doj = st.date_input("date of Journey:")

source = st.selectbox(
    "Source",
    options = train.source.unique()
)

destination = st.selectbox(
    "Destination",
    options = train.destination.unique()
)

dep_time = st.time_input("Departure Time:")

duration = st.number_input(
    "Duration (mins):",
    step = 15,
    min_value = 45
)

total_stops = st.number_input(
    "Total Stops:",
    step = 1,
    min_value = 0,
    max_value = 4
)

additional_info = st.selectbox(
    "Additional Info:",
    options = train.additional_info.unique()
)

departure_datetime = dt.datetime.combine(doj, dep_time)

# Calculate arrival time from departure time and duration
arrival_datetime = departure_datetime + dt.timedelta(minutes=duration)
arrival_time = arrival_datetime.time()

# creating a dataframe
x_new = pd.DataFrame(dict(
    airline = [airline],
    date_of_journey = [doj],
    source = [source],
    destination= [destination],
    dep_time = [dep_time],
    arrival_time = [arrival_time],
    duration = [duration],
    total_stops = [total_stops],
    additional_info = [additional_info]
)).astype({
    col : "str"
    for col in ["date_of_journey","dep_time","arrival_time"]
})

if st.button("Predict"):
    saved_preprocessor = joblib.load("final_preprocessor.joblib")
    x_new_pre = saved_preprocessor.transform(x_new)

    with open("final_xgboost-model.pkl","rb") as f:
        model = pickle.load(f)
    pred = model.predict(x_new_pre)
    print(pred)
    st.info(f"The predicted price is {pred.astype(int)} INR")
