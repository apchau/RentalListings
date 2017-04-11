import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn import model_selection, preprocessing, ensemble
from scipy import sparse
import os
import sys
import operator

def get_file(filename):
    df = pd.read_json(filename, convert_dates = ['created'])
    return df

def numerical_engineering(df):
    numerical_features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price']
    return numerical_features

def feature_engineering(df):
    df['num_photos'] = df['photos'].apply(len)
    df['num_features'] = df['features'].apply(len)
    df['num_description_words'] = df['description'].apply(lambda x: len(x.split(" ")))
    return df

def encode_labels(df_train, df_test, label):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train[label].values) + list(df_test[label].values))
    df_train = lbl.transform(list(df_train[label].values))
    df_test = lbl.transform(list(df_test[label].values))
    return df_train, df_test

def vectorize_column(df_train, df_test, label):
    df_train[label] = df_train[label].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))
    df_test[label] = df_test[label].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))
    count_vect = CountVectorizer(stop_words = 'english', max_features = 200)
    sparse_train = count_vect.fit_transform(df_train[label])
    sparse_test = count_vect.transform(df_test[label])
    return sparse_train, sparse_test

def dates(df):
    df['created_year'] = df['created'].dt.year
    df['created_month'] = df['created'].dt.month
    df['created_day'] = df['created'].dt.day
    df['created_hour'] = df['created'].dt.hour
    return df

if __name__ == '__main__':
    df_train = get_file('../data/train.json')
    df_test = get_file('../data/test.json')
    df_train = feature_engineering(df_train)
    df_test = feature_engineering(df_test)
    df_train = dates(df_train)
    df_test = dates(df_test)
    df_train, df_test = encode_labels(df_train, df_test, 'display_address')
    df_train, df_test = encode_labels(df_train, df_test, 'manager_id')
    df_train, df_test = encode_labels(df_train, df_test, 'building_id')
    df_train, df_test = encode_labels(df_train, df_test, 'street_address')
    sparse_train, sparse_test = vectorize_column(df_train, df_test, 'features')
    final_features = ['num_photos', 'num_features', 'num_description_words', 'created_year', 'created_month', 'created_day', 'created_hour', 'display_address', 'manager_id', 'building_id', 'street_address']
    final_train = sparse.hstack([df_train[final_features], sparse_train]).tocsr()
    final_test = sparse.hstack([df_test[final_features], sparse_test]).tocsr()
