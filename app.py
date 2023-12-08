import numpy as np
import pandas as pd


from flask import Flask, request, jsonify, render_template
from matplotlib import pyplot as plt
import base64
import pickle

import os

import seaborn as sns


app = Flask(__name__)
model = pickle.load(open('model_knn.pkl', 'rb'))

import matplotlib
matplotlib.use('Agg')

dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:,[1, 4, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)

#partie description


def generate_histogram(dataset, save_path='static/histograms'):
    os.makedirs(save_path, exist_ok=True)
    col = dataset.columns[:8]
    plt.figure(figsize=(20, 15))
    length = len(col)

    for i, j in zip(col, range(length)):
        plt.subplot(length // 2, 3, j + 1)
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        dataset[i].hist(bins=20)
        plt.title(i)

    plt.savefig(os.path.join(save_path, 'histogram.png'))
    plt.close()
def generate_heatmap(dataset, save_path='static/heatmaps'):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(dataset.corr(), annot=True)
    heatmap.figure.savefig(os.path.join(save_path, 'heatmap.png'))

def generate_boxplots(dataset, save_path='static/boxplots'):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(12, 12))
    for i, col in enumerate(['Glucose', 'BloodPressure', 'BMI', 'Age']):
        plt.subplot(3, 3, i + 1)
        sns.boxplot(x=col, data=dataset)
    plt.savefig(os.path.join(save_path, 'boxplots.png'))

def generate_pairplot(dataset, save_path='static/pairplot'):
    os.makedirs(save_path, exist_ok=True)
    pairplot = sns.pairplot(dataset, hue='Outcome', diag_kind='kde')
    pairplot.savefig(os.path.join(save_path, 'pairplot.png'))



@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features) )

    if prediction == 1:
        pred = "You have a high risk of developing Diabetes!"
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('index2.html', prediction_text='{}'.format(output))

@app.route('/description')
def description():


    # Appelez la fonction pour générer l'histogramme
    generate_histogram(dataset)


  # Appelez la fonction pour générer la heatmap
    generate_heatmap(dataset)


    # Appelez la fonction pour générer les boxplots
    generate_boxplots(dataset)

    # Appelez la fonction pour générer le pairplot
    generate_pairplot(dataset)

    return render_template('description.html')


if __name__ == "__main__":
    app.run(debug=True)