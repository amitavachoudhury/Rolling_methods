# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 22:40:00 2022

@author: amitava
"""

from flask import Flask, render_template, request
import pickle
import numpy as np

with open(f'model/hot_roll_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__, template_folder='templates')


@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def home():
    c =  request.form['c']
    ce = request.form['ce']
    s =  request.form['s']
    p =  request.form['p']
    YieldStrength =  request.form['ys']
    Elongation =  request.form['el']
   
    arr = np.array([[c, ce, s, p, YieldStrength, Elongation]])
    prediction = model.predict(arr)
    return render_template('after.html', data=prediction)


if __name__ == "__main__":
    app.run()