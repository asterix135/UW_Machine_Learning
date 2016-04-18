"""
Code for first programming assignment, module 5
"""

import graphviz
import json
import numpy as np
import os
import pandas as pd
import sklearn
from sklearn import tree


# Load lending club dataset
os.chdir('..')
loans = pd.read_csv('data/lending-club-data.csv')

# create column to indicate safe/unsafe loan coded as +1/-1
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)

# Extract features and target value from full data frame
features = ['grade',  # grade of the loan
            'sub_grade',  # sub-grade of the loan
            'short_emp',  # one year or less of employment
            'emp_length_num',  # number of years of employment
            'home_ownership',  # home_ownership status: own, mortgage or rent
            'dti',  # debt to income ratio
            'purpose',  # the purpose of the loan
            'term',  # the term of the loan
            'last_delinq_none',  # has borrower had a delinquincy
            'last_major_derog_none',  # has borrower had 90 day or worse rating
            'revol_util',  # percent of available credit being used
            'total_rec_late_fee',  # total late fees received to day
            ]
target = 'safe_loans'  # prediction target (y) (+1 means safe, -1 is risky)
loans = loans[features + [target]]

# Unpack categorical variables
loans_data = pd.get_dummies(loans)

# Split test/train with provided indices
with open('data/module-5-assignment-1-validation-idx.json') as f:
    validation_idx = json.load(f)
with open('data/module-5-assignment-1-train-idx.json') as f:
    train_idx = json.load(f)
train_data = loans_data.iloc[train_idx]
validation_data = loans_data.iloc[validation_idx]

# Build decision tree classifier

# First convert to numpy arrays
train_matrix = train_data.drop('safe_loans', axis=1).as_matrix()
validation_matrix = validation_data.drop('safe_loans', axis=1).as_matrix()
train_target = train_data['safe_loans'].as_matrix()
validation_target = validation_data['safe_loans'].as_matrix()

# Train classifier
decision_tree_model = tree.DecisionTreeClassifier(max_depth=6)
decision_tree_model.fit(train_matrix, train_target)

small_model = tree.DecisionTreeClassifier(max_depth=2)
small_model.fit(train_matrix, train_target)

# Visualize small model
tree.export_graphviz(small_model, out_file='Week3_Trees/small_model.dot')
with open('Week3_Trees/small_model.dot', 'r') as f:
    text = f.read()
dot = graphviz.Source(text)
dot.format = 'png'
