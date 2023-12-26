# Import necessary libraries and modules
import pandas as pd
import streamlit as st
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from st_aggrid import AgGrid

# Define a mapping for model types
TYPE = {
    'rf': 'Random Forest',
    'dt': 'Decision Tree'
}

# Function to read data based on options (type and dataset)
def read_data(options):
    if options['type'] == 'url':
        if options['dataset'] == 'default':
            return pd.read_csv('Dataset/URL/training_final.csv')
        return pd.read_csv('Dataset/URL/custom_training.csv')
    elif options['type'] == 'file_path':
        if options['dataset'] == 'default':
            return pd.read_csv('Dataset/File Path/training_final.csv')
        return pd.read_csv('Dataset/File Path/custom_training.csv')
    elif options['type'] == 'event_chain':
        if options['dataset'] == 'default':
            return pd.read_csv('Dataset/Event/training_final.csv')
        return pd.read_csv('Dataset/Event/custom_training.csv')

# Function to calculate classification metrics
def score(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1_score': f1}

# Function to get a machine learning model based on the name and configuration
def get_model(name, config):
    if name == 'rf':
        model = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            min_samples_leaf=config['min_samples_leaf'],
            max_features=config['max_features'],
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
    elif name == 'dt':
        model = DecisionTreeClassifier(
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            min_samples_leaf=config['min_samples_leaf'],
            max_features=config['max_features'],
            random_state=42
        )

    return model

# Function to save the trained model and update the configuration file
def save(model, options):
    with open('config.json') as json_file:
        file = json.load(json_file)
    if options['type'] == 'url':
        filename = f"Model/URL/{options['name']}.sav"
        file['model']['URL']['custom'][options['name']] = {
            'value': filename, 'type': TYPE[options['model']]}
    elif options['type'] == 'file_path':
        filename = f"Model/File Path/{options['name']}.sav"
        file['model']['File Path']['custom'][options['name']] = {
            'value': filename, 'type': TYPE[options['model']]}
    elif options['type'] == 'event_chain':
        filename = f"Model/Event/{options['name']}.sav"
        file['model']['Event']['custom'][options['name']] = {
            'value': filename, 'type': TYPE[options['model']]}
    pickle.dump(model, open(filename, 'wb'))

    with open('config.json', 'w') as json_file:
        json.dump(file, json_file)

# Function to train a machine learning model based on options
def train_model(options):
    data = read_data(options)
    train = data.drop(columns=['Label'])
    if options['type'] == 'url':
        train = train.loc[:, options['features']]
    label = data['Label']

    model = get_model(options['model'], options['config'])

    if options['mode'] == 'testing':
        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            train, label, test_size=0.1, stratify=label, random_state=42)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        # Calculate and display classification metrics
        scores = score(y_test, prediction)
        score_df = pd.DataFrame([scores])
        AgGrid(score_df)
    else:
        model.fit(train, label)

    if options['save'] == 1:
        save(model, options)

# Main block to execute the script
if __name__ == '__main__':
    # Define default options for testing
    options = {
        'mode': 'testing',
        'save': 0,
        'name': 'rf'
    }
    # Load model configuration from the config file
    with open('config.json') as json_file:
        config = json.load(json_file)
    options['model'] = config['model config']['random forest']['name']
    options['config'] = config['model config']['random forest']['config']
    # Train the model based on the specified options
    train_model(options)
