from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input


app = Flask(__name__)

@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/")
def index():
    # Load your dataset
    # Replace 'your_dataset.csv' with the actual file path
    #dataset_path = 
    global df
    df = pd.read_csv('hack.csv')
    

    # Assuming 'drug_name', 'ic50_effect_size', 'target', and 'target_pathway' are the target variables
    global target_columns
    target_columns = ['drug_name', 'ic50_effect_size', 'target', 'target_pathway']

    # Extract features (X) and target variables (y)
    X = df['feature_name']  # Assuming 'feature_name' is the column containing feature names
    y = df[target_columns]

    # Use LabelEncoder to convert string labels to numeric values
    label_encoder = LabelEncoder()
    y_encoded = y.apply(lambda col: label_encoder.fit_transform(col))

    # One-hot encode the single feature column
    global X_encoded
    X_encoded = pd.get_dummies(X, prefix='feature_name')

    # Standardize the features
    global scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Build a simple neural network model for multi-output regression
    global input_layer
    global drug_name_output
    global ic50_effect_size_output
    global target_pathway_output
    input_layer = Input(shape=(X_scaled.shape[1],))
    hidden1 = Dense(64, activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(hidden1)
    hidden2 = Dense(32, activation='relu')(dropout1)
    drug_name_output = Dense(len(label_encoder.classes_[0]), activation='softmax', name='drug_name')(hidden2)  # Softmax for drug_name
    ic50_effect_size_output = Dense(1, activation='linear', name='ic50_effect_size')(hidden2)
    target_output = Dense(1, activation='linear', name='target')(hidden2)
    target_pathway_output = Dense(len(label_encoder.classes_[3]), activation='softmax', name='target_pathway')(hidden2)  # Softmax for target_pathway

    # Create a model for training
    model = Model(inputs=input_layer, outputs=[drug_name_output, ic50_effect_size_output, target_output, target_pathway_output])

    # Compile the model
    model.compile(loss={'drug_name': 'sparse_categorical_crossentropy', 'ic50_effect_size': 'mean_squared_error', 'target': 'mean_squared_error', 'target_pathway': 'sparse_categorical_crossentropy'},
                optimizer='adam',
                metrics={'drug_name': 'accuracy'})

    # Train the model
    history = model.fit(X_scaled, {'drug_name': y_encoded['drug_name'], 'ic50_effect_size': y_encoded['ic50_effect_size'], 'target': y_encoded['target'], 'target_pathway': y_encoded['target_pathway']},
                        epochs=10, batch_size=32, validation_split=0.2)

    # Save the model
    model.save('your_saved_model.h5')

    return render_template("index.html")

@app.route("/test", methods=["GET", "POST"])
def start():
    if request.method == "POST":
        """ Testing the model """
        specific_feature_name = request.form.get("gene")  # Change this to the desired feature name

        # Load the saved model
        evaluation_model = Model(inputs=input_layer, outputs=[drug_name_output, ic50_effect_size_output, target_pathway_output])
        evaluation_model.compile(optimizer='adam', loss={'drug_name': 'sparse_categorical_crossentropy', 'ic50_effect_size': 'mean_squared_error', 'target_pathway': 'sparse_categorical_crossentropy'})

        # Process user input to match the input format used during training
        user_input = pd.DataFrame([specific_feature_name], columns=['feature_name'])

        # One-hot encode the user input with the same columns as during training
        user_input_encoded = pd.get_dummies(user_input, prefix='feature_name')
        user_input_encoded = user_input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        user_input_scaled = scaler.transform(user_input_encoded)

        # Use the model to predict drug_name, ic50_effect_size, target, and target_pathway based on the user input
        predictions = evaluation_model.predict(user_input_scaled)

        # Display the search results without the predicted results
        search_results = df[df['feature_name'] == specific_feature_name][['feature_name'] + target_columns]
        search_results_sorted = search_results.sort_values(by='ic50_effect_size', ascending=False)

        # Display the top result with the highest ic50_effect_size as a list
        top_result_dict = search_results_sorted.iloc[0].to_dict()
        top_result_list = [(key, value) for key, value in top_result_dict.items()]
        
        global search_results_tuples
        search_results_tuples = list(search_results_sorted.to_records(index=False))

        # Display the search results as a list of tuples
        #print("\nSearch Results:")
        #print(search_results_tuples)


        #print("\nTop Result with Highest ic50_effect_size:")
        #print(top_result_list)

        return render_template("index.html", test=1, ans = top_result_list )
    
    else:
        return render_template("index.html")

@app.route("/show", methods=["GET","POST"])
def show():
    """ For showing all the results """
    
    #if request.method == "POST":
        
    return render_template("show.html", test=1, anss = search_results_tuples)

