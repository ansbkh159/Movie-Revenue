from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from models.feature_scaling import preprocess_data, prepare_features
import joblib
import os

def run_model():
    model_path = "best_rf_model.pkl"

    if os.path.exists(model_path):
        return joblib.load(model_path)

    df = pd.read_csv("revised datasets/output.csv")
    df.fillna(df.median(numeric_only=True), inplace=True)
    X, y = prepare_features(df)

    param_grid = {
        "n_estimators": [100],
        "max_depth": [10],
        "min_samples_split": [5],
        "min_samples_leaf": [2],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )
    grid_search.fit(X, y)
    
    best_params = grid_search.best_params_
    best_model = RandomForestRegressor(random_state=42, **best_params)
    best_model.fit(X, y)

    joblib.dump(best_model, model_path)

    return best_model

def predict_gross(input_data, best_model):
    processed_data = preprocess_data(pd.DataFrame([input_data]))
    expected_features = best_model.feature_names_in_
    print(expected_features)
    for feature in expected_features:
        if feature not in processed_data.columns:
            processed_data[feature] = 0

    processed_data = processed_data[expected_features]
    log_prediction = best_model.predict(processed_data)
    prediction = np.exp(log_prediction) - 1
    return prediction[0]

def predict_gross_range(gross):
    if gross <= 10_000_000:
        return "Low Revenue"
    elif gross <= 40_000_000:
        return "Medium-Low Revenue"
    elif gross <= 70_000_000:
        return "Medium Revenue"
    elif gross <= 120_000_000:
        return "Medium-High Revenue"
    elif gross <= 200_000_000:
        return "High Revenue"
    else:
        return "Ultra High Revenue"

st.markdown(
    """
    <h1 style='text-align: center; color: cyan;'>Movie Revenue Prediction</h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h2 style='text-align: center; color: white;'>Enter Movie Details</h2>
    """,
    unsafe_allow_html=True,
)

with st.form(key="movie_form"):
    col1, col2 = st.columns(2)

    with col1:
        released = st.text_input("Release Date")
        writer = st.text_input("Writer")
        rating = st.selectbox("MPAA Rating", ["G", "PG", "PG-13", "R", "NC-17"])
        name = st.text_input("Movie Name")
        genre = st.text_input("Genre")
        director = st.text_input("Director")
        star = st.text_input("Leading Star")

    with col2:
        country = st.text_input("Country of Production")
        company = st.text_input("Production Company")
        runtime = st.number_input("Runtime (minutes)", min_value=0.0)
        score = st.number_input("IMDb Score", min_value=0.0, max_value=10.0)
        budget = st.number_input("Budget", min_value=0.0)
        year = st.number_input("Shooting Year", min_value=1900, max_value=2100)
        votes = st.number_input("Initial Votes", min_value=0)

    submit_button = st.form_submit_button(label="Predict Revenue")

if submit_button:
    input_data = {
        "released": released,
        "writer": writer,
        "rating": rating,
        "name": name,
        "genre": genre,
        "director": director,
        "star": star,
        "country": country,
        "company": company,
        "runtime": runtime,
        "score": score,
        "budget": budget,
        "year": year,
        "votes": votes,
    }

    best_model = run_model()
    
    with st.spinner("Predicting..."):
        predicted_gross = predict_gross(input_data, best_model)
        predicted_gross_range = predict_gross_range(predicted_gross)

    st.markdown("## Prediction Result")
    st.success(f'Predicted Revenue for "{name}": ${predicted_gross:,.2f}')
    st.success(f"Predicted Revenue Range: {predicted_gross_range}")
