import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from models.feature_scaling import preprocess_data, prepare_features
from colorama import init, Fore, Style
import joblib

def begin_cli():
    init()
    os.system("cls" if os.name == "nt" else "clear")
    print("Movie Revenue Prediction System - Will your next movie succeed?")

def get_user_input():
    print(f"\n{Fore.YELLOW}Enter the movie details to predict revenue:{Style.RESET_ALL}\n")
    return {
        "released": input("Release Date: "),
        "writer": input("Writer: "),
        "rating": input("MPAA Rating: "),
        "name": input("Movie Name: "),
        "genre": input("Genre: "),
        "director": input("Director: "),
        "star": input("Lead Star: "),
        "country": input("Country of Production: "),
        "company": input("Production Company: "),
        "runtime": float(input("Runtime (minutes): ")),
        "score": float(input("Initial IMDb Score (0-10): ")),
        "budget": float(input("Budget ($): ")),
        "year": int(input("Shooting Year: ")),
        "votes": float(input("Initial Votes: ")),
    }

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
    
    best_model = RandomForestRegressor(random_state=42, **grid_search.best_params_)
    best_model.fit(X, y)

    joblib.dump(best_model, model_path)
    return best_model

def predict_gross(input_data, best_model):
    processed_data = preprocess_data(pd.DataFrame([input_data]))
    expected_features = best_model.feature_names_in_
    
    for feature in expected_features:
        if feature not in processed_data.columns:
            processed_data[feature] = 0

    processed_data = processed_data[expected_features]
    log_prediction = best_model.predict(processed_data)
    return np.exp(log_prediction) - 1

def predict_gross_range(gross):
    if gross <= 10_000_000:
        return "Low Revenue (<= $10M)"
    elif gross <= 40_000_000:
        return "Medium-Low Revenue ($10M - $40M)"
    elif gross <= 70_000_000:
        return "Medium Revenue ($40M - $70M)"
    elif gross <= 120_000_000:
        return "Medium-High Revenue ($70M - $120M)"
    elif gross <= 200_000_000:
        return "High Revenue ($120M - $200M)"
    else:
        return "Ultra High Revenue (>= $200M)"

if __name__ == "__main__":
    begin_cli()
    best_model = run_model()

    while True:
        input_data = get_user_input()
        predicted_gross = predict_gross(input_data, best_model)[0]
        predicted_gross_range = predict_gross_range(predicted_gross)

        print(f'\n{Fore.GREEN}Predicted Revenue{Style.RESET_ALL} for "{Fore.CYAN}{input_data["name"]}{Style.RESET_ALL}": ${predicted_gross:,.2f}')
        print(f"{Fore.GREEN}Predicted Revenue Range{Style.RESET_ALL}: {predicted_gross_range}\n")

        if input(f"{Fore.YELLOW}Predict another movie? (yes/no): {Style.RESET_ALL}").lower() != "yes":
            break
