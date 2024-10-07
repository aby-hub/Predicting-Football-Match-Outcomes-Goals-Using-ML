# Football Player Statistics Analysis using Machine Learning

This project focuses on analyzing football player statistics and predicting match outcomes or player comparisons using machine learning models. The notebook contains code to load, explore, and describe football player data, followed by building predictive models to compare player performance or predict match results based on key metrics.

## Features

- **Player Data Analysis**: Provides detailed statistical descriptions of football players, including metrics like passing accuracy, shooting ability, defense skills, and more.
- **Machine Learning Models**: Implements multiple machine learning models, such as Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and K-Nearest Neighbors (KNN), to predict outcomes like player comparisons or match results.
- **Prediction Engine**: Allows users to input data on two football players or teams, providing an output that predicts which one is better based on specific statistical parameters.
- **Team Analysis**: Aggregates the individual player stats to compute team-level metrics and predicts match results between two teams based on selected players.
- **Data Visualization**: Includes visualizations like bar charts and pie charts to provide a clearer understanding of player statistics and model predictions.
- **Streamlit Web App**: A web interface is built using Streamlit, where users can interactively select players or teams, and the app displays winning probabilities and other visual outputs.

## Requirements

Make sure to install the following packages before running the project:

```bash
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install xgboost
pip install streamlit
