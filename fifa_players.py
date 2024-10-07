import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import base64
import random
import time

# 1. Function to convert the image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

# 2. Function to add background image
def add_bg_from_local(image_path):
    base64_img = get_base64_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_img}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# List of image paths
image_paths = [
    r"C:\Users\XHIBA\Pictures\lionel-messi-fifa-world-cup-2022-football-trophy-argentina-hd-wallpaper-preview.jpg",
    r"C:\Users\XHIBA\Desktop\HD wallpaper_ Lionel Messi 4K, headshot, portrait, studio shot, beard, adult.jpeg",
    # Add more paths here if needed
]

# Function to choose a random image
def get_random_image_path(image_list):
    return random.choice(image_list)

# Set up session state for image change
if 'last_image_change' not in st.session_state:
    st.session_state.last_image_change = time.time()
    st.session_state.current_image = get_random_image_path(image_paths)

# Check if 5 seconds have passed and update the image
if time.time() - st.session_state.last_image_change > 5:
    st.session_state.current_image = get_random_image_path(image_paths)
    st.session_state.last_image_change = time.time()

# Instead of rerunning, you can reload the background when the image changes
add_bg_from_local(st.session_state.current_image)

# 1. Load the dataset and models
@st.cache_data
def load_data():
    return pd.read_csv('fifa_players.csv')

@st.cache_resource
def load_models():
    try:
        log_reg = joblib.load('logistic_regression_model.pkl')
        rf_clf = joblib.load('random_forest_classifier.pkl')
        gb_clf = joblib.load('gradient_boosting_classifier.pkl')
        rf_reg_team1 = joblib.load('random_forest_regressor_team1.pkl')
        gb_reg_team2 = joblib.load('gradient_boosting_regressor_team2.pkl')
        return log_reg, rf_clf, gb_clf, rf_reg_team1, gb_reg_team2
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

df = load_data()
log_reg, rf_clf, gb_clf, rf_reg_team1, gb_reg_team2 = load_models()

# Ensure models loaded correctly
if not all([log_reg, rf_clf, gb_clf, rf_reg_team1, gb_reg_team2]):
    st.stop()

# 2. Define relevant features for prediction
features = [
    'overall_rating', 'potential', 'long_shots', 'aggression', 'interceptions', 
    'positioning', 'vision', 'penalties', 'composure', 'marking', 
    'standing_tackle', 'sliding_tackle'
]

# 3. Function to calculate aggregate stats for the selected team
def create_team_stats(team):
    if not team:
        return np.zeros(len(features))
    team_stats = df[df['name'].isin(team)][features].mean().values
    if len(team_stats) == 0:
        st.error("No valid players selected from the dataset.")
    return team_stats

# 4. Streamlit app layout
st.title("⚽ Football Match Winning and Goal Prediction")
st.write("Select 11 players for each team to predict which team has a higher chance of winning and how many goals each team might score.")

# 5. Dropdowns for team selection
all_players = df['name'].unique()

def select_players(team_number):
    selected = st.multiselect(
        f'⚽ Select 11 Players for Team {team_number}', 
        all_players, 
        default=None
    )
    if len(selected) > 11:
        st.warning(f"You have selected more than 11 players for Team {team_number}. Only the first 11 will be used.")
        selected = selected[:11]
    elif len(selected) < 11:
        st.warning(f"You have selected fewer than 11 players for Team {team_number}. Predictions require exactly 11 players.")
    return selected

team1 = select_players(1)
team2 = select_players(2)

# Debug team selections
st.write("Selected Team 1:", team1)
st.write("Selected Team 2:", team2)

# 6. Button for prediction
if st.button("🏆 Predict Winning Team and Goals"):
    if len(team1) == 11 and len(team2) == 11:
        # Aggregate stats for both teams
        team1_stats = create_team_stats(team1)
        team2_stats = create_team_stats(team2)

        # Ensure team stats were correctly calculated
        if len(team1_stats) == 0 or len(team2_stats) == 0:
            st.error("Team stats not properly calculated.")
            st.stop()

        # Debug team stats
        st.write("Team 1 Stats:", team1_stats)
        st.write("Team 2 Stats:", team2_stats)
        
        # Calculate feature differences between the two teams
        match_features = np.abs(team1_stats - team2_stats).reshape(1, -1)
        
        # Make predictions for winning probability
        try:
            log_reg_pred = log_reg.predict_proba(match_features)[0][1]
            rf_clf_pred = rf_clf.predict_proba(match_features)[0][1]
            gb_clf_pred = gb_clf.predict_proba(match_features)[0][1]
            final_win_pred = np.mean([log_reg_pred, rf_clf_pred, gb_clf_pred])
        except Exception as e:
            st.error(f"Error during classification prediction: {e}")
            st.stop()

        # Make predictions for goals
        try:
            team1_goals_pred = rf_reg_team1.predict(match_features)[0]
            team2_goals_pred = gb_reg_team2.predict(match_features)[0]
        except Exception as e:
            st.error(f"Error during goal prediction: {e}")
            st.stop()

        # Display results
        st.subheader("📊 Prediction Results")
        st.write(f"**Winning Probability for Team 1:** {final_win_pred * 100:.2f}%")
        st.write(f"**Winning Probability for Team 2:** {(1 - final_win_pred) * 100:.2f}%")
        st.write(f"**Predicted Goals for Team 1:** {team1_goals_pred:.2f}")
        st.write(f"**Predicted Goals for Team 2:** {team2_goals_pred:.2f}")
        
        # Visualization 1: Bar chart comparing team stats
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        index = np.arange(len(features))
        
        ax.bar(index, team1_stats, bar_width, label='Team 1', color='blue')
        ax.bar(index + bar_width, team2_stats, bar_width, label='Team 2', color='red')
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Average Stats')
        ax.set_title('Team 1 vs Team 2: Skill Comparison')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.legend()

        st.pyplot(fig)
        
        # Visualization 2: Pie chart for winning probabilities
        fig, ax = plt.subplots()
        ax.pie(
            [final_win_pred, 1 - final_win_pred], 
            labels=['Team 1', 'Team 2'], 
            autopct='%1.1f%%', 
            colors=['grey', 'pink']
        )
        ax.set_title('Winning Probability')
        st.pyplot(fig)
        
        # Visualization 3: Pie chart for predicted goals
        fig, ax = plt.subplots()
        ax.pie(
            [team1_goals_pred, team2_goals_pred], 
            labels=['Team 1 Goals', 'Team 2 Goals'], 
            autopct='%1.1f%%', 
            colors=['grey', 'yellow']
        )
        ax.set_title('Predicted Goal Distribution')
        st.pyplot(fig)

    else:
        st.error("❗ Please select exactly 11 players for both teams.")
