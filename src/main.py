#base imports
import os
import pandas as pd
import numpy as np
import random
import streamlit as st

st.set_page_config(page_title="Brock's Baseball Stats Explorer", page_icon="⚾", layout="wide")

#plotting and stats
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
from scipy import stats
#baseball functions
from load_data import load_and_filter_data, load_and_prepare_data, get_team_color
from home_screen import display_home_screen
from colors import load_team_colors, get_team_color
from individual_player_stats import individual_player_view
from compare_player_stats import compare_players_view
from race import process_data_for_race, create_race_plot, race_chart_view
from histogram import plot_league_wide_stat, create_hover_text, league_wide_stats_view
from similarity import calculate_similarity_scores, player_similarity_view
from make_war import custom_war_generator
from milestone_tracker import milestone_tracker
from goat import how_is_he_the_goat
from outliers import anomaly_tracker, identify_statistical_outliers, plot_stat_distributions
from era_adjustment import adjust_stat, calculate_era_factors, era_adjustment_tool
from basic_statistics_mode import basic_statistics_mode
from supervised_learning_mode import *
from unsupervised_learning_mode import *
from bangbang import generate_astros_cheating_fact, display_astros_cheating_fact


def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose the mode",
                                ["Home", "Individual Player", "Compare Players", "Historical Histogram", 
                                 "Milestone Tracker", "Find Outliers", "Career Stat Race", "Player Similarity", 
                                 "Custom WAR Generator", "How is he the GOAT?", "Era Adjustment Tool", 
                                 #"Basic Statistics", 
                                # "Supervised Learning", "Unsupervised Learning"
                                ])
    
    if app_mode == "Home":
        display_home_screen()
    elif app_mode == "Individual Player":
        individual_player_view()
    elif app_mode == "Compare Players":
        compare_players_view()
    elif app_mode == "Historical Histogram":
        league_wide_stats_view()
    elif app_mode == "Career Stat Race":
        race_chart_view()
    elif app_mode == "Player Similarity":
        player_similarity_view()
    elif app_mode == "Custom WAR Generator":
        custom_war_generator()
    elif app_mode == "Milestone Tracker":
        milestone_tracker()
    elif app_mode == "Find Outliers":
        anomaly_tracker()
    elif app_mode == "Era Adjustment Tool":
        era_adjustment_tool()
    elif app_mode == "How is he the GOAT?":
        how_is_he_the_goat()
    # elif app_mode == "Basic Statistics":
    #     basic_statistics_mode()
    #elif app_mode == "Supervised Learning":
     #   supervised_learning_mode()
    #elif app_mode == "Unsupervised Learning":
    #    unsupervised_learning_mode()

    # Display the Astros cheating fact at the bottom of each page
    display_astros_cheating_fact()

if __name__ == "__main__":
    main()
