import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Union, Optional, Tuple

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the data directory
data_dir = os.path.join(current_dir, 'data')

@st.cache_data
def load_and_filter_data(data_type: str, player_names_or_ids: Optional[List[Union[str, int]]] = None) -> pd.DataFrame:
    data_subdir = 'hitter_data' if data_type == "Hitter" else 'pitcher_data'
    data_path = os.path.join(data_dir, data_subdir)
    all_data = []
    for filename in os.listdir(data_path):
        if filename.endswith('.csv'):
            year = filename.split('_')[-1].split('.')[0]
            file_path = os.path.join(data_path, filename)
            data = pd.read_csv(file_path)
            data['year'] = int(year)
            data['player_type'] = data_type.lower()
            all_data.append(data)
    
    full_data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    if player_names_or_ids:
        filtered_data = full_data[
            full_data['Name'].isin(player_names_or_ids) | 
            full_data['IDfg'].astype(str).isin(map(str, player_names_or_ids))
        ]
        return filtered_data if not filtered_data.empty else pd.DataFrame()
    else:
        return full_data

def load_and_prepare_data(data_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_df = load_and_filter_data(data_type)
    
    # Replace NaN values in IDfg with a placeholder (e.g., -1)
    data_df['IDfg'] = data_df['IDfg'].fillna(-1)
    
    # Convert IDfg to integer
    data_df['IDfg'] = data_df['IDfg'].astype(int)
    
    # Calculate first and last year for each player
    player_years = data_df.groupby('IDfg').agg({
        'Name': 'first',
        'year': ['min', 'max']
    }).reset_index()
    player_years.columns = ['IDfg', 'Name', 'FirstYear', 'LastYear']
    
    # Create unique labels for each player
    player_years['Label'] = player_years.apply(lambda row: f"{row['Name']} ({row['FirstYear']}-{row['LastYear']})", axis=1)
    
    return data_df, player_years

def load_hof_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hof_path = os.path.join(current_dir, 'data', 'HOF_voting_results', 'HallOfFame_all_years.csv')
    print("Attempting to load HOF data from:", hof_path)
    try:
        hof_data = pd.read_csv(hof_path)
        return hof_data
    except FileNotFoundError:
        print("File not found:", hof_path)
        return None
    except Exception as e:
        print(f"Error loading HOF data: {e}")
        return None

@st.cache_data
def load_team_colors():
    team_colors_path = os.path.join(data_dir, 'team_colors.txt')
    team_colors = {}
    try:
        with open(team_colors_path, 'r') as file:
            for line in file:
                team, color = line.strip().split(': ')
                team_colors[team.upper()] = color
        return team_colors
    except FileNotFoundError:
        print(f"Team colors file not found: {team_colors_path}")
        return {}
    except Exception as e:
        print(f"Error loading team colors: {e}")
        return {}

def get_team_color(team, team_colors):
    return team_colors.get(str(team).upper(), 'grey')