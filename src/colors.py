import os
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_team_colors():
    team_colors = {}
    # Use __file__ to dynamically determine the path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'data', 'team_colors.txt')
    try:
        with open(file_path, 'r') as file:
            for line in file:
                team, color = line.strip().split(': ')
                team_colors[team.upper()] = color
    except FileNotFoundError:
        st.error(f"The team colors file was not found at {file_path}")
    except ValueError:
        st.error("There was an error parsing the team colors file. Please check its format.")
    return team_colors

def get_team_color(team, team_colors=None):
    # Check if team_colors dictionary has been loaded, otherwise load it
    if team_colors is None:
        team_colors = load_team_colors()
    
    # Return the color for the team if it exists, otherwise return 'grey'
    return team_colors.get(str(team).upper(), 'grey')