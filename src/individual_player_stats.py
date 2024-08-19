import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from load_data import load_and_prepare_data, load_team_colors
from typing import List, Dict

def calculate_career_stats(player_data: pd.DataFrame, player_type: str) -> pd.DataFrame:
    # Define stat order for pitchers and hitters
    pitcher_stat_order = ['WAR', 'W', 'L', 'ERA', 'G', 'GS', 'IP', 'AVG', 'WHIP', 'FIP', 'CG', 'ShO', 'SV', 'K/9', 'BB/9', 'K/BB', 'H/9', 'HR/9', 'BS', 'TBF', 'H', 'R', 'HR', 'SO', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'GB%', 'FB%', 'LD%', 'IFH', 'IFFB', 'Balls', 'Strikes', 'Pitches']
    hitter_stat_order = ['WAR', 'G', 'AB', 'PA', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'AVG', 'OBP', 'SLG', 'OPS', 'BB', 'SO', 'HBP', 'SF', 'SH', 'GDP', 'SB', 'CS', 'GB', 'FB', 'BB%', 'K%', 'BB/K', 'ISO']
    
    # Define rate stats (stats that should be averaged instead of summed)
    rate_stats = ['AVG', 'OBP', 'SLG', 'OPS', 'BB%', 'K%', 'ISO', 'BABIP', 'wRC+', 'wOBA', 'ERA', 'WHIP', 'K/9', 'BB/9', 'H/9', 'HR/9', 'K/BB', 'FIP', 'xFIP', 'GB%', 'FB%', 'LD%']

    # Basic career stats
    career_stats = player_data.groupby('IDfg').agg({
        'Name': 'first',
        'G': 'sum',
        'year': ['min', 'max']
    })
    career_stats.columns = ['Name', 'Games', 'First Year', 'Last Year']

    # Function to calculate weighted average
    def weighted_average(group, stat, weight):
        return (group[stat] * group[weight]).sum() / group[weight].sum()

    # Calculate other career stats based on player type
    stat_order = pitcher_stat_order if player_type == "Pitcher" else hitter_stat_order
    weight_column = 'IP' if player_type == "Pitcher" else 'AB'

    for stat in stat_order:
        if stat in player_data.columns:
            if stat in rate_stats:
                career_stats[stat] = player_data.groupby('IDfg').apply(lambda x: weighted_average(x, stat, weight_column))
            else:
                career_stats[stat] = player_data.groupby('IDfg')[stat].sum()

    # Reorder columns based on stat_order
    ordered_columns = ['Name', 'Games', 'First Year', 'Last Year'] + [col for col in stat_order if col in career_stats.columns]
    career_stats = career_stats[ordered_columns]

    return career_stats

def display_player_stats(player_data: pd.DataFrame, player_type: str, full_data: pd.DataFrame):
    team_colors = load_team_colors()
    
    st.header("Career Summary")
    for idfg in player_data['IDfg'].unique():
        player_career = player_data[player_data['IDfg'] == idfg]
        player_name = player_career['Name'].iloc[0]
        st.write(f"{player_name}: {player_career['year'].min()} - {player_career['year'].max()} ({len(player_career)} seasons)")

    st.header("Career Stats")
    career_stats = calculate_career_stats(player_data, player_type)
    st.dataframe(career_stats)

    st.header("Yearly Stats")
    yearly_stats = player_data.sort_values(['IDfg', 'year'])
    st.dataframe(yearly_stats)

    st.header("Stat Explorer")
    display_stat_explorer(player_data, player_type, team_colors, full_data)


def calculate_league_averages(data: pd.DataFrame, stat: str, player_type: str) -> Dict[int, float]:
    league_averages = {}
    min_ab = 200 if player_type == "Hitter" else 50
    ab_column = 'AB' if player_type == "Hitter" else 'IP'
    
    for year in data['year'].unique():
        year_data = data[data['year'] == year]
        qualified = year_data[year_data[ab_column] >= min_ab]
        if not qualified.empty:
            if stat in ['AVG', 'OBP', 'SLG', 'OPS', 'ERA', 'WHIP']:
                # For rate stats, we need to weight by AB or IP
                weight = 'AB' if player_type == "Hitter" else 'IP'
                league_averages[year] = (qualified[stat] * qualified[weight]).sum() / qualified[weight].sum()
            else:
                # For counting stats, we can just take the mean
                league_averages[year] = qualified[stat].mean()
    
    return league_averages

def display_stat_explorer(player_data: pd.DataFrame, player_type: str, team_colors: dict, full_data: pd.DataFrame):
    numeric_columns = player_data.select_dtypes(include=['int64', 'float64']).columns
    stat_options = [col for col in numeric_columns if col not in ['year', 'IDfg', 'season']]
    
    default_stat = 'WAR' if 'WAR' in stat_options else stat_options[0]
    default_index = stat_options.index(default_stat)
    
    selected_stat = st.selectbox("Choose a stat to visualize:", stat_options, index=default_index)

    fig = go.Figure()

    career_start = player_data['year'].min()
    career_end = player_data['year'].max()

    # Calculate league averages for all years in the dataset
    all_league_averages = calculate_league_averages(full_data, selected_stat, player_type)
    
    # Filter league averages to only include years within the player's career
    league_averages = {year: avg for year, avg in all_league_averages.items() if career_start <= year <= career_end}

    for idfg in player_data['IDfg'].unique():
        player_subset = player_data[player_data['IDfg'] == idfg].sort_values('year')
        player_name = player_subset['Name'].iloc[0]
        team = player_subset['Team'].iloc[-1]
        color = team_colors.get(team, 'grey')

        # Player's performance line
        fig.add_trace(go.Scatter(
            x=player_subset['year'],
            y=player_subset[selected_stat],
            mode='lines+markers',
            name=player_name,
            line=dict(color=color),
            hovertemplate="<br>".join([
                "Year: %{x}",
                f"{selected_stat}: %{{y:.3f}}",
                "Name: " + player_name,
                "Team: " + team
            ])
        ))

        # Player's career average line
        career_avg = player_subset[selected_stat].mean()
        fig.add_trace(go.Scatter(
            x=[career_start, career_end],
            y=[career_avg, career_avg],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=f'{player_name} Career Average',
            hoverinfo='skip'
        ))

    # League average line
    league_avg_years = sorted(league_averages.keys())
    league_avg_values = [league_averages[year] for year in league_avg_years]
    fig.add_trace(go.Scatter(
        x=league_avg_years,
        y=league_avg_values,
        mode='lines',
        line=dict(color='black', dash='dot'),
        name='League Average',
        hovertemplate="<br>".join([
            "Year: %{x}",
            f"League Average {selected_stat}: %{{y:.3f}}",
        ])
    ))

    fig.update_layout(
        title=f"Yearly {selected_stat}",
        xaxis_title="Year",
        yaxis_title=selected_stat,
        legend_title="Player",
        hovermode="closest"
    )

    st.plotly_chart(fig)

def individual_player_view():
    st.title("Individual Player Statistics")

    player_type = st.radio("Select player type:", ["Hitter", "Pitcher"])
    data_df, player_years = load_and_prepare_data(player_type)

    # Set default player based on player type
    default_player = "Shohei Ohtani (2018-2024)" if player_type == "Hitter" else "Clayton Kershaw (2008-2024)"
    
    selected_player = st.selectbox("Select a player:", player_years['Label'].tolist(), index=player_years['Label'].tolist().index(default_player))
    selected_id = player_years[player_years['Label'] == selected_player]['IDfg'].iloc[0]

    player_data = data_df[data_df['IDfg'] == selected_id]

    if not player_data.empty:
        display_player_stats(player_data, player_type, data_df)
    else:
        st.write("No data available for the selected player.")

if __name__ == "__main__":
    individual_player_view()