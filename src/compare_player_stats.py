import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

    # Set Name as index and drop the Name column
    career_stats = career_stats.set_index('Name')

    return career_stats

def calculate_league_averages(data: pd.DataFrame, stat: str, player_type: str) -> Dict[int, float]:
    league_averages = {}
    min_ab = 200 if player_type == "Hitter" else 50
    ab_column = 'AB' if player_type == "Hitter" else 'IP'
    
    for year in data['year'].unique():
        year_data = data[data['year'] == year]
        qualified = year_data[year_data[ab_column] >= min_ab]
        if not qualified.empty:
            if stat in ['AVG', 'OBP', 'SLG', 'OPS', 'ERA', 'WHIP']:
                weight = 'AB' if player_type == "Hitter" else 'IP'
                league_averages[year] = (qualified[stat] * qualified[weight]).sum() / qualified[weight].sum()
            else:
                league_averages[year] = qualified[stat].mean()
    
    return league_averages

def display_comparative_stat_explorer(players_data: List[pd.DataFrame], player_type: str, team_colors: dict, full_data: pd.DataFrame):
    all_data = pd.concat(players_data)
    numeric_columns = all_data.select_dtypes(include=['int64', 'float64']).columns
    stat_options = [col for col in numeric_columns if col not in ['year', 'IDfg', 'season']]
    
    default_stat = 'WAR' if 'WAR' in stat_options else stat_options[0]
    default_index = stat_options.index(default_stat)
    
    selected_stat = st.selectbox("Choose a stat to compare:", stat_options, index=default_index)

    # Define rate stats
    rate_stats = ['AVG', 'OBP', 'SLG', 'OPS', 'BB%', 'K%', 'ISO', 'BABIP', 'wRC+', 'wOBA', 'ERA', 'WHIP', 'K/9', 'BB/9', 'H/9', 'HR/9', 'K/BB', 'FIP', 'xFIP']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Yearly Comparison", "Career Progression"))

    # Calculate overall career span for all players
    max_career_length = max(len(player_data) for player_data in players_data)

    for player_data in players_data:
        player_name = player_data['Name'].iloc[0]
        team = player_data['Team'].iloc[-1]  # Use the most recent team
        color = team_colors.get(team, 'grey')

        player_data = player_data.sort_values('year')  # Ensure data is sorted by year
        player_data['Career Year'] = list(range(1, len(player_data) + 1))

        # Yearly comparison plot
        fig.add_trace(go.Scatter(
            x=player_data['Career Year'].tolist(),
            y=player_data[selected_stat].tolist(),
            mode='lines+markers',
            name=player_name,
            line=dict(color=color),
            hovertemplate="<br>".join([
                "Career Year: %{x}",
                "Actual Year: %{customdata}",
                f"{selected_stat}: %{{y:.3f}}",
                "Name: " + player_name,
                "Team: " + team
            ]),
            customdata=player_data['year'].tolist()
        ), row=1, col=1)

        # Career average line for yearly comparison
        career_avg = player_data[selected_stat].mean()
        fig.add_trace(go.Scatter(
            x=[1, len(player_data)],
            y=[career_avg, career_avg],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=f'{player_name} Career Average',
            hoverinfo='skip'
        ), row=1, col=1)

        # Career progression plot
        if selected_stat in rate_stats:
            weight = 'IP' if player_type == "Pitcher" else ('PA' if 'PA' in player_data.columns else 'G')
            cumulative_stat = (player_data[selected_stat] * player_data[weight]).cumsum() / player_data[weight].cumsum()
            plot_title = f"Career Average {selected_stat}"
        else:
            cumulative_stat = player_data[selected_stat].cumsum()
            plot_title = f"Cumulative {selected_stat}"

        fig.add_trace(go.Scatter(
            x=player_data['Career Year'].tolist(),
            y=cumulative_stat.tolist(),
            mode='lines+markers',
            name=player_name,
            line=dict(color=color),
            hovertemplate="<br>".join([
                "Career Year: %{x}",
                "Actual Year: %{customdata}",
                f"{plot_title}: %{{y:.3f}}",
                "Name: " + player_name,
                "Team: " + team
            ]),
            customdata=player_data['year'].tolist()
        ), row=2, col=1)

    # Calculate and add league average line for yearly comparison
    league_averages = calculate_league_averages(full_data, selected_stat, player_type)
    league_avg_years = list(range(1, max_career_length + 1))
    league_avg_values = [league_averages.get(year, None) for year in range(min(league_averages.keys()), max(league_averages.keys()) + 1)]
    league_avg_values += [None] * (max_career_length - len(league_avg_values))
    
    fig.add_trace(go.Scatter(
        x=league_avg_years,
        y=league_avg_values,
        mode='lines',
        line=dict(color='black', dash='dot'),
        name='League Average',
        hovertemplate="<br>".join([
            "Career Year: %{x}",
            f"League Average {selected_stat}: %{{y:.3f}}",
        ])
    ), row=1, col=1)

    fig.update_layout(
        title=f"{selected_stat} Comparison",
        xaxis_title="Career Year",
        yaxis_title=selected_stat,
        xaxis2_title="Career Year",
        yaxis2_title=plot_title,
        legend_title="Player",
        hovermode="closest",
        height=800,  # Increase the height to accommodate two plots
    )

    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1, range=[1, max_career_length])

    st.plotly_chart(fig)

def compare_players_stats(players_data: List[pd.DataFrame], player_type: str, full_data: pd.DataFrame):
    team_colors = load_team_colors()

    st.header("Career Comparison")
    career_stats = pd.concat([calculate_career_stats(player_data, player_type) for player_data in players_data])
    st.dataframe(career_stats)

    st.header("Stat Explorer")
    display_comparative_stat_explorer(players_data, player_type, team_colors, full_data)

def compare_players_view():
    st.title("Compare Players Statistics")

    player_type = st.radio("Select player type:", ["Hitter", "Pitcher"])
    data_df, player_years = load_and_prepare_data(player_type)

    # Set default players based on player type
    if player_type == "Hitter":
        default_players = ["Shohei Ohtani (2018-2024)", "Barry Bonds (1986-2007)"]
    else:
        default_players = ["Clayton Kershaw (2008-2024)", "Nolan Ryan (1966-1993)"]

    num_players = st.number_input("Number of players to compare", min_value=2, max_value=5, value=2)
    
    selected_players = []
    for i in range(num_players):
        default_index = player_years['Label'].tolist().index(default_players[i]) if i < len(default_players) else 0
        player = st.selectbox(f"Select player {i+1}:", player_years['Label'].tolist(), index=default_index, key=f"player_{i}")
        selected_players.append(player)

    players_data = []
    for player in selected_players:
        selected_id = player_years[player_years['Label'] == player]['IDfg'].iloc[0]
        player_data = data_df[data_df['IDfg'] == selected_id]
        if not player_data.empty:
            players_data.append(player_data)

    if len(players_data) == num_players:
        compare_players_stats(players_data, player_type, data_df)
    else:
        st.write("Data not available for all selected players.")

if __name__ == "__main__":
    compare_players_view()