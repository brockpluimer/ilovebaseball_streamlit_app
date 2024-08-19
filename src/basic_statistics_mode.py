import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from load_data import load_and_prepare_data
from colors import load_team_colors, get_team_color

def basic_statistics_mode():
    st.title("Basic Statistics Mode")

    st.write("""
    This mode allows you to explore basic statistics for players and teams. 
    You can analyze individual seasons, compare players or teams, and investigate 
    trends across different years. Use the options below to customize your analysis.
    """)

    # Load data and team colors
    data_type = st.radio("Select Player Type", ["Hitters", "Pitchers"])
    data_df, player_years = load_and_prepare_data("Hitter" if data_type == "Hitters" else "Pitcher")
    team_colors = load_team_colors()

    # Year range selection
    year_range = st.slider("Select Year Range", 
                           min_value=int(data_df['year'].min()), 
                           max_value=int(data_df['year'].max()), 
                           value=(int(data_df['year'].min()), int(data_df['year'].max())))

    # Filter data based on year range
    filtered_data = data_df[(data_df['year'] >= year_range[0]) & (data_df['year'] <= year_range[1])]

    # Minimum PA/IP filter
    if data_type == "Hitters":
        min_pa = st.slider("Minimum Plate Appearances", 0, 700, 0)
        filtered_data = filtered_data[filtered_data['PA'] >= min_pa]
    else:
        min_ip = st.slider("Minimum Innings Pitched", 0, 300, 0)
        filtered_data = filtered_data[filtered_data['IP'] >= min_ip]

    # Analysis subject selection
    analysis_subject = st.radio("Select Analysis Subject", ["Players", "Teams"])

    # Analysis type selection
    if analysis_subject == "Players":
        analysis_type = st.selectbox("Select Analysis Type", 
                                     ["Descriptive Statistics", 
                                      "Player Comparison",
                                      "T-Tests", 
                                      "Correlation Matrix", 
                                      "Histograms",
                                      "Violin Plots"])
    else:
        analysis_type = st.selectbox("Select Analysis Type", 
                                     ["Descriptive Statistics", 
                                      "Team Comparison",
                                      "T-Tests", 
                                      "Correlation Matrix", 
                                      "Histograms",
                                      "Violin Plots"])

    # Career or Season analysis
    career_or_season = st.radio("Analyze Career or Season Statistics", ["Season", "Career"])

    if career_or_season == "Career":
        filtered_data = filtered_data.groupby(['IDfg', 'Name', 'Team'])
        rate_stats = ['AVG', 'OBP', 'SLG', 'OPS', 'ERA', 'WHIP', 'FIP']
        agg_dict = {col: 'mean' if col in rate_stats else 'sum' for col in filtered_data.columns if col not in ['IDfg', 'Name', 'Team']}
        filtered_data = filtered_data.agg(agg_dict)
        filtered_data = filtered_data.reset_index()

    if analysis_subject == "Players":
        if analysis_type == "Descriptive Statistics":
            descriptive_statistics(filtered_data, data_type, team_colors)
        elif analysis_type == "Player Comparison":
            player_comparison(filtered_data, data_type, team_colors)
        elif analysis_type == "T-Tests":
            t_tests(filtered_data, data_type, team_colors, group_by="Player")
        elif analysis_type == "Correlation Matrix":
            correlation_matrix(filtered_data, data_type, group_by="Player")
        elif analysis_type == "Histograms":
            histograms(filtered_data, data_type, team_colors)
        else:
            violin_plots(filtered_data, data_type, team_colors)
    else:  # Teams
        if analysis_type == "Descriptive Statistics":
            descriptive_statistics(filtered_data, data_type, team_colors, group_by="Team")
        elif analysis_type == "Team Comparison":
            team_comparison(filtered_data, data_type, team_colors)
        elif analysis_type == "T-Tests":
            t_tests(filtered_data, data_type, team_colors, group_by="Team")
        elif analysis_type == "Correlation Matrix":
            correlation_matrix(filtered_data, data_type, group_by="Team")
        elif analysis_type == "Histograms":
            histograms(filtered_data, data_type, team_colors, group_by="Team")
        else:
            violin_plots(filtered_data, data_type, team_colors, group_by="Team")

def get_relevant_stats(data_type):
    if data_type == "Hitters":
        return ["WAR", "AVG", "OBP", "SLG", "HR", "RBI", "SB", "2B", "R", "OPS"]
    else:  # Pitchers
        return ["WAR", "ERA", "WHIP", "W", "L", "SO", "BB", "FIP"]

def descriptive_statistics(data, data_type, team_colors, group_by=None):
    st.subheader("Descriptive Statistics")

    if group_by:
        entity = st.selectbox(f"Select {group_by}", sorted(data[group_by].unique()))
        data = data[data[group_by] == entity]

    relevant_stats = get_relevant_stats(data_type)
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    selected_columns = st.multiselect("Select statistics for analysis", 
                                      [col for col in numeric_columns if col in relevant_stats], 
                                      default=relevant_stats[:1])

    if not selected_columns:
        st.warning("Please select at least one statistic for analysis.")
        return

    desc_stats = data[selected_columns].describe()
    st.write(desc_stats)

    fig = px.box(data, y=selected_columns, color="Team", color_discrete_map=team_colors,
                 title="Distribution of Selected Statistics")
    st.plotly_chart(fig)

def player_comparison(data, data_type, team_colors):
    st.subheader("Player Comparison")

    # Set default years and players
    if data_type == "Hitters":
        default_year1, default_player1 = 2021, "Shohei Ohtani"
        default_year2, default_player2 = 2004, "Barry Bonds"
    else:  # Pitchers
        default_year1, default_player1 = 2021, "Shohei Ohtani"
        default_year2, default_player2 = 2014, "Clayton Kershaw"

    # Player 1 selection
    years1 = sorted(data['year'].unique(), reverse=True)
    selected_year1 = st.selectbox("Select Year for Player 1", years1, 
                                  index=years1.index(default_year1) if default_year1 in years1 else 0, 
                                  key='year1')
    players1 = sorted(data[data['year'] == selected_year1]['Name'].unique())
    
    # Find player with highest WAR if default player not available
    if default_player1 not in players1:
        default_player1 = data[(data['year'] == selected_year1)].sort_values('WAR', ascending=False)['Name'].iloc[0]
    
    player1 = st.selectbox("Select Player 1", players1, 
                           index=players1.index(default_player1) if default_player1 in players1 else 0,
                           key='player1')

    # Player 2 selection
    years2 = sorted(data['year'].unique(), reverse=True)
    selected_year2 = st.selectbox("Select Year for Player 2", years2, 
                                  index=years2.index(default_year2) if default_year2 in years2 else 0, 
                                  key='year2')
    players2 = sorted(data[data['year'] == selected_year2]['Name'].unique())
    
    # Find player with highest WAR if default player not available
    if default_player2 not in players2:
        default_player2 = data[(data['year'] == selected_year2)].sort_values('WAR', ascending=False)['Name'].iloc[0]
    
    player2 = st.selectbox("Select Player 2", players2, 
                           index=players2.index(default_player2) if default_player2 in players2 else 0,
                           key='player2')

    relevant_stats = get_relevant_stats(data_type)
    player1_data = data[(data['year'] == selected_year1) & (data['Name'] == player1)].iloc[0]
    player2_data = data[(data['year'] == selected_year2) & (data['Name'] == player2)].iloc[0]

    player1_stats = player1_data[relevant_stats]
    player2_stats = player2_data[relevant_stats]

    comparison_df = pd.DataFrame({f"{player1} ({selected_year1})": player1_stats, 
                                  f"{player2} ({selected_year2})": player2_stats})
    st.write(comparison_df)

    # Separate stats into groups
    if data_type == "Hitters":
        rate_stats = ["AVG", "OBP", "SLG", "OPS"]
        counting_stats = [stat for stat in relevant_stats if stat not in rate_stats and stat != "WAR"]
    else:  # Pitchers
        rate_stats = ["ERA", "WHIP", "FIP"]
        counting_stats = [stat for stat in relevant_stats if stat not in rate_stats and stat != "WAR"]

    # Plot rate stats
    fig_rate = go.Figure()
    for player, stats, year, color in [
        (player1, player1_stats, selected_year1, get_team_color(player1_data['Team'], team_colors)),
        (player2, player2_stats, selected_year2, get_team_color(player2_data['Team'], team_colors))
    ]:
        fig_rate.add_trace(go.Bar(
            x=rate_stats, y=stats[rate_stats], name=f"{player} ({year})",
            marker_color=color
        ))
    fig_rate.update_layout(barmode='group', title=f"Comparison of Rate Stats: {player1} ({selected_year1}) vs {player2} ({selected_year2})")
    st.plotly_chart(fig_rate)

    # Plot counting stats with WAR on a split axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for player, stats, year, color in [
        (player1, player1_stats, selected_year1, get_team_color(player1_data['Team'], team_colors)),
        (player2, player2_stats, selected_year2, get_team_color(player2_data['Team'], team_colors))
    ]:
        fig.add_trace(
            go.Bar(x=counting_stats, y=stats[counting_stats], name=f"{player} ({year})", marker_color=color),
            secondary_y=False
        )
        fig.add_trace(
            go.Bar(x=["WAR"], y=[stats["WAR"]], name=f"{player} ({year}) WAR", marker_color=color, opacity=0.7, showlegend=False),
            secondary_y=True
        )

    fig.update_layout(
        title=f"Comparison of Counting Stats and WAR: {player1} ({selected_year1}) vs {player2} ({selected_year2})",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Counting Stats", secondary_y=False)
    fig.update_yaxes(title_text="WAR", secondary_y=True)

    st.plotly_chart(fig)

def team_comparison(data, data_type, team_colors):
    st.subheader("Team Comparison")

    # Default years and teams
    default_year1, default_team1 = 1927, "NYY"
    default_year2, default_team2 = 2017, "LAD"

    # Team 1 selection
    years1 = sorted(data['year'].unique(), reverse=True)
    selected_year1 = st.selectbox("Select Year for Team 1", years1, 
                                  index=years1.index(default_year1) if default_year1 in years1 else 0, 
                                  key='year1')
    teams1 = sorted(data[data['year'] == selected_year1]['Team'].unique())
    team1 = st.selectbox("Select Team 1", teams1, 
                         index=teams1.index(default_team1) if default_team1 in teams1 else 0, 
                         key='team1')

    # Team 2 selection
    years2 = sorted(data['year'].unique(), reverse=True)
    selected_year2 = st.selectbox("Select Year for Team 2", years2, 
                                  index=years2.index(default_year2) if default_year2 in years2 else 0, 
                                  key='year2')
    teams2 = sorted(data[data['year'] == selected_year2]['Team'].unique())
    team2 = st.selectbox("Select Team 2", teams2, 
                         index=teams2.index(default_team2) if default_team2 in teams2 else 0, 
                         key='team2')

    relevant_stats = get_relevant_stats(data_type)
    team1_stats = data[(data['year'] == selected_year1) & (data['Team'] == team1)][relevant_stats].mean()
    team2_stats = data[(data['year'] == selected_year2) & (data['Team'] == team2)][relevant_stats].mean()

    comparison_df = pd.DataFrame({f"{team1} ({selected_year1})": team1_stats, 
                                  f"{team2} ({selected_year2})": team2_stats})
    st.write(comparison_df)

    # Separate stats into groups
    if data_type == "Hitters":
        rate_stats = ["AVG", "OBP", "SLG", "OPS"]
        counting_stats = [stat for stat in relevant_stats if stat not in rate_stats and stat != "WAR"]
    else:  # Pitchers
        rate_stats = ["ERA", "WHIP", "FIP"]
        counting_stats = [stat for stat in relevant_stats if stat not in rate_stats and stat != "WAR"]

    # Plot rate stats
    fig_rate = go.Figure()
    for team, stats, year, color in [
        (team1, team1_stats, selected_year1, get_team_color(team1, team_colors)),
        (team2, team2_stats, selected_year2, get_team_color(team2, team_colors))
    ]:
        fig_rate.add_trace(go.Bar(
            x=rate_stats, y=stats[rate_stats], name=f"{team} ({year})",
            marker_color=color
        ))
    fig_rate.update_layout(
        title=f"Comparison of Rate Stats: {team1} ({selected_year1}) vs {team2} ({selected_year2})",
        barmode='group'
    )
    st.plotly_chart(fig_rate)

    # Plot counting stats with WAR on a split axis
    fig_counting = make_subplots(specs=[[{"secondary_y": True}]])

    for team, stats, year, color in [
        (team1, team1_stats, selected_year1, get_team_color(team1, team_colors)),
        (team2, team2_stats, selected_year2, get_team_color(team2, team_colors))
    ]:
        fig_counting.add_trace(
            go.Bar(x=counting_stats, y=stats[counting_stats], name=f"{team} ({year})", marker_color=color),
            secondary_y=False
        )
        fig_counting.add_trace(
            go.Bar(x=["WAR"], y=[stats["WAR"]], name=f"{team} ({year}) WAR", marker_color=color, opacity=0.7, showlegend=False),
            secondary_y=True
        )

    fig_counting.update_layout(
        title=f"Comparison of Counting Stats and WAR: {team1} ({selected_year1}) vs {team2} ({selected_year2})",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_counting.update_yaxes(title_text="Counting Stats", secondary_y=False)
    fig_counting.update_yaxes(title_text="WAR", secondary_y=True)

    st.plotly_chart(fig_counting)

    # Comparison to League Average
    st.subheader("Comparison to League Average")
    league_avg1 = data[data['year'] == selected_year1][relevant_stats].mean()
    league_avg2 = data[data['year'] == selected_year2][relevant_stats].mean()

    team1_normalized = (team1_stats / league_avg1 - 1) * 100
    team2_normalized = (team2_stats / league_avg2 - 1) * 100

    # Create subplots: 3 rows, 1 column
    fig_normalized = make_subplots(rows=3, cols=1, subplot_titles=("Rate Stats", "Counting Stats", "WAR"),
                                   vertical_spacing=0.1)

    # Plot rate stats
    for team, stats, year, color in [
        (team1, team1_normalized, selected_year1, get_team_color(team1, team_colors)),
        (team2, team2_normalized, selected_year2, get_team_color(team2, team_colors))
    ]:
        fig_normalized.add_trace(
            go.Bar(x=rate_stats, y=stats[rate_stats], name=f"{team} ({year})", marker_color=color),
            row=1, col=1
        )

    # Plot counting stats
    for team, stats, year, color in [
        (team1, team1_normalized, selected_year1, get_team_color(team1, team_colors)),
        (team2, team2_normalized, selected_year2, get_team_color(team2, team_colors))
    ]:
        fig_normalized.add_trace(
            go.Bar(x=counting_stats, y=stats[counting_stats], name=f"{team} ({year})", marker_color=color),
            row=2, col=1
        )

    # Plot WAR
    for team, stats, year, color in [
        (team1, team1_normalized, selected_year1, get_team_color(team1, team_colors)),
        (team2, team2_normalized, selected_year2, get_team_color(team2, team_colors))
    ]:
        fig_normalized.add_trace(
            go.Bar(x=["WAR"], y=[stats["WAR"]], name=f"{team} ({year})", marker_color=color),
            row=3, col=1
        )

    # Update layout
    fig_normalized.update_layout(
        height=900,  # Increase height to accommodate three subplots
        title_text="Comparison to League Average (% above/below average)",
        barmode='group',
    )

    # Update y-axis titles
    fig_normalized.update_yaxes(title_text="% above/below average", row=1, col=1)
    fig_normalized.update_yaxes(title_text="% above/below average", row=2, col=1)
    fig_normalized.update_yaxes(title_text="% above/below average", row=3, col=1)

    st.plotly_chart(fig_normalized)

def t_tests(data, data_type, team_colors):
    st.subheader("T-Tests")

    relevant_stats = get_relevant_stats(data_type)
    selected_column = st.selectbox("Select statistic for analysis", relevant_stats)

    group_column = st.selectbox("Select grouping variable", ["Team", "year"])
    group1 = st.selectbox(f"Select first {group_column}", data[group_column].unique())
    group2 = st.selectbox(f"Select second {group_column}", [g for g in data[group_column].unique() if g != group1])

    group1_data = data[data[group_column] == group1][selected_column]
    group2_data = data[data[group_column] == group2][selected_column]
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)

    st.write(f"T-statistic: {t_stat}")
    st.write(f"P-value: {p_value}")

    if p_value < 0.05:
        st.write("There is a statistically significant difference between the two groups.")
    else:
        st.write("There is no statistically significant difference between the two groups.")

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=group1_data, name=f"{group_column}={group1}",
        marker_color=get_team_color(group1, team_colors) if group_column == "Team" else None
    ))
    fig.add_trace(go.Box(
        y=group2_data, name=f"{group_column}={group2}",
        marker_color=get_team_color(group2, team_colors) if group_column == "Team" else None
    ))
    fig.update_layout(title=f"Distribution of {selected_column} by {group_column}")
    st.plotly_chart(fig)

def correlation_matrix(data, data_type):
    st.subheader("Correlation Matrix")

    relevant_stats = get_relevant_stats(data_type)
    selected_columns = st.multiselect("Select statistics for correlation analysis", relevant_stats, default=relevant_stats[:4])

    if len(selected_columns) < 2:
        st.warning("Please select at least two statistics for correlation analysis.")
        return

    corr_matrix = data[selected_columns].corr()

    fig = px.imshow(corr_matrix, 
                    x=selected_columns, 
                    y=selected_columns, 
                    color_continuous_scale="RdBu_r", 
                    title="Correlation Matrix")
    st.plotly_chart(fig)

    st.write(corr_matrix)

def histograms(data, data_type, team_colors):
    st.subheader("Histograms")

    relevant_stats = get_relevant_stats(data_type)
    selected_column = st.selectbox("Select statistic for histogram", relevant_stats)

    fig = px.histogram(data, x=selected_column, color="Team", color_discrete_map=team_colors,
                       title=f"Distribution of {selected_column}")
    st.plotly_chart(fig)

    summary_stats = data[selected_column].describe()
    st.write(summary_stats)

def violin_plots(data, data_type, team_colors):
    st.subheader("Violin Plots")

    relevant_stats = get_relevant_stats(data_type)
    selected_column = st.selectbox("Select statistic for violin plot", relevant_stats)
    group_column = st.selectbox("Select grouping variable", ["Team", "year"])

    fig = px.violin(data, y=selected_column, x=group_column, box=True, points="all",
                    color="Team" if group_column == "Team" else None,
                    color_discrete_map=team_colors if group_column == "Team" else None,
                    title=f"Distribution of {selected_column} by {group_column}")
    st.plotly_chart(fig)

if __name__ == "__main__":
    basic_statistics_mode()