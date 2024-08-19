import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from load_data import load_and_prepare_data

def calculate_era_factors(data, stat, start_year, end_year):
    yearly_averages = data[(data['year'] >= start_year) & (data['year'] <= end_year)].groupby('year')[stat].mean()
    overall_average = yearly_averages.mean()
    era_factors = overall_average / yearly_averages
    return era_factors

def adjust_stat(value, year, era_factors, is_counting_stat):
    if year in era_factors.index:
        adjusted = value * era_factors[year]
        return max(0, adjusted) if is_counting_stat else adjusted
    return value

def format_stat(value, original_dtype):
    if np.issubdtype(original_dtype, np.integer):
        return int(max(0, round(value)))
    elif np.issubdtype(original_dtype, np.float64):
        return round(value, 3)  # Adjust the number of decimal places as needed
    return value

def era_adjustment_tool():
    st.subheader("Era Adjustment Tool")

    st.write("""
    This tool allows you to adjust player statistics for different eras of baseball, 
    accounting for changes in the game over time. Select a statistic, a player, and a 
    time range to see how their performance compares when adjusted for era.
    """)

    # Load data
    data_type = st.radio("Select player type:", ("Hitter", "Pitcher"))
    data, _ = load_and_prepare_data(data_type)

    # Select statistic to adjust
    if data_type == "Hitter":
        default_stats = ['AVG', 'OBP', 'SLG', 'OPS', 'HR', 'RBI', 'SB', 'WAR']
    else:
        default_stats = ['ERA', 'WHIP', 'K/9', 'BB/9', 'FIP', 'WAR']

    stat_to_adjust = st.selectbox("Select statistic to adjust:", default_stats)

    # Select player
    players = sorted(data['Name'].unique())
    selected_player = st.selectbox("Select a player:", players)

    # Select time range
    min_year, max_year = int(data['year'].min()), int(data['year'].max())
    start_year, end_year = st.slider("Select time range:", min_year, max_year, (min_year, max_year))

    # Calculate era factors
    era_factors = calculate_era_factors(data, stat_to_adjust, start_year, end_year)

    # Get player data and adjust stats
    player_data = data[data['Name'] == selected_player].copy()
    is_counting_stat = np.issubdtype(player_data[stat_to_adjust].dtype, np.integer)
    original_dtype = player_data[stat_to_adjust].dtype
    
    player_data['Adjusted ' + stat_to_adjust] = player_data.apply(
        lambda row: format_stat(
            adjust_stat(row[stat_to_adjust], row['year'], era_factors, is_counting_stat),
            original_dtype
        ),
        axis=1
    )

    # Display results
    st.subheader(f"Era-Adjusted {stat_to_adjust} for {selected_player}")

    # Toggle for regression lines
    show_regression = st.checkbox("Show regression lines", value=False)

    # Create the scatter plot
    fig = go.Figure()

    # Add actual values as circles
    fig.add_trace(go.Scatter(
        x=player_data['year'],
        y=player_data[stat_to_adjust],
        mode='markers',
        name='Actual',
        marker=dict(symbol='circle', size=10),
        hovertemplate='Year: %{x}<br>Actual ' + stat_to_adjust + ': %{y:.3f}<extra></extra>'
    ))

    # Add adjusted values as X's
    fig.add_trace(go.Scatter(
        x=player_data['year'],
        y=player_data['Adjusted ' + stat_to_adjust],
        mode='markers',
        name='Era-Adjusted',
        marker=dict(symbol='x', size=10),
        hovertemplate='Year: %{x}<br>Adjusted ' + stat_to_adjust + ': %{y:.3f}<extra></extra>'
    ))

    # # Add regression lines if toggled
    # if show_regression:
    #     for data_type in ['Actual', 'Adjusted']:
    #         y = player_data[stat_to_adjust] if data_type == 'Actual' else player_data['Adjusted ' + stat_to_adjust]
    #         slope, intercept, r_value, _, _ = stats.linregress(player_data['year'], y)
    #         line_y = slope * player_data['year'] + intercept
    #         fig.add_trace(go.Scatter(
    #             x=player_data['year'],
    #             y=line_y,
    #             mode='lines',
    #             name=f'{data_type} Trend (R² = {r_value**2:.3f})',
    #             line=dict(dash='dash'),
    #         ))

    fig.update_layout(
        title=f"{selected_player}'s {stat_to_adjust} - Original vs Era-Adjusted",
        xaxis_title="Year",
        yaxis_title=stat_to_adjust,
        legend_title="Data Type",
        hovermode="closest"
    )

    st.plotly_chart(fig)

    # Display data table with formatted year
    display_data = player_data[['year', stat_to_adjust, 'Adjusted ' + stat_to_adjust]].copy()
    display_data['year'] = display_data['year'].astype(int)  # Remove commas from year
    st.write(display_data)

    # # Predict stat for a given year
    # st.subheader("Predict Stat for a Given Year")
    # predict_year = st.number_input("Enter a year to predict the stat:", min_value=int(min_year), max_value=int(max_year), value=int(start_year))

    # # Calculate predictions
    # actual_slope, actual_intercept, _, _, _ = stats.linregress(player_data['year'], player_data[stat_to_adjust])
    # adjusted_slope, adjusted_intercept, _, _, _ = stats.linregress(player_data['year'], player_data['Adjusted ' + stat_to_adjust])

    # predicted_actual = format_stat(actual_slope * predict_year + actual_intercept, original_dtype)
    # predicted_adjusted = format_stat(adjusted_slope * predict_year + adjusted_intercept, original_dtype)

    # st.write(f"Predicted {stat_to_adjust} for {selected_player} in {predict_year}:")
    # st.write(f"Actual (based on trend): {predicted_actual}")
    # st.write(f"Era-Adjusted (based on trend): {predicted_adjusted}")

    # Explanation of adjustment
    st.subheader("How the Adjustment Works")
    st.write("""
    The era adjustment is calculated by comparing the league average for the selected statistic
    in each year to the overall average across the selected time range. This creates an 'era factor'
    for each year, which is then applied to the player's statistics.

    If a year's era factor is greater than 1, it means the statistic was generally lower in that year
    compared to the overall average, so player values are adjusted upwards. If it's less than 1,
    the opposite is true.

    This method helps to level the playing field when comparing players across different eras of baseball.
    """)

    # Display era factors
    st.subheader("Era Factors")
    st.write(era_factors)

if __name__ == "__main__":
    era_adjustment_tool()