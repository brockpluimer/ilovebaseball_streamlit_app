import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
from typing import List, Union, Optional
import plotly.express as px
from load_data import load_and_filter_data

def anomaly_tracker():
    st.subheader("Statistical Outliers Identifier")

    st.markdown("""
    This tool identifies statistically anomalous seasons or careers, highlighting extreme 
    performances in baseball history. Here's how to use it:

    1. Choose whether you want to identify outliers for hitters or pitchers.
    2. Select whether you're analyzing a single-season or career performances.
    3. Choose the statistics you're interested in analyzing.
    4. Set the z-score threshold to define what constitutes an outlier.
    5. Optionally, set minimum playing time requirements to filter out small sample sizes.
    6. The tool will display all players whose performance exceeds the specified threshold.

    This can be used to explore rare performances, track historical anomalies, or see which players 
    had exceptional seasons or careers.
    """)

    player_type = st.radio("Select player type:", ("Hitters", "Pitchers"), key="anomaly_player_type")
    data_type = "Hitter" if player_type == "Hitters" else "Pitcher"

    anomaly_type = st.radio("Select anomaly type:", ("Single Season", "Career"), key="anomaly_type")

    data_df = load_and_filter_data(data_type)

    if data_type == "Hitter":
        default_stats = ['G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'OPS', 'wRC+', 'WAR']
    else:  # Pitcher
        default_stats = ['W', 'L', 'ERA', 'G', 'GS', 'CG', 'SHO', 'SV', 'IP', 'H', 'R', 'ER', 'HR', 'BB', 'SO', 'WHIP', 'K/9', 'BB/9', 'FIP', 'WAR']

    mode = st.radio("Select mode:", ("Single Stat", "Multiple Stats"), key="anomaly_mode")
    selected_stats = st.multiselect("Select statistics to analyze for outliers:", default_stats, key="anomaly_stats")

    z_threshold = st.number_input("Z-score threshold for outliers:", min_value=1.0, max_value=10.0, value=3.0, step=0.1, key="z_threshold")

    # Minimum playing time filter
    use_min_filter = st.checkbox("Set minimum playing time filter?", key="anomaly_min_filter")
    min_pa = min_ip = None
    if use_min_filter:
        if data_type == "Hitter":
            min_pa = st.number_input("Minimum PA:", min_value=1, value=300, key="anomaly_min_pa")
        else:  # Pitcher
            min_ip = st.number_input("Minimum IP:", min_value=1, value=50, key="anomaly_min_ip")

    if st.button("Identify Statistical Outliers"):
        if selected_stats:
            # Apply minimum playing time filter
            if use_min_filter:
                if data_type == "Hitter":
                    filtered_df = data_df[data_df['PA'] >= min_pa]
                else:  # Pitcher
                    filtered_df = data_df[data_df['IP'] >= min_ip]
            else:
                filtered_df = data_df

            if anomaly_type == "Single Season":
                if mode == "Single Stat":
                    outlier_df = identify_statistical_outliers(filtered_df, selected_stats, z_threshold)
                else:
                    outlier_df = identify_multi_stat_outliers(filtered_df, selected_stats, z_threshold)
                
                if not outlier_df.empty:
                    st.success(f"Found {len(outlier_df)} statistically anomalous seasons based on the selected threshold!")
                    
                    display_columns = ['Name', 'year', 'Team'] + [col for col in default_stats if col in selected_stats]
                    
                    format_dict = {
                        'year': '{:d}'
                    }

                    # Apply formatting based on column data type
                    for stat in selected_stats:
                        if outlier_df[stat].dtype.kind == 'i':  # Integer type
                            format_dict[stat] = '{:,d}'
                        else:  # Float type
                            if stat in ['AVG', 'OBP', 'SLG', 'OPS', 'ERA', 'WHIP']:
                                format_dict[stat] = '{:.3f}'
                            elif stat in ['WAR', 'K/9', 'BB/9', 'FIP']:
                                format_dict[stat] = '{:.1f}'
                            else:
                                format_dict[stat] = '{:.0f}'

                    st.dataframe(outlier_df[display_columns].style.format(format_dict))
                    
                    plot_stat_distributions(filtered_df, selected_stats, z_threshold, mode)

                else:
                    st.warning("No outliers found with the current settings.")
            else:  # Career anomaly
                career_stats = filtered_df.groupby(['IDfg', 'Name']).agg({
                    **{stat: 'sum' if stat not in ['AVG', 'OBP', 'SLG', 'OPS', 'ERA', 'WHIP', 'K/9', 'BB/9', 'FIP'] else 'mean' for stat in selected_stats},
                    'year': ['min', 'max']
                }).reset_index()
                career_stats.columns = ['IDfg', 'Name'] + selected_stats + ['First Year', 'Last Year']
                career_stats['Years Played'] = career_stats['Last Year'] - career_stats['First Year'] + 1

                # Apply minimum playing time filter for career
                if use_min_filter:
                    if data_type == "Hitter":
                        career_totals = filtered_df.groupby(['IDfg', 'Name']).agg({'PA': 'sum'}).reset_index()
                        career_stats = career_stats.merge(career_totals, on=['IDfg', 'Name'])
                        career_stats = career_stats[career_stats['PA'] >= min_pa * career_stats['Years Played']]
                    else:  # Pitcher
                        career_totals = filtered_df.groupby(['IDfg', 'Name']).agg({'IP': 'sum'}).reset_index()
                        career_stats = career_stats.merge(career_totals, on=['IDfg', 'Name'])
                        career_stats = career_stats[career_stats['IP'] >= min_ip * career_stats['Years Played']]

                if mode == "Single Stat":
                    outlier_df = identify_statistical_outliers(career_stats, selected_stats, z_threshold)
                else:
                    outlier_df = identify_multi_stat_outliers(career_stats, selected_stats, z_threshold)

                if not outlier_df.empty:
                    st.success(f"Found {len(outlier_df)} players with statistically anomalous careers based on the selected threshold!")
                    
                    outlier_df['Display Name'] = outlier_df['Name'] + ' (' + outlier_df['First Year'].astype(str) + '-' + outlier_df['Last Year'].astype(str) + ')'
                    
                    display_columns = ['Display Name'] + selected_stats + ['First Year', 'Last Year', 'Years Played']
                    
                    format_dict = {
                        'First Year': '{:d}',
                        'Last Year': '{:d}'
                    }

                    # Apply formatting based on column data type
                    for stat in selected_stats:
                        if outlier_df[stat].dtype.kind == 'i':  # Integer type
                            format_dict[stat] = '{:,d}'
                        else:  # Float type
                            if stat in ['AVG', 'OBP', 'SLG', 'OPS', 'ERA', 'WHIP']:
                                format_dict[stat] = '{:.3f}'
                            elif stat in ['WAR', 'K/9', 'BB/9', 'FIP']:
                                format_dict[stat] = '{:.1f}'
                            else:
                                format_dict[stat] = '{:.0f}'

                    st.dataframe(outlier_df[display_columns].style.format(format_dict))

                    plot_stat_distributions(career_stats, selected_stats, z_threshold, mode)

                else:
                    st.warning("No outliers found with the current settings.")
        else:
            st.warning("Please select at least one statistic to analyze for outliers.")

def identify_statistical_outliers(df: pd.DataFrame, stats: List[str], threshold: float) -> pd.DataFrame:
    """
    Identify and return rows in the DataFrame where any of the specified stats exceed
    the given z-score threshold for outliers.
    
    :param df: DataFrame containing player statistics.
    :param stats: List of statistics to check for outliers.
    :param threshold: Z-score threshold to identify outliers.
    :return: DataFrame of outlier rows.
    """
    z_scores = df[stats].apply(lambda x: zscore(x, nan_policy='omit'))
    outliers = (np.abs(z_scores) > threshold).any(axis=1)
    outlier_df = df[outliers]
    return outlier_df

def identify_multi_stat_outliers(df: pd.DataFrame, stats: List[str], threshold: float) -> pd.DataFrame:
    """
    Identify and return rows in the DataFrame where all of the specified stats exceed
    the given z-score threshold for outliers.
    
    :param df: DataFrame containing player statistics.
    :param stats: List of statistics to check for outliers.
    :param threshold: Z-score threshold to identify outliers.
    :return: DataFrame of outlier rows where all selected stats are outliers.
    """
    z_scores = df[stats].apply(lambda x: zscore(x, nan_policy='omit'))
    outliers = (np.abs(z_scores) > threshold).all(axis=1)
    outlier_df = df[outliers]
    return outlier_df

def plot_stat_distributions(df: pd.DataFrame, stats: List[str], threshold: float, mode: str):
    """
    Plot the distribution of each selected stat and highlight the z-score boundaries for outliers using Plotly.

    :param df: DataFrame containing player statistics.
    :param stats: List of statistics to plot.
    :param threshold: Z-score threshold for highlighting outliers.
    :param mode: Analysis mode - "Single Stat" or "Multiple Stats".
    """
    for stat in stats:
        data = df[stat].dropna()
        
        if data.empty:
            st.warning(f"No data available for {stat}.")
            continue

        mean = data.mean()
        std_dev = data.std()

        # Calculate the z-score boundaries
        left_boundary = max(0, mean - (threshold * std_dev))  # Ensure no negative values unless data includes negatives
        right_boundary = mean + (threshold * std_dev)

        # Create histogram using Plotly
        fig = px.histogram(
            data, 
            nbins=30, 
            title=f'Distribution of {stat} with Outlier Boundaries', 
            labels={stat: stat},
            template="plotly_dark"
        )
        fig.update_traces(marker_color='skyblue', marker_line_color='black', marker_line_width=1.5)

        # Add lines for the z-score boundaries
        y_max = data.value_counts().max()  # Get the max frequency count for y-axis height

        fig.add_shape(
            type="line",
            x0=left_boundary,
            y0=0,
            x1=left_boundary,
            y1=y_max,
            line=dict(color="red", width=2, dash="dash"),
            name=f'Lower Bound (z=-{threshold})'
        )

        fig.add_shape(
            type="line",
            x0=right_boundary,
            y0=0,
            x1=right_boundary,
            y1=y_max,
            line=dict(color="green", width=2, dash="dash"),
            name=f'Upper Bound (z={threshold})'
        )

        # Update layout to add legend and enhance plot aesthetics
        fig.update_layout(
            xaxis_title=stat,
            yaxis_title='Frequency',
            legend_title='Boundaries',
            showlegend=True,
            bargap=0.1,  # Add some space between bars
            title_font=dict(size=20),  # Increase title font size
            xaxis=dict(showgrid=False),  # Disable grid for cleaner look
            yaxis=dict(showgrid=False)
        )

        # Add annotation for boundaries
        fig.add_annotation(
            x=left_boundary,
            y=y_max / 2,
            text=f'Lower Bound (z=-{threshold})',
            showarrow=True,
            arrowhead=1,
            ax=-40,  # X offset for arrow
            ay=-40,  # Y offset for arrow
            bgcolor="red",
            opacity=0.8,
            font=dict(color="white", size=10)
        )

        fig.add_annotation(
            x=right_boundary,
            y=y_max / 2,
            text=f'Upper Bound (z={threshold})',
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=-40,
            bgcolor="green",
            opacity=0.8,
            font=dict(color="white", size=10)
        )

        st.plotly_chart(fig)