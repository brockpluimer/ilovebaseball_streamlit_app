# Brock's Baseball Analytics Tool

This repository contains a baseball analytics tool I built with Python and Streamlit. It offers different modes for analyzing player statistics, comparing players, visualizing historical data, and performing advanced analytics. It's based on the pybaseball database, consisting of hitting and pitching data. I'll be adding fielding statistics, playoff statistics and additional features soon.

You can access the current version of the app here and on ilovemoneyball.streamlit.app

## Features

- **Individual Player Analysis**: View statistics for individual players.
- **Player Comparison**: Compare statistics between multiple players.
- **Historical Histogram**: Visualize league-wide statistics over time.
- **Career Stat Race**: Track and compare career statistics for multiple players.
- **Player Similarity**: Find players with similar statistical profiles.
- **Custom WAR Generator**: Create custom Wins Above Replacement (WAR) calculations.
- **Milestone Tracker**: Track progress towards career milestones.
- **Outlier Detection**: Identify statistical outliers in player performance.
- **Era Adjustment Tool**: Adjust statistics for different eras of baseball. (making adjustments, not optimized)
- **GOAT Analysis**: Analyze what makes a player the Greatest of All Time.
- **Supervised Learning**: Apply machine learning techniques to baseball data. (currently overloads streamlit server, working on distributing/otherwise adjusting)
- **Unsupervised Learning**: Discover patterns in baseball data using clustering and dimensionality reduction. (same as supervised)

## Dependencies

- pandas
- numpy
- streamlit
- plotly
- scikit-learn
- scipy

## Usage

To run the application:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run main.py
   ```

3. Use the sidebar to navigate between different analysis modes.

## File Structure

- `main.py`: The main application file containing the Streamlit interface.
- `load_data.py`: Functions for loading and preparing baseball data.
- `home_screen.py`: Displays the home screen of the application.
- `colors.py`: Manages team colors for visualizations.
- `individual_player_stats.py`: Handles individual player analysis.
- `compare_player_stats.py`: Manages player comparison functionality.
- `race.py`: Creates career stat race visualizations.
- `histogram.py`: Generates historical league-wide statistic visualizations.
- `similarity.py`: Calculates player similarity scores.
- `make_war.py`: Custom WAR calculation tool.
- `milestone_tracker.py`: Tracks career milestones.
- `goat.py`: Analyzes Greatest of All Time criteria.
- `outliers.py`: Identifies statistical outliers.
- `era_adjustment.py`: Adjusts statistics for different eras.
- `supervised_learning_mode.py`: Implements supervised learning techniques.
- `unsupervised_learning_mode.py`: Implements unsupervised learning techniques.
- `bangbang.py`: Generates fun facts about the Astros cheating scandal.

## Contributing

Contributions to improve the tool or add new features are welcome. Please submit a pull request or open an issue to discuss proposed changes.

## License

MIT License
