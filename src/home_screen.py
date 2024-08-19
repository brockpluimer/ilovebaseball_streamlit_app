import streamlit as st

def display_home_screen():
    st.title("⚾ Welcome to Brock's Baseball Stats Explorer! ⚾")

    st.write("""
    Here you'll find tools to help you explore baseball data. I have pulled data for all players available for all seasons archived in Fangraphs (accessed via PyBaseball).

    Click the panel on the left to explore the different tools:
    """)

    st.subheader("Available Tools:")

    tools = {
        "Individual Player": "Explore yearly and career statistics for any individual player.",
        "Compare Players": "Compare statistics between 2-10 players.",
        "Historical Histogram": "View the distribution of chosen statistics across different seasons.",
        "Milestone Tracker": "Track a player's progress towards significant career milestones.",
        "Find Outliers": "Identify statistically significant outliers in player performances.",
        "Career Stat Race": "Visualize how league leaders in career stats have progressed over time.",
        "Player Similarity": "Identify players with similar statistical profiles.",
        "Custom WAR Generator": "Create your own custom Wins Above Replacement (WAR) metric.",
        "How is he the GOAT?": "Explore what makes certain players stand out as the 'Greatest of All Time'.",
        "Era Adjustment Tool": "Adjust statistics to account for different eras in baseball.",
        #"Basic Statistics": "Perform basic statistical analyses on player data.",
        "Supervised Learning": "Use machine learning to classify players or predict career outcomes e.g. Hall of Fame.",
        "Unsupervised Learning": "Discover hidden patterns and groupings in player data."
    }

    for tool, description in tools.items():
        st.write(f"**{tool}**: {description}")

    st.write("""
    If you have any questions or feedback, please don't hesitate to reach out @ brock.pluimer@gmail.com!

    """)