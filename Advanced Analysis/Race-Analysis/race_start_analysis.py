import fastf1
import fastf1.plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
fastf1.Cache.enable_cache('cache/')
fastf1.plotting.setup_mpl()

# --- Helper Functions ---

def get_completed_races(year):
    """
    Fetches the schedule for a given year and returns a list of completed race events.
    """
    print(f"Fetching event schedule for {year}...")
    try:
        schedule = fastf1.get_event_schedule(year)
        today = pd.to_datetime('today').date()
        
        # Ensure EventDate is datetime and then extract date part
        if 'EventDate' in schedule.columns:
            schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
            
        # Filter out non-race events and ensure the event date is in the past
        completed_races = schedule.loc[
            (~schedule['EventFormat'].isin(['testing', 'practice'])) & 
            (schedule['EventDate'].dt.date < today)
        ]
        print(f"Found {len(completed_races)} completed race(s) to analyze.")
        return completed_races
    except Exception as e:
        print(f"Error fetching schedule for {year}: {e}")
        return pd.DataFrame()

def get_lap1_data(session):
    """
    Extracts grid position and lap 1 position data from a race session.
    """
    try:
        session.load(telemetry=False, weather=False, messages=False)
        print(f"Successfully loaded session data for {session.event['EventName']}.")

        # Check if session.results is valid and contains necessary columns
        if session.results.empty or 'Abbreviation' not in session.results.columns or 'GridPosition' not in session.results.columns:
            print(f"Session results for {session.event['EventName']} are empty or missing required columns (Abbreviation, GridPosition).")
            return None

        laps = session.laps
        if laps.empty:
            print("No lap data available for this session.")
            return None

        grid_positions = session.results[['Abbreviation', 'GridPosition']].set_index('Abbreviation')
        lap1 = laps.loc[laps['LapNumber'] == 1]
        
        if lap1.empty:
            print("No Lap 1 data available.")
            return None

        lap1_positions = lap1[['Abbreviation', 'Position']].set_index('Abbreviation')
        
        # Get Lap 1 time for context
        lap1_times = lap1[['Abbreviation', 'LapTime']].set_index('Abbreviation')

        # Combine the data
        combined_data = grid_positions.join(lap1_positions, how='inner').join(lap1_times, how='inner')
        combined_data.rename(columns={'Position': 'EndLap1Position'}, inplace=True)
        
        return combined_data.reset_index()

    except Exception as e:
        print(f"An error occurred loading data for {session.event['EventName']}: {e}")
        return None

def calculate_start_performance(data):
    """
    Calculates performance metrics based on lap 1 data.
    """
    data['PositionsGained'] = data['GridPosition'] - data['EndLap1Position']
    
    # Calculate time delta to the lap 1 leader
    leader_lap1_time = data['LapTime'].min()
    data['Lap1TimeDelta'] = data['Lap1Time'] - leader_lap1_time
    data['Lap1TimeDelta(s)'] = data['Lap1TimeDelta'].dt.total_seconds()

    return data

# --- Visualization Functions ---

def plot_performance_distribution(df, year):
    """
    Plots a box plot showing the distribution of positions gained for each driver.
    """
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='Driver', y='PositionsGained', hue='Driver', palette='viridis', dodge=False)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f'Distribution of Positions Gained on Lap 1 - {year} Season', fontsize=16, fontweight='bold')
    plt.xlabel('Driver', fontsize=12)
    plt.ylabel('Positions Gained (Positive) / Lost (Negative)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend().set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_gains_vs_grid_position(df, year):
    """
    Plots a scatter plot of positions gained vs. starting grid position.
    """
    plt.figure(figsize=(14, 8))
    sns.regplot(data=df, x='GridPosition', y='PositionsGained', scatter_kws={'alpha':0.6})
    plt.title(f'Lap 1 Performance vs. Starting Grid Position - {year} Season', fontsize=16, fontweight='bold')
    plt.xlabel('Starting Grid Position', fontsize=12)
    plt.ylabel('Positions Gained', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().invert_xaxis() # P1 is on the left
    plt.tight_layout()
    plt.show()

# --- Main Analysis Function ---

def analyze_race_start_performance(year):
    """
    Performs a comprehensive analysis of first-lap performance for all drivers
    across a given F1 season.
    """
    print(f"\n--- Analyzing Race Start Performance for {year} ---")
    
    race_events = get_completed_races(year)
    if race_events.empty:
        return

    all_starts_data = []
    for _, event in race_events.iterrows():
        print(f"\nProcessing {event['EventName']} (Round {event['RoundNumber']})...")
        session = fastf1.get_session(year, event['RoundNumber'], 'Race')
        lap1_data = get_lap1_data(session)

        if lap1_data is not None:
            performance_data = calculate_start_performance(lap1_data)
            performance_data['Round'] = event['RoundNumber']
            performance_data['GrandPrix'] = event['EventName']
            all_starts_data.append(performance_data)

    if not all_starts_data:
        print("\nNo sufficient race start data collected for the season.")
        return
        
    full_season_df = pd.concat(all_starts_data)
    # Analyze all drivers present in the data
    analysis_df = full_season_df.copy()
    analysis_df.rename(columns={'Abbreviation': 'Driver'}, inplace=True)

    if analysis_df.empty:
        print(f"\nNo data found for any drivers.")
        return

    # --- Visualizations ---
    plot_performance_distribution(analysis_df, year)
    plot_gains_vs_grid_position(analysis_df, year)

    # --- Summary Statistics & Insights ---
    print("\n--- Season Summary of Race Start Performance ---")
    summary = analysis_df.groupby('Driver')['PositionsGained'].agg(
        ['mean', 'sum', 'std', 'max', 'min']
    ).reset_index()
    summary.rename(columns={
        'mean': 'AvgGain', 'sum': 'TotalGain', 'std': 'Consistency',
        'max': 'BestStart', 'min': 'WorstStart'
    }, inplace=True)
    summary = summary.sort_values(by='AvgGain', ascending=False)

    print("Key Metrics:")
    print("- AvgGain: Average positions gained. Higher is better.")
    print("- TotalGain: Net positions gained over the season.")
    print("- Consistency (Std Dev): Lower is more predictable. NaN if only one race.")
    print("-" * 60)

    for _, row in summary.iterrows():
        driver = row['Driver']
        avg_gain = row['AvgGain']
        consistency = row['Consistency']
        
        insight = "is a remarkably consistent starter."
        if consistency > 1.5:
            insight = "has shown variable start performance."
        elif consistency < 0.75:
            insight = "is a highly consistent and predictable starter."

        print(f"\n{driver} ({insight})")
        print(f"  - Average Positions Gained: {avg_gain:.2f}")
        print(f"  - Total Positions Gained (Season): {row['TotalGain']:.0f}")
        print(f"  - Consistency (Std Dev): {consistency:.2f}")
        print(f"  - Best Start: +{row['BestStart']:.0f} positions | Worst Start: {row['WorstStart']:.0f} positions")

# --- Example Usage ---
if __name__ == "__main__":
    CURRENT_YEAR = pd.to_datetime('today').year
    YEAR_TO_ANALYZE = CURRENT_YEAR

    # Check if any races have been completed in the current year
    if get_completed_races(CURRENT_YEAR).empty:
        YEAR_TO_ANALYZE = CURRENT_YEAR - 1
        print(f"No completed races in {CURRENT_YEAR} yet. Analyzing the {YEAR_TO_ANALYZE} season.")
    
    analyze_race_start_performance(YEAR_TO_ANALYZE)
