import fastf1
import fastf1.plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Enable the cache for faster data loading on subsequent runs
fastf1.Cache.enable_cache('cache/')

# Set up plotting style
fastf1.plotting.setup_mpl()

def get_race_pace_data(year, grand_prix, driver_code):
    """
    Helper function to get race pace for a single driver.
    Returns race_pace (median lap time) or None if data is not available.
    """
    try:
        race_session = fastf1.get_session(year, grand_prix, 'Race')
        race_session.load(telemetry=False, weather=False, messages=False)
        race_laps = race_session.laps.pick_driver(driver_code)

        valid_race_laps = race_laps.loc[
            (race_laps['IsAccurate'] == True) &
            (race_laps['LapTime'].notna()) &
            (race_laps['PitInTime'].isna()) &
            (race_laps['PitOutTime'].isna())
        ].copy()

        if not valid_race_laps.empty:
            valid_race_laps['LapTime(s)'] = valid_race_laps['LapTime'].dt.total_seconds()
            return valid_race_laps['LapTime(s)'].median()
        else:
            return None
    except Exception as e:
        return None

def get_consistency_tier(std_dev):
    """
    Categorizes a driver or constructor into a consistency tier based on standard deviation.
    """
    if std_dev < 0.5:
        return 'Tier 1 (Highly Consistent)'
    elif 0.5 <= std_dev < 0.75:
        return 'Tier 2 (Moderately Consistent)'
    else:
        return 'Tier 3 (Inconsistent)'

def analyze_consistency(year, driver_codes):
    """
    Analyzes the consistency of driver and constructor race pace throughout a season.

    Args:
        year (int): The year of the F1 season.
        driver_codes (list): A list of three-letter driver codes to analyze.
    """
    print(f"\n--- Analyzing Driver and Constructor Race Pace Consistency for {year} ---")

    schedule = fastf1.get_event_schedule(year)
    race_events = schedule.loc[schedule['EventFormat'] != 'testing']
    race_events = race_events.loc[race_events['EventFormat'] != 'practice']

    season_data = []

    for round_num, event in race_events.iterrows():
        grand_prix = event['EventName']
        print(f"Processing {grand_prix} (Round {round_num})...")

        try:
            race_session = fastf1.get_session(year, grand_prix, 'Race')
            race_session.load(telemetry=False, weather=False, messages=False)
            results = race_session.results
        except Exception as e:
            print(f"Could not load race session for {grand_prix}: {e}")
            continue

        all_race_paces = []
        driver_paces = {}
        for _, driver_row in results.iterrows():
            driver_code = driver_row['Abbreviation']
            pace = get_race_pace_data(year, grand_prix, driver_code)
            if pace is not None:
                all_race_paces.append(pace)
                driver_paces[driver_code] = {
                    'Pace': pace,
                    'Team': driver_row['TeamName']
                }
        
        if not all_race_paces:
            continue
            
        fastest_race_pace = min(all_race_paces)

        for driver_code, data in driver_paces.items():
            if driver_code in driver_codes:
                race_pace_rel = (data['Pace'] - fastest_race_pace) / fastest_race_pace * 100
                season_data.append({
                    'Round': round_num,
                    'GrandPrix': grand_prix,
                    'Driver': driver_code,
                    'Team': data['Team'],
                    'RacePaceRelative': race_pace_rel
                })

    season_df = pd.DataFrame(season_data)
    season_df.dropna(subset=['RacePaceRelative'], inplace=True)

    if season_df.empty:
        print("No sufficient data collected for the specified drivers across the season.")
        return

    # --- Driver Consistency Analysis ---
    print("\n--- Driver Race Pace Consistency Summary ---")
    driver_stats = season_df.groupby('Driver')['RacePaceRelative'].agg(['mean', 'std']).reset_index()
    driver_stats.rename(columns={'mean': 'MeanRelativePace', 'std': 'PaceStandardDeviation'}, inplace=True)
    driver_stats['ConsistencyTier'] = driver_stats['PaceStandardDeviation'].apply(get_consistency_tier)
    driver_stats = driver_stats.sort_values(by='PaceStandardDeviation', ascending=True)
    print(driver_stats)

    # --- Constructor Consistency Analysis ---
    print("\n--- Constructor Race Pace Consistency Summary ---")
    constructor_stats = season_df.groupby('Team')['RacePaceRelative'].agg(['mean', 'std']).reset_index()
    constructor_stats.rename(columns={'mean': 'MeanRelativePace', 'std': 'PaceStandardDeviation'}, inplace=True)
    constructor_stats['ConsistencyTier'] = constructor_stats['PaceStandardDeviation'].apply(get_consistency_tier)
    constructor_stats = constructor_stats.sort_values(by='PaceStandardDeviation', ascending=True)
    print(constructor_stats)

    # --- Visualizations ---
    # Driver Consistency
    plt.figure(figsize=(12, 8))
    sns.barplot(data=driver_stats, x='Driver', y='PaceStandardDeviation', hue='ConsistencyTier', palette='viridis', dodge=False)
    plt.title(f'Driver Race Pace Consistency - {year} Season', fontsize=16)
    plt.xlabel('Driver', fontsize=12)
    plt.ylabel('Standard Deviation of Relative Race Pace', fontsize=12)
    plt.legend(title='Consistency Tier')
    plt.tight_layout()
    plt.show()

    # Constructor Consistency
    plt.figure(figsize=(12, 8))
    sns.barplot(data=constructor_stats, x='Team', y='PaceStandardDeviation', hue='ConsistencyTier', palette='plasma', dodge=False)
    plt.title(f'Constructor Race Pace Consistency - {year} Season', fontsize=16)
    plt.xlabel('Constructor', fontsize=12)
    plt.ylabel('Standard Deviation of Relative Race Pace', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Consistency Tier')
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    YEAR = 2025
    DRIVER_CODES = ['VER', 'NOR', 'PIA', 'LEC', 'SAI', 'HAM', 'RUS', 'ALO', 'STR', 'TSU', 'RIC']
    analyze_consistency(YEAR, DRIVER_CODES)
