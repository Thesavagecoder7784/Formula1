import fastf1
import fastf1.plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Enable the cache for faster data loading on subsequent runs
fastf1.Cache.enable_cache('cache/')

# Set up plotting style
fastf1.plotting.setup_mpl()

def get_driver_paces(year, grand_prix, driver_code):
    """
    Helper function to get qualifying and race pace for a single driver.
    Returns a tuple (qualifying_pace, race_pace) or (None, None) if data is not available.
    """
    qualifying_pace = None
    race_pace = None

    try:
        # Load Qualifying session
        qualifying_session = fastf1.get_session(year, grand_prix, 'Qualifying')
        qualifying_session.load(telemetry=False, weather=False, messages=False)
        qualifying_laps = qualifying_session.laps.pick_driver(driver_code)
        if not qualifying_laps.empty:
            fastest_qualifying_lap = qualifying_laps.pick_fastest()
            if fastest_qualifying_lap is not None:
                qualifying_pace = fastest_qualifying_lap['LapTime'].total_seconds()

        # Load Race session
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
            race_pace = valid_race_laps['LapTime(s)'].median()

    except Exception as e:
        print(f"Could not load data for {driver_code} in {year} {grand_prix}: {e}")

    return qualifying_pace, race_pace

def get_most_recent_completed_race(year):
    """
    Finds the most recently completed race from the schedule.
    """
    schedule = fastf1.get_event_schedule(year)
    today = pd.to_datetime('today').date()
    # Filter for events that have already occurred
    completed_races = schedule[schedule['EventDate'].dt.date < today].copy()
    if completed_races.empty:
        return None, None
    # Get the last event from the filtered list
    most_recent_race = completed_races.iloc[-1]
    return most_recent_race['EventName'], most_recent_race['RoundNumber']

def analyze_quali_race_pace_delta(year, grand_prix):
    """
    Analyzes the percentage difference between qualifying and race pace for all drivers
    in a given Grand Prix.

    Args:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix.
    """
    print(f"\n--- Analyzing Qualifying vs. Race Pace Delta for {year} {grand_prix} ---")

    try:
        race_session = fastf1.get_session(year, grand_prix, 'Race')
        race_session.load(telemetry=False, weather=False, messages=False)
        driver_codes = race_session.results['Abbreviation'].tolist()
        print(f"Found {len(driver_codes)} drivers for the event.")
    except Exception as e:
        print(f"Could not load race session for {year} {grand_prix}: {e}")
        return

    results = []
    for driver_code in driver_codes:
        print(f"Fetching data for {driver_code}...")
        q_pace, r_pace = get_driver_paces(year, grand_prix, driver_code)

        if q_pace is not None and r_pace is not None:
            delta_percent = ((r_pace - q_pace) / q_pace) * 100
            results.append({
                'Driver': driver_code,
                'QualiPace': q_pace,
                'RacePace': r_pace,
                'DeltaPercent': delta_percent
            })
            print(f"{driver_code}: Quali={q_pace:.3f}s, Race={r_pace:.3f}s, Delta={delta_percent:.2f}%")
        else:
            print(f"Insufficient data for {driver_code}. Skipping.")

    if not results:
        print("No sufficient data to perform analysis.")
        return

    results_df = pd.DataFrame(results)
    results_df.sort_values(by='DeltaPercent', inplace=True)

    # Enhanced Insights
    average_delta = results_df['DeltaPercent'].mean()
    print(f"\n--- Insights (Average Delta: {average_delta:.2f}%) ---")
    for _, row in results_df.iterrows():
        driver = row['Driver']
        delta = row['DeltaPercent']
        if delta < average_delta - 1.5: # Significantly better than average
            print(f"* {driver} ({delta:.2f}%): Exceptional race pace compared to the field. Maintained speed much better than average.")
        elif delta > average_delta + 1.5: # Significantly worse than average
            print(f"* {driver} ({delta:.2f}%): Pace drop-off was much higher than average. Potential issues with setup or tire degradation.")
        else:
            print(f"* {driver} ({delta:.2f}%): Performance is in line with the field average.")

    # Visualization
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Driver', y='DeltaPercent', data=results_df, palette='coolwarm', hue='Driver', dodge=False)
    plt.axhline(average_delta, color='gold', linestyle='--', linewidth=1.2, label=f'Average Delta ({average_delta:.2f}%)')
    plt.title(f"Qualifying vs. Race Pace Delta ({year} {grand_prix})", fontsize=16)
    plt.xlabel("Driver", fontsize=12)
    plt.ylabel("Race Pace Delta from Quali Pace (%)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend().set_visible(False)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    CURRENT_YEAR = pd.to_datetime('today').year
    GRAND_PRIX, _ = get_most_recent_completed_race(CURRENT_YEAR)

    if GRAND_PRIX:
        analyze_quali_race_pace_delta(CURRENT_YEAR, GRAND_PRIX)
    else:
        # If no races are completed in the current year, analyze the last race of the previous year
        print(f"No completed races found for {CURRENT_YEAR}. Analyzing last race of {CURRENT_YEAR - 1}.")
        PREVIOUS_YEAR = CURRENT_YEAR - 1
        GRAND_PRIX, _ = get_most_recent_completed_race(PREVIOUS_YEAR)
        if GRAND_PRIX:
            analyze_quali_race_pace_delta(PREVIOUS_YEAR, GRAND_PRIX)
        else:
            print("Could not find any recent races to analyze.")
