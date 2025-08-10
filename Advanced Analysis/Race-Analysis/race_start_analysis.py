import fastf1
import fastf1.plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Enable the cache for faster data loading on subsequent runs
fastf1.Cache.enable_cache('cache/')

# Set up plotting style
fastf1.plotting.setup_mpl()

def analyze_race_start_performance(year, driver_codes):
    """
    Analyzes how many positions each driver gains or loses on the first lap of a race
    across a season.

    Args:
        year (int): The year of the F1 season.
        driver_codes (list): A list of three-letter driver codes to analyze.
    """
    print(f"\n--- Analyzing Race Start Performance for {year} ---")

    schedule = fastf1.get_event_schedule(year)
    today = pd.to_datetime('today').date()
    race_events = schedule.loc[(schedule['EventFormat'] == 'race') & (schedule['EventDate'].dt.date < today)]

    print(f"Found {len(race_events)} completed race(s) to analyze for the {year} season.")

    all_start_data = []

    for _, event in race_events.iterrows():
        grand_prix = event['EventName']
        round_num = event['RoundNumber']
        print(f"\nProcessing {grand_prix} (Round {round_num})...")

        try:
            race = fastf1.get_session(year, grand_prix, 'Race')
            race.load(telemetry=False, weather=False, messages=False)
            print(f"Successfully loaded session data for {grand_prix}.")

            laps = race.laps
            if laps.empty:
                print("No lap data available for this session.")
                continue

            grid_positions = race.results[['Abbreviation', 'GridPosition']].set_index('Abbreviation')
            lap1 = laps.loc[laps['LapNumber'] == 1]
            lap1_positions = lap1[['Abbreviation', 'Position']].set_index('Abbreviation')

            print(f"Found {len(grid_positions)} drivers on the grid.")
            print(f"Found {len(lap1_positions)} drivers who completed Lap 1.")

            for driver_code in driver_codes:
                start_pos = grid_positions.loc[driver_code]['GridPosition'] if driver_code in grid_positions.index else None
                end_lap1_pos = lap1_positions.loc[driver_code]['Position'] if driver_code in lap1_positions.index else None

                if start_pos is not None and end_lap1_pos is not None and start_pos > 0:
                    positions_gained = start_pos - end_lap1_pos
                    all_start_data.append({
                        'Round': round_num,
                        'GrandPrix': grand_prix,
                        'Driver': driver_code,
                        'StartPosition': start_pos,
                        'EndLap1Position': end_lap1_pos,
                        'PositionsGained': positions_gained
                    })
                    print(f"  -> Recorded data for {driver_code}: Start={start_pos}, EndLap1={end_lap1_pos}, Gained={positions_gained}")

        except Exception as e:
            print(f"An error occurred loading data for {grand_prix}: {e}")

    if not all_start_data:
        print("\nNo sufficient race start data collected for the specified drivers across the season.")
        return
        
    start_df = pd.DataFrame(all_start_data)
    start_df.dropna(inplace=True)

    plt.figure(figsize=(15, 7))
    sns.lineplot(data=start_df, x='Round', y='PositionsGained', hue='Driver', marker='o')
    plt.title(f'Positions Gained/Lost on Lap 1 - {year} Season', fontsize=16)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Positions Gained (Positive) / Lost (Negative)', fontsize=12)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xticks(start_df['Round'].unique())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Driver', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    print("\n--- Season Summary of Race Start Performance ---")
    summary = start_df.groupby('Driver')['PositionsGained'].agg(['mean', 'sum']).reset_index()
    summary.columns = ['Driver', 'AvgPositionsGained', 'TotalPositionsGained']
    
    for index, row in summary.iterrows():
        print(f"\n{row['Driver']}:")
        print(f"  Average Positions Gained/Lost on Lap 1: {row['AvgPositionsGained']:.2f}")
        print(f"  Total Positions Gained/Lost on Lap 1 (Season): {row['TotalPositionsGained']:.0f}")


# --- Example Usage ---
if __name__ == "__main__":
    CURRENT_YEAR = 2024
    print(f"Current year is {CURRENT_YEAR}. Determining season to analyze...")
    
    schedule = fastf1.get_event_schedule(CURRENT_YEAR)
    if schedule.loc[schedule['EventDate'].dt.date < pd.to_datetime('today').date()].empty:
        YEAR_TO_ANALYZE = CURRENT_YEAR - 1
        print(f"No completed races in {CURRENT_YEAR} yet. Analyzing the {YEAR_TO_ANALYZE} season.")
    else:
        YEAR_TO_ANALYZE = CURRENT_YEAR
        print(f"Found completed races in {CURRENT_YEAR}. Analyzing the current season.")

    DRIVER_CODES = ['VER', 'HAM', 'LEC', 'RUS', 'PER', 'NOR', 'SAI']

    analyze_race_start_performance(YEAR_TO_ANALYZE, DRIVER_CODES)

