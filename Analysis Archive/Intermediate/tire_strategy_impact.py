import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import numpy as np

# Enable the cache for faster data loading on subsequent runs
fastf1.Cache.enable_cache('cache/')

# Set up plotting style
fastf1.plotting.setup_mpl()

def plot_strategy_dashboard(laps, stints, year, grand_prix, session_type):
    """Plots a comprehensive dashboard of the tire strategy for each driver on a single graph."""
    drivers = laps['Driver'].unique()
    
    fig, ax = plt.subplots(figsize=(16, 10))

    # Define driver colors and compound markers
    driver_colors = plt.cm.get_cmap('tab10', len(drivers))
    compound_markers = {'SOFT': 'o', 'MEDIUM': 's', 'HARD': '^'}

    for i, driver in enumerate(drivers):
        driver_laps = laps[laps['Driver'] == driver].copy()
        
        # Lap times and rolling average
        driver_laps['LapTime(s)'] = driver_laps['LapTime'].dt.total_seconds()
        driver_laps['RollingAvg'] = driver_laps['LapTime(s)'].rolling(window=3, min_periods=1).mean()

        # Plotting lap times with compound markers
        for compound, compound_laps in driver_laps.groupby('Compound'):
            ax.plot(compound_laps['LapNumber'], compound_laps['RollingAvg'], 
                    color=driver_colors(i),
                    marker=compound_markers.get(compound, 'x'), 
                    linestyle='-',
                    label=f'{driver} - {compound}')

        # Pit stop markers
        pit_stops = driver_laps[driver_laps['PitOutTime'].notna()]
        for _, pit_stop in pit_stops.iterrows():
            ax.axvline(pit_stop['LapNumber'], color='red', linestyle='--', alpha=0.5)
            ax.text(pit_stop['LapNumber'], ax.get_ylim()[0], f'{driver} Pit', color='red', 
                    horizontalalignment='center', verticalalignment='bottom', fontsize=8)

    ax.set_title(f'Tire Strategy Comparison - {year} {grand_prix} {session_type}')
    ax.set_xlabel('Lap Number')
    ax.set_ylabel('Lap Time (s) - Rolling Average')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

    # Stint summary tables
    for i, driver in enumerate(drivers):
        driver_stints = stints[stints['Driver'] == driver].copy()
        stint_summary = driver_stints[['StintNumber', 'Compound', 'StintLength', 'AvgLapTime(s)']]
        stint_summary['AvgLapTime(s)'] = stint_summary['AvgLapTime(s)'].round(3)
        
        table_pos = [0.1 + i * 0.4, -0.2, 0.3, 0.15] # Position tables side-by-side
        table = ax.table(cellText=stint_summary.values, colLabels=stint_summary.columns, loc='bottom', 
                         cellLoc='center', bbox=table_pos)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax.text(table_pos[0] + 0.15, -0.15, f'{driver} Stint Summary', horizontalalignment='center', fontsize=10)


    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()

def analyze_tire_strategy_impact(year, grand_prix, session_type, driver_codes):
    """
    Analyzes the impact of different tire strategies on overall race performance
    for multiple drivers.

    Args:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix.
        session_type (str): The session type (e.g., 'Race').
        driver_codes (list): A list of three-letter driver codes to compare.
    """
    print(f"Loading data for {year} {grand_prix} - {session_type} for drivers {driver_codes}...")

    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)

        laps = session.laps.pick_drivers(driver_codes)
        laps['LapTime(s)'] = laps['LapTime'].dt.total_seconds()

        stints = laps.groupby(['Driver', 'Stint']).agg(
            Compound=('Compound', 'first'),
            LapStart=('LapNumber', 'min'),
            LapEnd=('LapNumber', 'max'),
            StintLength=('LapNumber', 'count')
        ).reset_index().rename(columns={'Stint': 'StintNumber'})

        avg_lap_times = []
        for _, stint in stints.iterrows():
            stint_laps = laps[
                (laps['Driver'] == stint['Driver']) &
                (laps['Stint'] == stint['StintNumber']) &
                (laps['IsAccurate'] == True) &
                (laps['LapTime'].notna())
            ]
            if not stint_laps.empty:
                avg_time = stint_laps['LapTime(s)'].mean()
                avg_lap_times.append(avg_time)
            else:
                avg_lap_times.append(np.nan)

        stints['AvgLapTime(s)'] = avg_lap_times
        stints.dropna(subset=['AvgLapTime(s)'], inplace=True)

        if stints.empty:
            print("No valid stint data found for any of the specified drivers.")
            return

        # --- Visualization ---
        plot_strategy_dashboard(laps, stints, year, grand_prix, session_type)

        # --- Numerical Summary ---
        print(f"\n--- Tire Strategy Impact Analysis Results - {year} {grand_prix} {session_type} ---")
        for driver_code in driver_codes:
            print(f"\nDriver: {driver_code}")
            driver_stints_summary = stints[stints['Driver'] == driver_code]
            if not driver_stints_summary.empty:
                print(driver_stints_summary[['StintNumber', 'Compound', 'StintLength', 'AvgLapTime(s)']].to_string(index=False))
            else:
                print("No stint data.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have an internet connection and the specified race/driver data is available.")
        print("You might also want to try a different year, grand prix, session type, or driver.")
        print("If the error persists, check the exact Grand Prix name and driver codes.")


# --- Example Usage ---
if __name__ == "__main__":
    YEAR = 2025
    GRAND_PRIX = 'Hungarian Grand Prix'
    SESSION_TYPE = 'Race'
    DRIVER_CODES = ['NOR', 'PIA'] 

    analyze_tire_strategy_impact(YEAR, GRAND_PRIX, SESSION_TYPE, DRIVER_CODES)
