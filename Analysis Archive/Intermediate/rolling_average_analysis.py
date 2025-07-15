import fastf1
import fastf1.plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Enable the cache for faster data loading on subsequent runs
fastf1.Cache.enable_cache('cache/')

# Set up plotting style
fastf1.plotting.setup_mpl()

def analyze_rolling_average_lap_times(year, grand_prix, session_type, driver_code, window_size=5):
    """
    Calculates and visualizes the rolling average of lap times for a single driver.

    Args:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix (e.g., 'Las Vegas Grand Prix').
        session_type (str): The session type (e.g., 'Race').
        driver_code (str): The three-letter driver code (e.g., 'NOR').
        window_size (int): The size of the rolling window for calculating the average.
    """
    print(f"Loading data for {year} {grand_prix} - {session_type} for driver {driver_code}...")

    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load()

        driver_laps = session.laps.pick_driver(driver_code)

        valid_laps = driver_laps.loc[
            (driver_laps['IsAccurate'] == True) &
            (driver_laps['LapTime'].notna()) &
            (driver_laps['PitInTime'].isna()) &
            (driver_laps['PitOutTime'].isna())
        ].copy()

        if valid_laps.empty:
            print(f"No valid laps found for {driver_code} in {year} {grand_prix} {session_type}.")
            return

        valid_laps['LapTime(s)'] = valid_laps['LapTime'].dt.total_seconds()

        # Calculate rolling average
        valid_laps['RollingAvgLapTime'] = valid_laps['LapTime(s)'].rolling(window=window_size, min_periods=1).mean()

        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")

        # Plot individual lap times
        sns.scatterplot(
            data=valid_laps,
            x='LapNumber',
            y='LapTime(s)',
            hue='Compound',
            palette=fastf1.plotting.COMPOUND_COLORS,
            s=100,
            alpha=0.7,
            edgecolor='w',
            linewidth=0.5
        )

        # Plot rolling average
        sns.lineplot(
            data=valid_laps,
            x='LapNumber',
            y='RollingAvgLapTime',
            color='black',
            linestyle='-',
            linewidth=2,
            label=f'Rolling Average ({window_size}-lap)'
        )

        plt.title(f"{driver_code} Lap Times with Rolling Average - {year} {grand_prix} {session_type}", fontsize=16)
        plt.xlabel("Lap Number", fontsize=12)
        plt.ylabel("Lap Time (Seconds)", fontsize=12)
        plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        # Display numerical data
        print("\n--- Rolling Average Analysis Results ---")
        print(f"Driver: {driver_code}, Grand Prix: {grand_prix}, Session: {session_type}")
        print(f"Rolling Average Window Size: {window_size} laps")

        # Overall average lap time
        overall_avg_lap_time = valid_laps['LapTime(s)'].mean()
        print(f"\nOverall Average Lap Time: {overall_avg_lap_time:.3f}s")

        # Average lap times per compound
        avg_lap_times_compound = valid_laps.groupby('Compound')['LapTime(s)'].mean().reset_index()
        avg_lap_times_compound['LapTime(s)'] = avg_lap_times_compound['LapTime(s)'].apply(lambda x: f"{x:.3f}s")
        print("\nAverage Lap Times per Compound:")
        print(avg_lap_times_compound.to_string(index=False))

        # Tire Degradation Rates per Compound
        degradation_info = []
        for compound in valid_laps['Compound'].unique():
            compound_data = valid_laps[valid_laps['Compound'] == compound]
            if len(compound_data) > 1:
                slope, intercept = np.polyfit(compound_data['LapNumber'], compound_data['LapTime(s)'], 1)
                degradation_info.append(f"{compound}: {slope:.3f} s/lap")
        if degradation_info:
            print("\nTire Degradation Rates (s/lap):")
            for info in degradation_info:
                print(info)
        else:
            print("\nNo sufficient data to calculate tire degradation rates per compound.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have an internet connection and the specified race/driver data is available.")
        print("You might also want to try a different year, grand prix, or driver.")
        print("If the error persists, check the exact Grand Prix name and driver code.")


# --- Example Usage ---
if __name__ == "__main__":
    YEAR = 2024
    GRAND_PRIX = 'Las Vegas Grand Prix'
    SESSION_TYPE = 'Race'
    DRIVER_CODE = 'NOR' # Lando Norris
    WINDOW_SIZE = 5 # 5-lap rolling average

    analyze_rolling_average_lap_times(YEAR, GRAND_PRIX, SESSION_TYPE, DRIVER_CODE, WINDOW_SIZE)
