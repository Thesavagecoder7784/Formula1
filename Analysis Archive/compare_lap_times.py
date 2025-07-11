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

def compare_drivers_lap_times(year, grand_prix, session_type, driver_codes):
    """
    Compares lap times between multiple drivers for a given race session,
    color-coded by tire compound.

    Args:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix (e.g., 'Belgian Grand Prix').
        session_type (str): The session type (e.g., 'Race').
        driver_codes (list): A list of three-letter driver codes (e.g., ['RUS', 'HAM']).
    """
    print(f"Loading data for {year} {grand_prix} - {session_type} for drivers {driver_codes}...")

    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load()

        all_drivers_laps = pd.DataFrame()

        for driver_code in driver_codes:
            driver_laps = session.laps.pick_driver(driver_code)

            valid_laps = driver_laps.loc[
                (driver_laps['IsAccurate'] == True) &
                (driver_laps['LapTime'].notna()) &
                (driver_laps['PitInTime'].isna()) &
                (driver_laps['PitOutTime'].isna())
            ].copy()

            if valid_laps.empty:
                print(f"No valid laps found for {driver_code} in {year} {grand_prix} {session_type}.")
                continue

            valid_laps['LapTime(s)'] = valid_laps['LapTime'].dt.total_seconds()
            valid_laps['Driver'] = driver_code # Add driver column for differentiation
            all_drivers_laps = pd.concat([all_drivers_laps, valid_laps])

        if all_drivers_laps.empty:
            print("No valid laps found for any of the specified drivers.")
            return

        plt.figure(figsize=(15, 8))
        sns.set_style("whitegrid")

        # Define a custom set of colors for drivers to ensure distinctness
        custom_driver_colors = {
            'HAM': '#00FFFF',  # Cyan for Hamilton
            'RUS': '#C0C0C0',  # Magenta for Russell
        }

        # Create a palette dictionary for the scatter plot
        scatter_palette = {driver: custom_driver_colors.get(driver, 'gray') for driver in all_drivers_laps['Driver'].unique()}

        sns.scatterplot(
            data=all_drivers_laps,
            x='LapNumber',
            y='LapTime(s)',
            hue='Driver',  # Color by driver
            style='Compound', # Use different shapes for each compound
            palette=scatter_palette,  # Use custom driver colors for scatter points
            s=100,
            alpha=0.8,
            edgecolor='w',
            linewidth=0.5
        )

        # Add lines for each driver's average lap time trend
        for driver in driver_codes:
            driver_data = all_drivers_laps[all_drivers_laps['Driver'] == driver]
            if not driver_data.empty:
                # Calculate rolling average for a smoother line
                driver_data = driver_data.sort_values(by='LapNumber')
                driver_data['RollingAvgLapTime'] = driver_data['LapTime(s)'].rolling(window=5, min_periods=1).mean()
                sns.lineplot(
                    data=driver_data,
                    x='LapNumber',
                    y='RollingAvgLapTime',
                    color=custom_driver_colors.get(driver, 'gray'),  # Use custom driver colors for lines
                    linestyle='--',
                    linewidth=2,
                    label=f'{driver} (Avg Trend)'
                )

        plt.title(f"Lap Time Comparison - {year} {grand_prix} {session_type}", fontsize=16)
        plt.xlabel("Lap Number", fontsize=12)
        plt.ylabel("Lap Time (Seconds)", fontsize=12)
        plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        # Print numerical summary for each driver
        for driver_code in driver_codes:
            driver_laps_summary = all_drivers_laps[all_drivers_laps['Driver'] == driver_code]
            if not driver_laps_summary.empty:
                avg_lap_times = driver_laps_summary.groupby('Compound')['LapTime(s)'].mean().reset_index()
                avg_lap_times['LapTime(s)'] = avg_lap_times['LapTime(s)'].apply(lambda x: f"{x:.3f}s")

                degradation_info = []
                for compound in driver_laps_summary['Compound'].unique():
                    compound_data = driver_laps_summary[driver_laps_summary['Compound'] == compound]
                    if len(compound_data) > 1:
                        slope, intercept = np.polyfit(compound_data['LapNumber'], compound_data['LapTime(s)'], 1)
                        degradation_info.append(f"{compound}: {slope:.3f} s/lap")

                print(f"\n--- Analysis Results for {driver_code} - {year} {grand_prix} {session_type} ---")
                print("Average Lap Times per Compound:")
                print(avg_lap_times.to_string(index=False))
                print("\nTire Degradation Rates (s/lap):")
                for info in degradation_info:
                    print(info)
            else:
                print(f"\n--- No valid laps to summarize for {driver_code} in {year} {grand_prix} {session_type} ---")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have an internet connection and the specified race/driver data is available.")
        print("You might also want to try a different year, grand prix, or driver.")
        print("If the error persists, check the exact Grand Prix name and driver codes.")


# --- Example Usage ---
if __name__ == "__main__":
    YEAR = 2024
    GRAND_PRIX = 'Belgian Grand Prix' # Spa-Francorchamps is the circuit, GP name is Belgian
    SESSION_TYPE = 'Race'
    DRIVER_CODES = ['RUS', 'HAM'] # George Russell and Lewis Hamilton

    compare_drivers_lap_times(YEAR, GRAND_PRIX, SESSION_TYPE, DRIVER_CODES)
