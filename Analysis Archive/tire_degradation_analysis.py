import fastf1
import fastf1.plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
fastf1.Cache.enable_cache('cache/')

fastf1.plotting.setup_mpl()

def analyze_tire_degradation(year, grand_prix, session_type, driver_code):
    """
    Analyzes and visualizes tire degradation for a specific driver during a race.

    Args:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix (e.g., 'Italian Grand Prix').
        session_type (str): The session type (e.g., 'Race').
        driver_code (str): The three-letter driver code (e.g., 'VER' for Verstappen).
    """
    print(f"Loading data for {year} {grand_prix} - {session_type} for driver {driver_code}...")

    try:
        # Load the session data
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load()

        # Get all laps for the specified driver
        driver_laps = session.laps.pick_driver(driver_code)

        # Filter out unwanted laps (pit-in/out, invalid, safety car, etc.)
        # We'll focus on 'Race' laps that are not 'Inlap' or 'Outlap' and are valid.
        # Also, ensure 'LapTime' is not NaT (Not a Time)
        valid_laps = driver_laps.loc[
            (driver_laps['IsAccurate'] == True) &
            (driver_laps['LapTime'].notna()) &
            (driver_laps['PitInTime'].isna()) &  # Exclude pit-in laps
            (driver_laps['PitOutTime'].isna())  # Exclude pit-out laps
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning

        if valid_laps.empty:
            print(f"No valid laps found for {driver_code} in {year} {grand_prix} {session_type}.")
            return

        # Convert LapTime to total seconds for plotting
        valid_laps['LapTime(s)'] = valid_laps['LapTime'].dt.total_seconds()

        # Calculate average lap times per compound
        # This provides a quick summary of performance on each tire type.
        avg_lap_times = valid_laps.groupby('Compound')['LapTime(s)'].mean().reset_index()
        avg_lap_times['LapTime(s)'] = avg_lap_times['LapTime(s)'].apply(lambda x: f"{x:.3f}s")

        # Prepare data for plotting
        plot_data = valid_laps[['LapNumber', 'LapTime(s)', 'Compound']].reset_index(drop=True)

        # Plotting
        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")  # Set a nice background style

        # Create a scatter plot with different colors for each compound
        sns.scatterplot(
            data=plot_data,
            x='LapNumber',
            y='LapTime(s)',
            hue='Compound',
            palette=fastf1.plotting.COMPOUND_COLORS,  # Use fastf1's predefined compound colors
            s=100,  # Marker size
            alpha=0.8,  # Transparency
            edgecolor='w',  # White edge for markers
            linewidth=0.5
        )

        # Add a regression line for each compound to show the trend (degradation)
        # This helps visualize the degradation rate for each tire type
        degradation_info = []
        for compound in plot_data['Compound'].unique():
            compound_data = plot_data[plot_data['Compound'] == compound]
            sns.regplot(
                data=compound_data,
                x='LapNumber',
                y='LapTime(s)',
                color=fastf1.plotting.COMPOUND_COLORS[compound],
                scatter=False,  # Don't plot scatter points again
                line_kws={'linestyle': '--', 'alpha': 0.7},  # Dashed line for trend
                ci=None  # Do not show confidence interval
            )
            # Calculate degradation rate (slope of the regression line)
            # The slope indicates how many seconds per lap the tire degrades.
            if len(compound_data) > 1:
                # Using numpy for polyfit to get the slope
                slope, intercept = np.polyfit(compound_data['LapNumber'], compound_data['LapTime(s)'], 1)
                degradation_info.append(f"{compound}: {slope:.3f} s/lap")

        # Annotate pit stops
        # Mark pit stops on the plot for better context of tire changes.
        if 'IsPitOut' in driver_laps.columns:
            pit_stops = driver_laps[driver_laps['IsPitOut'] == True]
            for _, pit_lap in pit_stops.iterrows():
                plt.axvline(x=pit_lap['LapNumber'], color='red', linestyle=':', linewidth=1.5, label='Pit Stop' if 'Pit Stop' not in plt.gca().get_legend_handles_labels()[1] else "")
                plt.text(pit_lap['LapNumber'] + 0.5, plt.ylim()[1] * 0.95, 'Pit', rotation=90, verticalalignment='top', color='red')
        else:
            print("'IsPitOut' column not found in data. Skipping pit stop annotations.")

        plt.title(f"{driver_code} Tire Degradation - {year} {grand_prix} {session_type}", fontsize=16)
        plt.xlabel("Lap Number", fontsize=12)
        plt.ylabel("Lap Time (Seconds)", fontsize=12)

        # Create custom legend for average lap times and degradation
        # This combines the compound legend with average lap times for each.
        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = []
        for label in labels:
            if label in avg_lap_times['Compound'].values:
                avg_time = avg_lap_times[avg_lap_times['Compound'] == label]['LapTime(s)'].iloc[0]
                new_labels.append(f"{label} (Avg: {avg_time})")
            else:
                new_labels.append(label)

        # Add degradation info to the legend or as a text box
        # Display the calculated degradation rates on the plot.
        degradation_text = "\n".join(degradation_info)
        if degradation_text:
            plt.text(1.05, 0.7, f"Degradation Rate (s/lap):\n{degradation_text}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        plt.legend(handles, new_labels, title="Tire Compound", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to prevent labels from overlapping with text box
        plt.show()

        print("\n--- Analysis Results ---")
        print("Average Lap Times per Compound:")
        print(avg_lap_times.to_string(index=False))
        print("\nTire Degradation Rates (s/lap):")
        for info in degradation_info:
            print(info)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have an internet connection and the specified race/driver data is available.")
        print("You might also want to try a different year, grand prix, or driver.")
        print("If the error persists, check the exact Grand Prix name and driver code.")


# --- Example Usage ---
# You can change these parameters to analyze different races and drivers
YEAR = 2023
GRAND_PRIX = 'Silverstone'  # Example: 'Monaco Grand Prix', 'British Grand Prix'
SESSION_TYPE = 'Race'
DRIVER_CODE = 'VER'  # Example: 'LEC', 'HAM', 'PER'

analyze_tire_degradation(YEAR, GRAND_PRIX, SESSION_TYPE, DRIVER_CODE)

# Another example (uncomment to run)
# analyze_tire_degradation(2022, 'Hungarian Grand Prix', 'Race', 'LEC')
