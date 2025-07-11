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

def compare_tire_degradation(analysis_configs):
    """
    Compares and visualizes tire degradation for multiple drivers or races.

    Args:
        analysis_configs (list): A list of dictionaries, where each dictionary
                                 contains parameters for one analysis:
                                 - 'year' (int): The year of the Grand Prix.
                                 - 'grand_prix' (str): The name of the Grand Prix.
                                 - 'session_type' (str): The session type (e.g., 'Race').
                                 - 'driver_code' (str): The three-letter driver code.
    """
    num_plots = len(analysis_configs)
    if num_plots == 0:
        print("No analysis configurations provided.")
        return

    # Determine grid size for subplots
    rows = int(np.ceil(np.sqrt(num_plots)))
    cols = int(np.ceil(num_plots / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    for i, config in enumerate(analysis_configs):
        year = config['year']
        grand_prix = config['grand_prix']
        session_type = config['session_type']
        driver_code = config['driver_code']

        ax = axes[i] # Get the current subplot axis

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
                ax.set_title(f"No valid laps for {driver_code} in {year} {grand_prix}")
                ax.set_xlabel("Lap Number")
                ax.set_ylabel("Lap Time (Seconds)")
                print(f"No valid laps found for {driver_code} in {year} {grand_prix} {session_type}.")
                continue

            valid_laps['LapTime(s)'] = valid_laps['LapTime'].dt.total_seconds()

            avg_lap_times = valid_laps.groupby('Compound')['LapTime(s)'].mean().reset_index()
            avg_lap_times['LapTime(s)'] = avg_lap_times['LapTime(s)'].apply(lambda x: f"{x:.3f}s")

            plot_data = valid_laps[['LapNumber', 'LapTime(s)', 'Compound']].reset_index(drop=True)

            sns.scatterplot(
                data=plot_data,
                x='LapNumber',
                y='LapTime(s)',
                hue='Compound',
                palette=fastf1.plotting.COMPOUND_COLORS,
                s=50,
                alpha=0.8,
                edgecolor='w',
                linewidth=0.5,
                ax=ax # Specify the subplot axis
            )

            degradation_info = []
            for compound in plot_data['Compound'].unique():
                compound_data = plot_data[plot_data['Compound'] == compound]
                if len(compound_data) > 1:
                    slope, intercept = np.polyfit(compound_data['LapNumber'], compound_data['LapTime(s)'], 1)
                    degradation_info.append(f"{compound}: {slope:.3f} s/lap")
                    sns.regplot(
                        data=compound_data,
                        x='LapNumber',
                        y='LapTime(s)',
                        color=fastf1.plotting.COMPOUND_COLORS[compound],
                        scatter=False,
                        line_kws={'linestyle': '--', 'alpha': 0.7},
                        ci=None,
                        ax=ax # Specify the subplot axis
                    )

            if 'IsPitOut' in driver_laps.columns:
                pit_stops = driver_laps[driver_laps['IsPitOut'] == True]
                for _, pit_lap in pit_stops.iterrows():
                    ax.axvline(x=pit_lap['LapNumber'], color='red', linestyle=':', linewidth=1.0, label='Pit Stop' if 'Pit Stop' not in ax.get_legend_handles_labels()[1] else "")
                    ax.text(pit_lap['LapNumber'] + 0.5, ax.get_ylim()[1] * 0.95, 'Pit', rotation=90, verticalalignment='top', color='red', fontsize=8)
            else:
                print(f"'IsPitOut' column not found for {driver_code}. Skipping pit stop annotations.")

            ax.set_title(f"{driver_code} - {year} {grand_prix} {session_type}", fontsize=10)
            ax.set_xlabel("Lap Number", fontsize=9)
            ax.set_ylabel("Lap Time (Seconds)", fontsize=9)

            handles, labels = ax.get_legend_handles_labels()
            new_labels = []
            for label in labels:
                if label in avg_lap_times['Compound'].values:
                    avg_time = avg_lap_times[avg_lap_times['Compound'] == label]['LapTime(s)'].iloc[0]
                    new_labels.append(f"{label} (Avg: {avg_time})")
                else:
                    new_labels.append(label)

            degradation_text = "\n".join(degradation_info)
            if degradation_text:
                ax.text(1.05, 0.7, f"Degradation Rate (s/lap):\n{degradation_text}", transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

            ax.legend(handles, new_labels, title="Tire Compound", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.6)

            print(f"\n--- Analysis Results for {driver_code} - {year} {grand_prix} {session_type} ---")
            print("Average Lap Times per Compound:")
            print(avg_lap_times.to_string(index=False))
            print("\nTire Degradation Rates (s/lap):")
            for info in degradation_info:
                print(info)

        except Exception as e:
            ax.set_title(f"Error for {driver_code} - {year} {grand_prix}")
            ax.text(0.5, 0.5, f"Error: {e}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')
            print(f"An error occurred for {driver_code} in {year} {grand_prix}: {e}")
            print("Please ensure you have an internet connection and the specified race/driver data is available.")

    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout for overall figure
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    analysis_configs = [
        {'year': 2023, 'grand_prix': 'Italian Grand Prix', 'session_type': 'Race', 'driver_code': 'VER'},
        {'year': 2023, 'grand_prix': 'Italian Grand Prix', 'session_type': 'Race', 'driver_code': 'LEC'},
        {'year': 2022, 'grand_prix': 'Hungarian Grand Prix', 'session_type': 'Race', 'driver_code': 'HAM'},
        {'year': 2022, 'grand_prix': 'Hungarian Grand Prix', 'session_type': 'Race', 'driver_code': 'RUS'},
    ]

    compare_tire_degradation(analysis_configs)
