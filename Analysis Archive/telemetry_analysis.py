import fastf1
import fastf1.plotting
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import numpy as np

# Enable the cache for faster data loading on subsequent runs
fastf1.Cache.enable_cache('cache/')

# Set up plotting style
fastf1.plotting.setup_mpl()

def analyze_telemetry_comparison(year, grand_prix, session_type, driver_codes, corner_name, start_distance, end_distance):
    """
    Analyzes and visualizes telemetry data for two drivers through a specific corner,
    including a delta time plot and a track map visualization.

    Args:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix.
        session_type (str): The session type (e.g., 'Qualifying', 'Race').
        driver_codes (list): A list of two three-letter driver codes.
        corner_name (str): The name of the corner being analyzed.
        start_distance (float): The starting distance of the corner segment in meters.
        end_distance (float): The ending distance of the corner segment in meters.
    """
    if len(driver_codes) != 2:
        print("Error: This function is designed to compare exactly two drivers.")
        return

    print(f"Loading data for {year} {grand_prix} - {session_type} for drivers {driver_codes}...")

    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load(laps=True, telemetry=True, weather=False, messages=False)

        driver1_code, driver2_code = driver_codes[0], driver_codes[1]

        # Get fastest laps
        driver1_lap = session.laps.pick_driver(driver1_code).pick_fastest()
        driver2_lap = session.laps.pick_driver(driver2_code).pick_fastest()

        # Get telemetry
        driver1_tel = driver1_lap.get_telemetry().add_distance()
        driver2_tel = driver2_lap.get_telemetry().add_distance()
        
        # Get team colors
        driver1_color = fastf1.plotting.driver_color(driver1_code)
        driver2_color = fastf1.plotting.driver_color(driver2_code)

        # Filter telemetry for the corner
        tel1_segment = driver1_tel.loc[(driver1_tel['Distance'] >= start_distance) & (driver1_tel['Distance'] <= end_distance)]
        tel2_segment = driver2_tel.loc[(driver2_tel['Distance'] >= start_distance) & (driver2_tel['Distance'] <= end_distance)]

        # Calculate delta time
        delta_time, ref_tel, com_tel = fastf1.utils.delta_time(driver1_lap, driver2_lap)

        # --- Detailed Summary ---
        print(f"\n--- Telemetry Summary for {corner_name} ({start_distance:.0f}m - {end_distance:.0f}m) ---")
        for i, tel_segment in enumerate([tel1_segment, tel2_segment]):
            driver_code = driver_codes[i]
            apex_speed = tel_segment['Speed'].min()
            time_to_apex = tel_segment[tel_segment['Speed'] == apex_speed]['Time'].iloc[0] - tel_segment['Time'].iloc[0]
            exit_speed = tel_segment['Speed'].iloc[-1]
            print(f"\nDriver: {driver_code}")
            print(f"  Apex Speed: {apex_speed:.2f} km/h")
            print(f"  Time to Apex: {time_to_apex.total_seconds()*1000:.0f} ms")
            print(f"  Exit Speed: {exit_speed:.2f} km/h")

        # --- Plotting ---
        fig = plt.figure(figsize=(15, 12), constrained_layout=True)
        gs = fig.add_gridspec(4, 2)

        # Track Map
        ax_track = fig.add_subplot(gs[:, 1])
        ax_track.plot(driver1_tel['X'], driver1_tel['Y'], color='grey', linewidth=2)
        ax_track.plot(tel1_segment['X'], tel1_segment['Y'], color=driver1_color, linewidth=3, label=driver1_code)
        ax_track.plot(tel2_segment['X'], tel2_segment['Y'], color=driver2_color, linewidth=3, label=driver2_code)
        ax_track.set_title(f'{corner_name} on Track')
        ax_track.set_xlabel('X (m)')
        ax_track.set_ylabel('Y (m)')
        ax_track.legend()
        ax_track.axis('equal')

        # Speed Plot
        ax_speed = fig.add_subplot(gs[0, 0])
        ax_speed.plot(tel1_segment['Distance'], tel1_segment['Speed'], color=driver1_color, label=driver1_code)
        ax_speed.plot(tel2_segment['Distance'], tel2_segment['Speed'], color=driver2_color, label=driver2_code)
        ax_speed.set_ylabel('Speed (km/h)')
        ax_speed.legend()

        # Throttle/Brake Plot
        ax_inputs = fig.add_subplot(gs[1, 0], sharex=ax_speed)
        ax_inputs.plot(tel1_segment['Distance'], tel1_segment['Throttle'], color=driver1_color, label=f'{driver1_code} Throttle')
        ax_inputs.plot(tel1_segment['Distance'], tel1_segment['Brake'] * 100, '--', color=driver1_color, label=f'{driver1_code} Brake')
        ax_inputs.plot(tel2_segment['Distance'], tel2_segment['Throttle'], color=driver2_color, label=f'{driver2_code} Throttle')
        ax_inputs.plot(tel2_segment['Distance'], tel2_segment['Brake'] * 100, '--', color=driver2_color, label=f'{driver2_code} Brake')
        ax_inputs.set_ylabel('Input (%)')
        ax_inputs.legend()

        # Gear Plot
        ax_gear = fig.add_subplot(gs[2, 0], sharex=ax_speed)
        ax_gear.plot(tel1_segment['Distance'], tel1_segment['nGear'], color=driver1_color, label=driver1_code)
        ax_gear.plot(tel2_segment['Distance'], tel2_segment['nGear'], color=driver2_color, label=driver2_code)
        ax_gear.set_ylabel('Gear')

        # Delta Time Plot
        ax_delta = fig.add_subplot(gs[3, 0], sharex=ax_speed)
        ax_delta.plot(ref_tel['Distance'], delta_time, color='white')
        ax_delta.axhline(0, color='grey', linestyle='--')
        ax_delta.set_ylabel(f"<-- {driver2_code} ahead | {driver1_code} ahead -->")
        ax_delta.set_xlabel('Distance (m)')

        fig.suptitle(f"Telemetry Comparison - {year} {grand_prix} {session_type} - {corner_name}", fontsize=18)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    YEAR = 2021
    GRAND_PRIX = 'Abu Dhabi Grand Prix'
    SESSION_TYPE = 'Race'
    DRIVER_CODES = ['HAM', 'VER'] # Compare Hamilton and Verstappen
    CORNER_NAME = "Turn 5/6/7 Complex"
    START_DISTANCE = 400  # Approximate start of the complex
    END_DISTANCE = 1000    # Approximate end of the complex

    analyze_telemetry_comparison(YEAR, GRAND_PRIX, SESSION_TYPE, DRIVER_CODES, CORNER_NAME, START_DISTANCE, END_DISTANCE)
