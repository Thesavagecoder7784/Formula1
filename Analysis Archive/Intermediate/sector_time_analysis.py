import fastf1
import fastf1.plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.collections import LineCollection

# Enable the cache for faster data loading on subsequent runs
fastf1.Cache.enable_cache('cache/')

# Set up plotting style
fastf1.plotting.setup_mpl()

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

def analyze_sector_times(year, grand_prix, session_type, driver_codes: list[str]):
    """
    Analyzes and visualizes sector times for a specific driver over a session,
    including a track map colored by speed and marked with sector boundaries.

    Args:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix (e.g., 'Monaco Grand Prix').
        session_type (str): The session type (e.g., 'Race', 'Qualifying').
        driver_codes (list[str]): A list of three-letter driver codes (e.g., ['VER', 'LEC']).
    """
    print(f"Loading data for {year} {grand_prix} - {session_type} for drivers {', '.join(driver_codes)}...")

    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load()

        all_drivers_laps = pd.DataFrame()
        fastest_laps_telemetry = {}

        for driver_code in driver_codes:
            driver_laps = session.laps.pick_driver(driver_code)

            # Filter for valid laps (accurate, not pit in/out, and with lap time)
            valid_laps = driver_laps.loc[
                (driver_laps['IsAccurate'] == True) &
                (driver_laps['LapTime'].notna()) &
                (driver_laps['PitInTime'].isna()) &
                (driver_laps['PitOutTime'].isna())
            ].copy()

            if valid_laps.empty:
                print(f"No valid laps found for {driver_code} in {year} {grand_prix} {session_type}. Skipping this driver.")
                continue

            # Add driver code to the valid_laps DataFrame for later filtering
            valid_laps['Driver'] = driver_code
            all_drivers_laps = pd.concat([all_drivers_laps, valid_laps])

            # Pick the fastest lap for detailed telemetry analysis for each driver
            fastest_lap = valid_laps.pick_fastest()
            telemetry = fastest_lap.get_telemetry().add_distance()
            fastest_laps_telemetry[driver_code] = {'fastest_lap': fastest_lap, 'telemetry': telemetry}

        if all_drivers_laps.empty:
            print(f"No valid laps found for any of the specified drivers in {year} {grand_prix} {session_type}.")
            return

        # --- Track Map Visualization (for each driver in the list) ---
        for driver_code, fastest_lap_data in fastest_laps_telemetry.items():
            fastest_lap = fastest_lap_data['fastest_lap']
            telemetry = fastest_lap_data['telemetry']

            min_speed = telemetry['Speed'].min()
            max_speed = telemetry['Speed'].max()

            x = telemetry['X']
            y = telemetry['Y']

            sector1_end_time = fastest_lap['Sector1SessionTime']
            sector2_end_time = fastest_lap['Sector2SessionTime']
            sector3_end_time = fastest_lap['LapStartTime'] + fastest_lap['LapTime']

            telemetry['TimeDelta'] = telemetry['Time']

            idx_s1 = (telemetry['TimeDelta'] - sector1_end_time).abs().idxmin()
            idx_s2 = (telemetry['TimeDelta'] - sector2_end_time).abs().idxmin()
            idx_s3 = (telemetry['TimeDelta'] - sector3_end_time).abs().idxmin()

            s1_x, s1_y = telemetry.loc[idx_s1, ['X', 'Y']]
            s2_x, s2_y = telemetry.loc[idx_s2, ['X', 'Y']]
            s3_x, s3_y = telemetry.loc[idx_s3, ['X', 'Y']]

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect('equal', adjustable='box')

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(min_speed, max_speed)
            lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=3)
            lc.set_array(telemetry['Speed'])
            line = ax.add_collection(lc)

            cbar = fig.colorbar(line, ax=ax)
            cbar.set_label('Speed (km/h)')

            ax.plot(s1_x, s1_y, 'o', color='red', markersize=8, label='Sector 1 End')
            ax.plot(s2_x, s2_y, 'o', color='orange', markersize=8, label='Sector 2 End')
            ax.plot(s3_x, s3_y, 'o', color='purple', markersize=8, label='Lap End (Sector 3 End)')

            ax.set_title(f"{driver_code} Fastest Lap Speed & Sector Boundaries - {year} {grand_prix} {session_type}", fontsize=14)
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
        
        if not fastest_laps_telemetry:
            print("Skipping track map visualization as no valid data for any driver was found.")

        # Numerical summary and new plots
        # Extract sector times for all drivers
        all_drivers_laps['Sector1(s)'] = all_drivers_laps['Sector1Time'].dt.total_seconds()
        all_drivers_laps['Sector2(s)'] = all_drivers_laps['Sector2Time'].dt.total_seconds()
        all_drivers_laps['Sector3(s)'] = all_drivers_laps['Sector3Time'].dt.total_seconds()

        print(f"\n--- Sector Time Analysis Results for {', '.join(driver_codes)} - {year} {grand_prix} {session_type} ---")
        for driver_code in driver_codes:
            driver_data = all_drivers_laps[all_drivers_laps['Driver'] == driver_code]
            if not driver_data.empty:
                print(f"\n--- Driver: {driver_code} ---")
                for sector in ['Sector1(s)', 'Sector2(s)', 'Sector3(s)']:
                    avg_time = driver_data[sector].mean()
                    print(f"Average {sector}: {avg_time:.3f}s")

                # Plotting Sector Times per Lap for all drivers (combined)
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        sectors_data = ['Sector1(s)', 'Sector2(s)', 'Sector3(s)']
        sector_titles = ['Sector 1', 'Sector 2', 'Sector 3']

        for i, sector_col in enumerate(sectors_data):
            sns.lineplot(x='LapNumber', y=sector_col, hue='Driver', data=all_drivers_laps, ax=axes2[i], marker='o', linestyle='-')
            axes2[i].set_title(f'{sector_titles[i]} Times per Lap')
            axes2[i].set_xlabel('Lap Number')
            axes2[i].set_ylabel('Time (s)')
            axes2[i].grid(True, linestyle='--', alpha=0.6)
            axes2[i].legend(title='Driver')
        plt.suptitle(f'Sector Times Progression - {year} {grand_prix} {session_type}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Calculate and print Ideal Lap Time
        ideal_lap_time = driver_data['Sector1(s)'].min() + \
                            driver_data['Sector2(s)'].min() + \
                            driver_data['Sector3(s)'].min()
        print(f"Ideal Lap Time: {ideal_lap_time:.3f}s")

        # Calculate and print Delta to Fastest Sector (across all drivers)
        for i, sector in enumerate(['Sector1(s)', 'Sector2(s)', 'Sector3(s)']):
            fastest_sector_time = all_drivers_laps[sector].min()
            delta = driver_data[sector].mean() - fastest_sector_time
            print(f"Delta to Fastest {sector.replace('(s)', '')}: {delta:.3f}s")

        # Calculate and print Average Speed per Sector (from fastest lap)
        if driver_code in fastest_laps_telemetry:
            fastest_lap_data = fastest_laps_telemetry[driver_code]
            fastest_lap = fastest_lap_data['fastest_lap']
            telemetry = fastest_lap_data['telemetry']

            # This is an approximation, as exact sector boundaries in telemetry are not trivial
            # We'll use the time deltas to approximate the telemetry points within each sector
            sector1_telemetry = telemetry[(telemetry['TimeDelta'] >= fastest_lap['LapStartTime']) & (telemetry['TimeDelta'] < fastest_lap['Sector1SessionTime'])]
            sector2_telemetry = telemetry[(telemetry['TimeDelta'] >= fastest_lap['Sector1SessionTime']) & (telemetry['TimeDelta'] < fastest_lap['Sector2SessionTime'])]
            sector3_telemetry = telemetry[(telemetry['TimeDelta'] >= fastest_lap['Sector2SessionTime']) & (telemetry['TimeDelta'] < fastest_lap['LapStartTime'] + fastest_lap['LapTime'])]

            if not sector1_telemetry.empty:
                print(f"Average Speed Sector 1 (Fastest Lap): {sector1_telemetry['Speed'].mean():.2f} km/h")
            if not sector2_telemetry.empty:
                print(f"Average Speed Sector 2 (Fastest Lap): {sector2_telemetry['Speed'].mean():.2f} km/h")
            if not sector3_telemetry.empty:
                print(f"Average Speed Sector 3 (Fastest Lap): {sector3_telemetry['Speed'].mean():.2f} km/h")

        # Calculate and print Delta to Fastest Sector (across all drivers)
        for i, sector in enumerate(['Sector1(s)', 'Sector2(s)', 'Sector3(s)']):
            fastest_sector_time = all_drivers_laps[sector].min()
            delta = driver_data[sector].mean() - fastest_sector_time
            print(f"Delta to Fastest {sector.replace('(s)', '')}: {delta:.3f}s")

        # Calculate and print Average Speed per Sector (from fastest lap)
        if driver_code in fastest_laps_telemetry:
            fastest_lap_data = fastest_laps_telemetry[driver_code]
            fastest_lap = fastest_lap_data['fastest_lap']
            telemetry = fastest_lap_data['telemetry']

            # This is an approximation, as exact sector boundaries in telemetry are not trivial
            # We'll use the time deltas to approximate the telemetry points within each sector
            sector1_telemetry = telemetry[(telemetry['TimeDelta'] >= fastest_lap['LapStartTime']) & (telemetry['TimeDelta'] < fastest_lap['Sector1SessionTime'])]
            sector2_telemetry = telemetry[(telemetry['TimeDelta'] >= fastest_lap['Sector1SessionTime']) & (telemetry['TimeDelta'] < fastest_lap['Sector2SessionTime'])]
            sector3_telemetry = telemetry[(telemetry['TimeDelta'] >= fastest_lap['Sector2SessionTime']) & (telemetry['TimeDelta'] < fastest_lap['LapStartTime'] + fastest_lap['LapTime'])]

            if not sector1_telemetry.empty:
                print(f"Average Speed Sector 1 (Fastest Lap): {sector1_telemetry['Speed'].mean():.2f} km/h")
            if not sector2_telemetry.empty:
                print(f"Average Speed Sector 2 (Fastest Lap): {sector2_telemetry['Speed'].mean():.2f} km/h")
            if not sector3_telemetry.empty:
                print(f"Average Speed Sector 3 (Fastest Lap): {sector3_telemetry['Speed'].mean():.2f} km/h")

        # Plotting Average Sector Times Comparison for all drivers
        avg_sector_times_df = all_drivers_laps.groupby('Driver')[['Sector1(s)', 'Sector2(s)', 'Sector3(s)']].mean().reset_index()
        avg_sector_times_df_melted = avg_sector_times_df.melt(id_vars='Driver', var_name='Sector', value_name='Average Time (s)')

        fig3, ax3 = plt.subplots(figsize=(12, 7))
        sns.barplot(x='Sector', y='Average Time (s)', hue='Driver', data=avg_sector_times_df_melted, ax=ax3, palette='deep')
        ax3.set_title(f'Average Sector Times Comparison - {year} {grand_prix} {session_type}')
        ax3.set_xlabel('Sector')
        ax3.set_ylabel('Average Time (s)')
        ax3.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Plotting Sector Time Consistency (Box Plots) for all drivers
        fig4, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        sectors = ['Sector1(s)', 'Sector2(s)', 'Sector3(s)']
        titles = ['Sector 1 Consistency', 'Sector 2 Consistency', 'Sector 3 Consistency']

        for i, sector in enumerate(sectors):
            sns.boxplot(x='Driver', y=sector, data=all_drivers_laps, ax=axes[i], palette='pastel')
            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Driver')
            axes[i].set_ylabel('Time (s)')
            axes[i].grid(axis='y', linestyle='--', alpha=0.6)
        plt.suptitle(f'Sector Time Consistency - {year} {grand_prix} {session_type}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap

        # Plotting Delta to Fastest Sector
        delta_to_fastest_df = pd.DataFrame()
        for driver_code in driver_codes:
            driver_data = all_drivers_laps[all_drivers_laps['Driver'] == driver_code]
            if not driver_data.empty:
                for sector_col in ['Sector1(s)', 'Sector2(s)', 'Sector3(s)']:
                    fastest_sector_time = all_drivers_laps[sector_col].min()
                    avg_driver_sector_time = driver_data[sector_col].mean()
                    delta = avg_driver_sector_time - fastest_sector_time
                    delta_to_fastest_df = pd.concat([
                        delta_to_fastest_df,
                        pd.DataFrame([{'Driver': driver_code, 'Sector': sector_col.replace('(s)', ''), 'Delta': delta}])
                    ])

        if not delta_to_fastest_df.empty:
            fig6, ax6 = plt.subplots(figsize=(12, 7))
            sns.barplot(x='Sector', y='Delta', hue='Driver', data=delta_to_fastest_df, ax=ax6, palette='coolwarm')
            ax6.set_title(f'Average Delta to Fastest Sector - {year} {grand_prix} {session_type}')
            ax6.set_xlabel('Sector')
            ax6.set_ylabel('Delta to Fastest (s)')
            ax6.axhline(0, color='grey', linestyle='--')
            ax6.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()

        # Plotting Average Speed per Sector
        avg_speed_df = pd.DataFrame()
        for driver_code in driver_codes:
            if driver_code in fastest_laps_telemetry:
                fastest_lap_data = fastest_laps_telemetry[driver_code]
                fastest_lap = fastest_lap_data['fastest_lap']
                telemetry = fastest_lap_data['telemetry']

                sector1_telemetry = telemetry[(telemetry['TimeDelta'] >= fastest_lap['LapStartTime']) & (telemetry['TimeDelta'] < fastest_lap['Sector1SessionTime'])]
                sector2_telemetry = telemetry[(telemetry['TimeDelta'] >= fastest_lap['Sector1SessionTime']) & (telemetry['TimeDelta'] < fastest_lap['Sector2SessionTime'])]
                sector3_telemetry = telemetry[(telemetry['TimeDelta'] >= fastest_lap['Sector2SessionTime']) & (telemetry['TimeDelta'] < fastest_lap['LapStartTime'] + fastest_lap['LapTime'])]

                if not sector1_telemetry.empty:
                    avg_speed_df = pd.concat([
                        avg_speed_df,
                        pd.DataFrame([{'Driver': driver_code, 'Sector': 'Sector 1', 'AvgSpeed': sector1_telemetry['Speed'].mean()}])
                    ])
                if not sector2_telemetry.empty:
                    avg_speed_df = pd.concat([
                        avg_speed_df,
                        pd.DataFrame([{'Driver': driver_code, 'Sector': 'Sector 2', 'AvgSpeed': sector2_telemetry['Speed'].mean()}])
                    ])
                if not sector3_telemetry.empty:
                    avg_speed_df = pd.concat([
                        avg_speed_df,
                        pd.DataFrame([{'Driver': driver_code, 'Sector': 'Sector 3', 'AvgSpeed': sector3_telemetry['Speed'].mean()}])
                    ])

        if not avg_speed_df.empty:
            fig7, ax7 = plt.subplots(figsize=(12, 7))
            sns.barplot(x='Sector', y='AvgSpeed', hue='Driver', data=avg_speed_df, ax=ax7, palette='rocket')
            ax7.set_title(f'Average Speed per Sector (Fastest Lap) - {year} {grand_prix} {session_type}')
            ax7.set_xlabel('Sector')
            ax7.set_ylabel('Average Speed (km/h)')
            ax7.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()

        # Interpretive Summary
        print("\n--- Interpreting Strengths and Weaknesses ---")
        print("The 'Average Delta to Fastest Sector' plot highlights where each driver gains or loses time relative to the fastest driver in each sector. A positive delta means they are slower, while a negative or near-zero delta indicates a strength.")
        print("The 'Average Speed per Sector' plot provides context. For example, a driver might have a high average speed in a sector but still a positive delta if their cornering speed is not optimal, or if they are carrying too much speed into a corner and losing time on exit.")
        print("By comparing these two plots, you can identify:")
        print("- **Strengths**: Low delta to fastest sector, potentially coupled with high average speed.")
        print("- **Weaknesses**: High positive delta to fastest sector, which might be explained by lower average speeds in certain sections of that sector.")
        print("This analysis helps pinpoint specific areas on the track where a driver excels or struggles, guiding further detailed analysis or setup changes.")

        # Stint Analysis
        # Identify stints (laps between pit stops or start/end of session)
        stints = all_drivers_laps[['Driver', 'LapNumber', 'Stint', 'Sector1(s)', 'Sector2(s)', 'Sector3(s)']].copy()
        stint_averages = stints.groupby(['Driver', 'Stint'])[['Sector1(s)', 'Sector2(s)', 'Sector3(s)']].mean().reset_index()
        stint_averages['TotalStintTime'] = stint_averages[['Sector1(s)', 'Sector2(s)', 'Sector3(s)']].sum(axis=1)

        if not stint_averages.empty:
            print("\n--- Stint Analysis ---")
            print(stint_averages)

            fig5, ax5 = plt.subplots(figsize=(14, 7))
            sns.barplot(x='Stint', y='TotalStintTime', hue='Driver', data=stint_averages, ax=ax5, palette='viridis')
            ax5.set_title(f'Average Total Stint Time per Driver - {year} {grand_prix} {session_type}')
            ax5.set_xlabel('Stint Number')
            ax5.set_ylabel('Average Total Stint Time (s)')
            ax5.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()

        plt.show() # Show all plots at the end

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have an internet connection and the specified race/driver data is available.")
        print("You might also want to try a different year, grand prix, session type, or driver.")
        print("If the error persists, check the exact Grand Prix name and driver code.")

# --- Example Usage ---
if __name__ == "__main__":
    CURRENT_YEAR = pd.to_datetime('today').year
    # Find the most recent race to analyze
    GRAND_PRIX, _ = get_most_recent_completed_race(CURRENT_YEAR)
    if GRAND_PRIX is None:
        # Fallback to previous year if no races are completed
        CURRENT_YEAR -= 1
        GRAND_PRIX, _ = get_most_recent_completed_race(CURRENT_YEAR)

    if GRAND_PRIX:
        SESSION_TYPE = 'Race'
        
        # --- Analysis 1: Multi-Driver Sector Time Analysis ---
        print("\n" + "="*50)
        print("  RUNNING MULTI-DRIVER SECTOR TIME ANALYSIS")
        print("="*50)
        analyze_sector_times(CURRENT_YEAR, GRAND_PRIX, SESSION_TYPE, ['VER', 'NOR', 'PIA','RUS'])

    else:
        print("Could not find any recent races to analyze.")
