import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np

# Enable the cache for faster data loading on subsequent runs
fastf1.Cache.enable_cache('cache/')

# Set up plotting style (optional, as this script focuses on text output)
fastf1.plotting.setup_mpl()

def analyze_winner_performance(year, grand_prix, winner_driver_code):
    """
    Analyzes why a specific driver won a Grand Prix, covering various aspects
    of race performance.

    Args:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix.
        winner_driver_code (str): The three-letter code of the winning driver.
    """
    print(f"\n--- Analyzing Why {winner_driver_code} Won the {grand_prix} {year} Grand Prix ---")

    try:
        # Load session data
        race = fastf1.get_session(year, grand_prix, 'Race')
        race.load(laps=True, telemetry=False, weather=False, messages=True)

        results = race.results
        laps = race.laps
        messages = race.race_control_messages

        # Identify winner and key rivals
        winner_info = results.loc[results['Abbreviation'] == winner_driver_code]
        if winner_info.empty:
            print(f"Error: Winner driver code '{winner_driver_code}' not found in race results.")
            return

        # Get top 3-5 finishers excluding winner for comparison
        top_finishers = results.sort_values(by='Position').head(5)
        rival_drivers = top_finishers.loc[top_finishers['Abbreviation'] != winner_driver_code]['Abbreviation'].tolist()
        all_drivers_to_analyze = [winner_driver_code] + rival_drivers

        print(f"Comparing {winner_driver_code} against rivals: {', '.join(rival_drivers)}")

        # --- 1. Pace Advantage ---
        print("\n--- 1. Pace Advantage ---")
        driver_paces = {}
        for driver in all_drivers_to_analyze:
            driver_laps = laps.pick_driver(driver).pick_accurate()
            if not driver_laps.empty:
                driver_paces[driver] = driver_laps['LapTime'].dt.total_seconds().median()
            else:
                driver_paces[driver] = np.nan

        winner_pace = driver_paces.get(winner_driver_code)
        if pd.isna(winner_pace):
            print(f"{winner_driver_code}: Insufficient lap data for pace analysis.")
        else:
            print(f"{winner_driver_code}'s Median Race Pace: {winner_pace:.3f}s")
            for driver, pace in driver_paces.items():
                if driver != winner_driver_code and not pd.isna(pace):
                    pace_diff = pace - winner_pace
                    if pace_diff > 0:
                        print(f"  - {driver} was {pace_diff:.3f}s slower per lap.")
                    else:
                        print(f"  - {driver} was {-pace_diff:.3f}s faster per lap (unlikely for winner, but possible in specific scenarios).")
            
            # Qualitative assessment
            if all(pd.isna(driver_paces[d]) or driver_paces[d] >= winner_pace for d in rival_drivers):
                print(f"Conclusion: {winner_driver_code} demonstrated a clear pace advantage over key rivals.")
            else:
                print(f"Conclusion: {winner_driver_code} had competitive pace, but not necessarily a dominant advantage over all rivals.")


        # --- 2. Strategy (Pit Stops & Tires) ---
        print("\n--- 2. Strategy (Pit Stops & Tires) ---")
        for driver in all_drivers_to_analyze:
            driver_laps_all = laps.pick_driver(driver)
            
            # Count pit stops by checking PitInTime
            pit_stops_count = driver_laps_all['PitInTime'].count()

            # Calculate stints using the provided robust method
            if not driver_laps_all.empty:
                driver_stints = driver_laps_all[["Driver", "Stint", "Compound", "LapNumber"]]
                driver_stints = driver_stints.groupby(["Driver", "Stint", "Compound"]).count().reset_index()
                driver_stints = driver_stints.rename(columns={"LapNumber": "StintLength"})
            else:
                driver_stints = pd.DataFrame()

            print(f"\nDriver: {driver}")
            print(f"  Total Pit Stops: {pit_stops_count}")
            if pit_stops_count > 0:
                print("  (Note: Pit stop durations are not directly available with this method.)")
            
            if not driver_stints.empty:
                print("  Tire Strategy (Stints):")
                print(driver_stints[['Stint', 'Compound', 'StintLength']].to_string(index=False))
            else:
                print("  No stint data available.")

        # Qualitative assessment (needs more logic to compare strategies effectively)
        print("Conclusion: (Detailed comparison of strategies would require more complex logic, e.g., comparing total race time on each compound, optimal pit windows.)")


        # --- 3. Consistency ---
        print("\n--- 3. Consistency ---")
        driver_consistency = {}
        for driver in all_drivers_to_analyze:
            driver_laps_for_consistency = laps.pick_driver(driver).pick_accurate()
            if not driver_laps_for_consistency.empty and len(driver_laps_for_consistency) > 1:
                driver_consistency[driver] = driver_laps_for_consistency['LapTime'].dt.total_seconds().std()
            else:
                driver_consistency[driver] = np.nan
        
        winner_consistency = driver_consistency.get(winner_driver_code)
        if pd.isna(winner_consistency):
            print(f"{winner_driver_code}: Insufficient lap data for consistency analysis.")
        else:
            print(f"{winner_driver_code}'s Lap Time Standard Deviation: {winner_consistency:.3f}s")
            for driver, std_dev in driver_consistency.items():
                if driver != winner_driver_code and not pd.isna(std_dev):
                    if std_dev < winner_consistency:
                        print(f"  - {driver} ({std_dev:.3f}s) was more consistent.")
                    else:
                        print(f"  - {driver} ({std_dev:.3f}s) was less consistent.")
            
            if all(pd.isna(driver_consistency[d]) or driver_consistency[d] >= winner_consistency for d in rival_drivers):
                print(f"Conclusion: {winner_driver_code} demonstrated excellent lap time consistency throughout the race.")
            else:
                print(f"Conclusion: {winner_driver_code} had good consistency, but some rivals might have been slightly more consistent.")


        # --- 4. Tire Management ---
        print("\n--- 4. Tire Management ---")
        # This is a simplified degradation calculation (slope of lap time vs lap number per stint)
        driver_degradation = {}
        for driver in all_drivers_to_analyze:
            # Get all laps for the current driver
            all_driver_laps = laps.pick_driver(driver)

            # Calculate stints for the current driver using the robust method
            if not all_driver_laps.empty:
                driver_stints_summary = all_driver_laps[["Driver", "Stint", "Compound", "LapNumber"]]
                driver_stints_summary = driver_stints_summary.groupby(["Driver", "Stint", "Compound"]).count().reset_index()
                driver_stints_summary = driver_stints_summary.rename(columns={"LapNumber": "StintLength"})
            else:
                driver_stints_summary = pd.DataFrame()

            degradation_rates = []
            # Iterate through each unique stint for the driver from the summary
            for _, stint_info in driver_stints_summary.iterrows():
                stint_num = stint_info['Stint']
                compound = stint_info['Compound']

                # Filter original laps for this specific stint and compound
                stint_laps = all_driver_laps.loc[
                    (all_driver_laps['Stint'] == stint_num) &
                    (all_driver_laps['Compound'] == compound)
                ].pick_fastest()

                if len(stint_laps) > 1:
                    lap_times_seconds = stint_laps['LapTime'].total_seconds()
                    lap_numbers_series = pd.Series(stint_laps['LapNumber'])
                    relative_lap_numbers = lap_numbers_series - lap_numbers_series.min()
                    
                    # Debug prints (can be removed after fix verification)
                    # print(f"Debug: stint_laps shape: {stint_laps.shape}")
                    # print(f"Debug: relative_lap_numbers type: {type(relative_lap_numbers)}, shape: {relative_lap_numbers.shape}, content: {relative_lap_numbers.tolist()}")
                    # print(f"Debug: lap_times_seconds type: {type(lap_times_seconds)}, shape: {lap_times_seconds.shape}, content: {lap_times_seconds.tolist()}")

                    if len(relative_lap_numbers) > 1 and not relative_lap_numbers.empty and not lap_times_seconds.empty and len(relative_lap_numbers) == len(lap_times_seconds):
                        slope, _ = np.polyfit(relative_lap_numbers, lap_times_seconds, 1)
                        degradation_rates.append(slope)
                    else:
                        print("Debug: Skipping polyfit due to empty or mismatched length arrays.")
            if degradation_rates:
                driver_degradation[driver] = np.mean(degradation_rates)
            else:
                driver_degradation[driver] = np.nan
        
        winner_degradation = driver_degradation.get(winner_driver_code)
        if pd.isna(winner_degradation):
            print(f"{winner_driver_code}: Insufficient data for tire degradation analysis.")
        else:
            print(f"{winner_driver_code}'s Average Tire Degradation: {winner_degradation:.3f}s/lap")
            for driver, deg_rate in driver_degradation.items():
                if driver != winner_driver_code and not pd.isna(deg_rate):
                    if deg_rate < winner_degradation:
                        print(f"  - {driver} ({deg_rate:.3f}s/lap) had better tire management.")
                    else:
                        print(f"  - {driver} ({deg_rate:.3f}s/lap) had worse tire management.")
            
            if all(pd.isna(driver_degradation[d]) or driver_degradation[d] >= winner_degradation for d in rival_drivers):
                print(f"Conclusion: {winner_driver_code} demonstrated superior tire management, maintaining pace effectively.")
            else:
                print(f"Conclusion: {winner_driver_code} had good tire management, but some rivals were comparable or better.")


        # --- 5. Overtakes/Defensive Driving ---
        print("\n--- 5. Overtakes/Defensive Driving ---")
        winner_laps = laps.pick_driver(winner_driver_code)
        if not winner_laps.empty:
            start_pos = winner_laps.iloc[0]['Position']
            end_pos = winner_laps.iloc[-1]['Position']
            net_position_change = start_pos - end_pos # Positive means gained positions

            print(f"{winner_driver_code} started in P{start_pos} and finished in P{end_pos}.")
            if net_position_change > 0:
                print(f"  - Gained {net_position_change} net positions during the race.")
            elif net_position_change < 0:
                print(f"  - Lost {-net_position_change} net positions during the race.")
            else:
                print("  - Maintained starting position (or net zero change).")
            
            # Simple overtake count (can be inaccurate, just looking at position changes)
            position_changes = winner_laps['Position'].diff().fillna(0)
            overtakes_made = (position_changes < 0).sum() # Position decreases means overtake
            positions_lost = (position_changes > 0).sum() # Position increases means lost position

            print(f"  - Estimated Overtakes Made: {overtakes_made}")
            print(f"  - Estimated Positions Lost: {positions_lost}")

            if overtakes_made > positions_lost:
                print(f"Conclusion: {winner_driver_code} was effective in on-track battles, making more overtakes than positions lost.")
            elif positions_lost > overtakes_made:
                print(f"Conclusion: {winner_driver_code} had to defend more or lost positions on track.")
            else:
                print(f"Conclusion: {winner_driver_code}'s on-track battles were balanced.")
        else:
            print(f"{winner_driver_code}: No lap data for position analysis.")


        # --- 6. Safety Car/VSC/Red Flag Incidents ---
        print("\n--- 6. Safety Car/VSC/Red Flag Incidents ---")
        safety_car_events = messages[messages['Category'].isin(['SafetyCar', 'VirtualSafetyCar', 'RedFlag'])]
        if not safety_car_events.empty:
            print("Race Control Incidents:")
            for _, msg in safety_car_events.iterrows():
                print(f"  - {msg['Time'].strftime('%H:%M:%S')} - {msg['Category']}: {msg['Message']}")
            
            # Assess impact (simplified)
            print("Conclusion: (Assessing the exact impact of these events on the winner vs. rivals would require detailed lap-by-lap position and time delta analysis around the incident laps.)")
        else:
            print("No Safety Car, VSC, or Red Flag incidents recorded.")


        # --- 7. Luck/Unforeseen Circumstances ---
        print("\n--- 7. Luck/Unforeseen Circumstances ---")
        rival_issues = []
        for driver in rival_drivers:
            driver_result = results.loc[results['Abbreviation'] == driver]
            if not driver_result.empty:
                status = driver_result.iloc[0]['Status']
                if status != 'Finished':
                    rival_issues.append(f"{driver} ({status})")
        
        if rival_issues:
            print(f"Key rivals faced issues: {', '.join(rival_issues)}")
            print("Conclusion: Unforeseen circumstances for rivals likely contributed to the winner's victory.")
        else:
            print("No significant DNF or penalty issues observed for key rivals.")
            print("Conclusion: The victory was primarily due to on-track performance rather than rival misfortunes.")


    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        print("Please ensure you have an internet connection and the specified race data is available.")
        print("You might also want to try a different year, grand prix, or winner driver code.")


# --- Example Usage ---
if __name__ == "__main__":
    analyze_winner_performance(2025, 'Hungarian Grand Prix', 'NOR')
