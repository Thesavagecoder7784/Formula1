import fastf1
import fastf1.plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Enable the cache for faster data loading on subsequent runs
fastf1.Cache.enable_cache('/Users/prabhatm/Documents/GitHub/Formula1/cache') # Adjusted path for cache

# Set up plotting style
fastf1.plotting.setup_mpl()

def get_session_data(year, grand_prix):
    """
    Loads the qualifying and race session data.
    """
    try:
        qualifying = fastf1.get_session(year, grand_prix, 'Q')
        qualifying.load(telemetry=False, weather=False, messages=False)

        race = fastf1.get_session(year, grand_prix, 'R')
        race.load(telemetry=False, weather=False, messages=False)
        return qualifying, race
    except Exception as e:
        print(f"Error loading session data for {year} {grand_prix}: {e}")
        return None, None

def analyze_qualifying_pace(session, driver_code):
    """
    Analyzes the qualifying pace for a single driver.
    Returns the best qualifying lap time in seconds.
    """
    laps = session.laps.pick_driver(driver_code)
    if laps.empty:
        return None
    fastest_lap = laps.pick_fastest()
    if fastest_lap is not None:
        return fastest_lap['LapTime'].total_seconds()
    return None

def analyze_race_stints(laps):
    """
    Analyzes race stints, calculating average pace and degradation.
    Returns a DataFrame with stint analysis.
    """
    # Ensure 'Stint' column exists and is correctly populated
    # FastF1 usually handles this, but a fallback can be useful
    if 'Stint' not in laps.columns or laps['Stint'].isnull().all():
        laps = laps.assign(Stint=(laps['PitOutTime'].notna() | laps['PitInTime'].notna()).cumsum() + 1)
        laps.loc[laps['PitOutTime'].notna(), 'Stint'] += 1 # Increment stint after pit out
        laps['Stint'] = laps['Stint'].fillna(1) # First stint

    stints = laps.groupby('Stint')
    stint_data = []

    for stint_num, stint_laps in stints:
        # Filter out pit in/out laps and inaccurate laps for pace calculation
        valid_stint_laps = stint_laps.loc[
            (stint_laps['IsAccurate'] == True) &
            (stint_laps['LapTime'].notna()) &
            (stint_laps['PitInTime'].isna()) &
            (stint_laps['PitOutTime'].isna())
        ].copy()

        if len(valid_stint_laps) < 3:  # Need at least 3 valid laps for meaningful analysis
            continue

        compound = valid_stint_laps['Compound'].mode()[0] # Most frequent compound in stint
        lap_times_seconds = valid_stint_laps['LapTime'].dt.total_seconds()
        
        # Remove outliers for average pace calculation (e.g., very slow laps due to traffic/safety car)
        q_low = lap_times_seconds.quantile(0.05)
        q_high = lap_times_seconds.quantile(0.95)
        filtered_lap_times = lap_times_seconds[(lap_times_seconds > q_low) & (lap_times_seconds < q_high)]

        if filtered_lap_times.empty:
            continue

        avg_pace = filtered_lap_times.mean()
        
        # Calculate degradation as the slope of a linear fit for valid laps
        # Use LapNumber relative to stint start for degradation calculation
        stint_lap_numbers = valid_stint_laps['LapNumber'] - valid_stint_laps['LapNumber'].min()
        
        try:
            # Filter out NaNs from both series before polyfit
            valid_indices = ~np.isnan(stint_lap_numbers) & ~np.isnan(lap_times_seconds)
            if valid_indices.sum() > 1: # Need at least 2 points for a line
                poly = np.polyfit(stint_lap_numbers[valid_indices], lap_times_seconds[valid_indices], 1)
                degradation = poly[0] * 60 # Convert degradation to seconds per minute (per lap)
            else:
                degradation = 0.0
        except (np.linalg.LinAlgError, ValueError):
            degradation = 0.0

        stint_data.append({
            'Stint': stint_num,
            'Compound': compound,
            'Laps': len(valid_stint_laps),
            'AvgPace': avg_pace,
            'Degradation_s_per_lap': degradation
        })

    return pd.DataFrame(stint_data)

def plot_pace_comparison(driver_paces, title):
    """
    Plots qualifying vs. race pace for multiple drivers with annotations.
    """
    plot_df = pd.DataFrame(driver_paces).melt(id_vars='Driver', var_name='Session', value_name='LapTime(s)')
    plot_df = plot_df.dropna(subset=['LapTime(s)'])

    if plot_df.empty:
        print("No data to plot for pace comparison.")
        return

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='Driver', y='LapTime(s)', hue='Session', data=plot_df, palette={'Qualifying': 'skyblue', 'Race': 'lightcoral'})
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')

    plt.title(title, fontsize=16)
    plt.xlabel("Driver", fontsize=12)
    plt.ylabel("Lap Time (Seconds)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title='Session')
    plt.tight_layout()
    plt.show()

def plot_quali_race_scatter(driver_paces, title):
    """
    Plots a scatter plot of qualifying pace vs. race pace delta.
    """
    plot_df = pd.DataFrame(driver_paces).dropna(subset=['Qualifying', 'Race'])
    if plot_df.empty:
        print("No data to plot for quali vs. race scatter.")
        return

    # Calculate Pace Delta: Race Pace - Qualifying Pace
    plot_df['Pace_Delta'] = plot_df['Race'] - plot_df['Qualifying']

    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(x='Qualifying', y='Pace_Delta', data=plot_df, hue='Driver', s=150, alpha=0.8, style='Driver')
    
    for i, row in plot_df.iterrows():
        ax.text(row['Qualifying'], row['Pace_Delta'] + 0.02, row['Driver'], ha='center', va='bottom', fontsize=9)

    # Add horizontal lines for interpretation
    ax.axhline(0, color='red', linestyle='--', label='Race Pace = Quali Pace')
    ax.axhline(0.5, color='orange', linestyle=':', label='Race Pace 0.5s Slower')
    ax.axhline(1.0, color='red', linestyle=':', label='Race Pace 1.0s Slower')

    plt.title(title, fontsize=16)
    plt.xlabel("Best Qualifying Lap Time (s)", fontsize=12)
    plt.ylabel("Race Pace - Quali Pace (s)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Driver', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_stint_analysis(driver_stints_data, title):
    """
    Plots the average pace, degradation, and laps for each stint of multiple drivers.
    """
    all_stints_df = pd.DataFrame()
    for driver, stints_df in driver_stints_data.items():
        if not stints_df.empty:
            stints_df['Driver'] = driver
            all_stints_df = pd.concat([all_stints_df, stints_df], ignore_index=True)

    if all_stints_df.empty:
        print("No stint data to plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=True)

    # Plot Average Pace
    ax0 = sns.barplot(x='Driver', y='AvgPace', hue='Compound', data=all_stints_df, ax=axes[0], palette=fastf1.plotting.COMPOUND_COLORS)
    axes[0].set_title(f'{title} - Average Stint Pace', fontsize=16)
    axes[0].set_ylabel('Average Lap Time (s)', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    axes[0].legend(title='Compound')
    for container in ax0.containers:
        ax0.bar_label(container, fmt='%.3f')

    # Plot Degradation
    ax1 = sns.barplot(x='Driver', y='Degradation_s_per_lap', hue='Compound', data=all_stints_df, ax=axes[1], palette=fastf1.plotting.COMPOUND_COLORS)
    axes[1].set_title(f'{title} - Stint Degradation (s/lap)', fontsize=16)
    axes[1].set_ylabel('Degradation (s/lap)', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    axes[1].legend(title='Compound')
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.3f')

    # Plot Laps in Stint
    ax2 = sns.barplot(x='Driver', y='Laps', hue='Compound', data=all_stints_df, ax=axes[2], palette=fastf1.plotting.COMPOUND_COLORS)
    axes[2].set_title(f'{title} - Laps per Stint', fontsize=16)
    axes[2].set_ylabel('Number of Laps', fontsize=12)
    axes[2].set_xlabel('Driver', fontsize=12)
    axes[2].grid(axis='y', linestyle='--', alpha=0.6)
    axes[2].legend(title='Compound')
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%d')

    plt.tight_layout()
    plt.show()

def main():
    YEAR = 2025
    GRAND_PRIX = "British Grand Prix"
    # Define drivers to analyze: main, teammate, and rivals
    MAIN_DRIVER = 'NOR'  
    TEAMMATE_DRIVER = 'PIA'
    RIVAL_DRIVERS = ['VER', 'RUS']

    drivers_to_analyze = [MAIN_DRIVER]
    if TEAMMATE_DRIVER and TEAMMATE_DRIVER not in drivers_to_analyze:
        drivers_to_analyze.append(TEAMMATE_DRIVER)
    if RIVAL_DRIVERS:
        for rival in RIVAL_DRIVERS:
            if rival not in drivers_to_analyze:
                drivers_to_analyze.append(rival)

    qualifying_session, race_session = get_session_data(YEAR, GRAND_PRIX)
    if qualifying_session is None or race_session is None:
        return

    driver_paces = []
    driver_stints_data = {}

    print(f"\n--- Analyzing Performance for {YEAR} {GRAND_PRIX} ---")

    for driver_code in drivers_to_analyze:
        print(f"Fetching data for {driver_code}...")
        quali_pace = analyze_qualifying_pace(qualifying_session, driver_code)
        
        # Ensure 'Stint' column is available before passing to analyze_race_stints
        race_laps = race_session.laps.pick_driver(driver_code).pick_accurate()
        if not race_laps.empty:
            # Assign stints based on pit stops
            race_laps['Stint'] = (race_laps['PitOutTime'].notna() | race_laps['PitInTime'].notna()).cumsum() + 1
            race_laps.loc[race_laps['PitOutTime'].notna(), 'Stint'] += 1
            race_laps['Stint'] = race_laps['Stint'].fillna(1).astype(int)

        stint_analysis_df = analyze_race_stints(race_laps)

        driver_paces.append({
            'Driver': driver_code,
            'Qualifying': quali_pace,
            'Race': stint_analysis_df['AvgPace'].mean() if not stint_analysis_df.empty else None # Overall average race pace
        })
        driver_stints_data[driver_code] = stint_analysis_df

        print(f"\n--- {driver_code} Summary ---")
        if quali_pace:
            print(f"Best Qualifying Lap: {quali_pace:.3f}s")
        else:
            print("No qualifying data.")
        
        if not stint_analysis_df.empty:
            print("Race Stint Analysis:")
            print(stint_analysis_df.to_string(index=False))
        else:
            print("No race stint data.")
        print("\n")

    # Plotting
    plot_pace_comparison(driver_paces, f"{YEAR} {GRAND_PRIX} - Qualifying vs. Race Pace")
    plot_quali_race_scatter(driver_paces, f"{YEAR} {GRAND_PRIX} - Quali vs. Race Pace Scatter")
    plot_stint_analysis(driver_stints_data, f"{YEAR} {GRAND_PRIX}")

    # Interpretations and Insights
    print("\n--- Key Insights ---")
    for driver_data in driver_paces:
        driver = driver_data['Driver']
        q_pace = driver_data['Qualifying']
        r_pace = driver_data['Race']

        if q_pace and r_pace:
            pace_delta = r_pace - q_pace
            print(f"* {driver}:\n  - Quali Pace: {q_pace:.3f}s, Race Pace: {r_pace:.3f}s, Pace Delta (Race - Quali): {pace_delta:.3f}s.")
            if pace_delta < 0.2: # Arbitrary threshold for 'consistent'
                print(f"  - Interpretation: Highly consistent across qualifying and race. This driver/car combination maintains strong performance in race conditions relative to their qualifying speed. Likely a well-balanced car and excellent tire management.")
            elif pace_delta > 0.5: # Arbitrary threshold for 'quali specialist'
                print(f"  - Interpretation: Appears to be a 'Qualifying Specialist'. While having excellent one-lap speed, their race pace is significantly slower, suggesting potential issues with tire degradation, heavy-fuel handling, or race setup.")
            else:
                print(f"  - Interpretation: Generally consistent, but with a slight drop-off in race pace compared to qualifying. This is common, but further analysis of stint degradation is recommended.")
        elif q_pace:
            print(f"* {driver}: Only qualifying data available. Best Quali Lap: {q_pace:.3f}s.")
        elif r_pace:
            print(f"* {driver}: Only race data available. Average Race Pace: {r_pace:.3f}s.")
        else:
            print(f"* {driver}: Insufficient data for comprehensive analysis.")

    # Teammate and Rival Comparisons (more detailed)
    main_driver_data = next((d for d in driver_paces if d['Driver'] == MAIN_DRIVER), None)
    if main_driver_data:
        print(f"\n--- Detailed Comparisons for {MAIN_DRIVER} ---")
        for driver_data in driver_paces:
            if driver_data['Driver'] == MAIN_DRIVER: continue

            comp_driver = driver_data['Driver']
            main_q = main_driver_data['Qualifying']
            comp_q = driver_data['Qualifying']
            main_r = main_driver_data['Race']
            comp_r = driver_data['Race']

            print(f"\nvs. {comp_driver}:")
            if main_q and comp_q:
                if main_q < comp_q:
                    print(f"  - Quali Pace: {MAIN_DRIVER} is faster ({main_q:.3f}s vs {comp_q:.3f}s).")
                elif main_q > comp_q:
                    print(f"  - Quali Pace: {comp_q:.3f}s vs {main_q:.3f}s).")
                else:
                    print(f"  - Quali Pace: Similar.")
            
            if main_r and comp_r:
                if main_r < comp_r:
                    print(f"  - Race Pace: {MAIN_DRIVER} is faster ({main_r:.3f}s vs {comp_r:.3f}s).")
                elif main_r > comp_r:
                    print(f"  - Race Pace: {comp_r:.3f}s vs {main_r:.3f}s). ")
                else:
                    print(f"  - Race Pace: Similar.")
            
            # Stint-by-stint comparison if data available
            main_stints = driver_stints_data.get(MAIN_DRIVER)
            comp_stints = driver_stints_data.get(comp_driver)

            if main_stints is not None and comp_stints is not None and not main_stints.empty and not comp_stints.empty:
                print("  - Stint-by-Stint Comparison:")
                merged_stints = pd.merge(main_stints, comp_stints, on='Compound', suffixes=(f'_{MAIN_DRIVER}', f'_{comp_driver}'))
                for index, row in merged_stints.iterrows():
                    print(f"    - {row['Compound']} Tires:")
                    if row[f'AvgPace_{MAIN_DRIVER}'] < row[f'AvgPace_{comp_driver}']:
                        print(f"      - {MAIN_DRIVER} Avg Pace: {row[f'AvgPace_{MAIN_DRIVER}']:.3f}s (faster than {comp_driver})")
                    else:
                        print(f"      - {comp_driver} Avg Pace: {row[f'AvgPace_{comp_driver}']:.3f}s (faster than {MAIN_DRIVER})")
                    
                    if row[f'Degradation_s_per_lap_{MAIN_DRIVER}'] < row[f'Degradation_s_per_lap_{comp_driver}']:
                        print(f"      - {MAIN_DRIVER} Degradation: {row[f'Degradation_s_per_lap_{MAIN_DRIVER}']:.3f} s/lap (better than {comp_driver})")
                    else:
                        print(f"      - {comp_driver} Degradation: {row[f'Degradation_s_per_lap_{comp_driver}']:.3f} s/lap (better than {MAIN_DRIVER})")


if __name__ == "__main__":
    main()