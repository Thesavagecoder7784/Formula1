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

def analyze_race_performance(year, grand_prix, session_type):
    """
    Analyzes and compares lap times, average pace, and consistency for top drivers
    in a given race session, returning insights for aggregation.

    Args:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix.
        session_type (str): The session type.

    Returns:
        dict: A dictionary containing insights for the race, or None if data is unavailable.
    """
    # print(f"\n--- Analyzing {year} {grand_prix} - {session_type} ---") # Suppress for season-long analysis

    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load()

        all_session_drivers = session.results['Abbreviation'].tolist()
        if not all_session_drivers:
            # print(f"No drivers found for {year} {grand_prix} {session_type}.")
            return None

        driver_performance_data = []

        for driver_code in all_session_drivers:
            driver_laps = session.laps.pick_drivers(driver_code)

            valid_laps = driver_laps.loc[
                (driver_laps['IsAccurate'] == True) &
                (driver_laps['LapTime'].notna()) &
                (driver_laps['PitInTime'].isna()) &
                (driver_laps['PitOutTime'].isna())
            ].copy()

            if valid_laps.empty:
                continue

            valid_laps['LapTime(s)'] = valid_laps['LapTime'].dt.total_seconds()
            
            avg_pace = valid_laps['LapTime(s)'].mean()
            consistency_std = valid_laps['LapTime(s)'].std()

            driver_performance_data.append({
                'Driver': driver_code,
                'AveragePace': avg_pace,
                'ConsistencyStd': consistency_std,
                'Position': session.results.loc[session.results['Abbreviation'] == driver_code, 'Position'].iloc[0]
            })

        if not driver_performance_data:
            # print("No valid performance data found for any drivers in this session.")
            return None

        performance_df = pd.DataFrame(driver_performance_data)
        performance_df.dropna(inplace=True) # Drop rows with NaN values if any

        # Identify Top 5 Finishers
        top_5_finishers = performance_df.sort_values(by='Position').head(5)

        # Identify Top 5 Fastest Average Pace
        top_5_fastest_pace = performance_df.sort_values(by='AveragePace').head(5)

        # Identify Top 5 Most Consistent
        top_5_consistent = performance_df.sort_values(by='ConsistencyStd').head(5)

        # Prepare insights for return
        race_insights = {
            'GrandPrix': grand_prix,
            'Year': year,
            'Winner': None,
            'WinnerAvgPace': None,
            'WinnerConsistencyStd': None,
            'OverallFastest': None,
            'OverallFastestAvgPace': None,
            'OverallFastestConsistencyStd': None,
            'OverallMostConsistent': None,
            'OverallMostConsistentAvgPace': None,
            'OverallMostConsistentConsistencyStd': None,
            'DominantPerformers': [],
            'FastButNotFinished': [],
            'ConsistentButNotFinished': [],
            'ResultWithoutPaceConsistency': []
        }

        # Winner's Performance
        race_winner_data = performance_df.loc[performance_df['Position'] == 1].iloc[0] if not top_5_finishers.empty else None
        if race_winner_data is not None:
            race_insights['Winner'] = race_winner_data['Driver']
            race_insights['WinnerAvgPace'] = race_winner_data['AveragePace']
            race_insights['WinnerConsistencyStd'] = race_winner_data['ConsistencyStd']

        # Overall Fastest and Most Consistent
        if not top_5_fastest_pace.empty:
            overall_fastest_driver_data = top_5_fastest_pace.iloc[0]
            race_insights['OverallFastest'] = overall_fastest_driver_data['Driver']
            race_insights['OverallFastestAvgPace'] = overall_fastest_driver_data['AveragePace']
            race_insights['OverallFastestConsistencyStd'] = overall_fastest_driver_data['ConsistencyStd']

        if not top_5_consistent.empty:
            overall_most_consistent_driver_data = top_5_consistent.iloc[0]
            race_insights['OverallMostConsistent'] = overall_most_consistent_driver_data['Driver']
            race_insights['OverallMostConsistentAvgPace'] = overall_most_consistent_driver_data['AveragePace']
            race_insights['OverallMostConsistentConsistencyStd'] = overall_most_consistent_driver_data['ConsistencyStd']

        # Cross-Category Insights (Top 5)
        top_5_finishers_set = set(top_5_finishers['Driver'])
        top_5_fastest_pace_set = set(top_5_fastest_pace['Driver'])
        top_5_consistent_set = set(top_5_consistent['Driver'])

        race_insights['DominantPerformers'] = list(top_5_finishers_set.intersection(top_5_fastest_pace_set, top_5_consistent_set))
        race_insights['FastButNotFinished'] = list(top_5_fastest_pace_set.difference(top_5_finishers_set))
        race_insights['ConsistentButNotFinished'] = list(top_5_consistent_set.difference(top_5_finishers_set))
        race_insights['ResultWithoutPaceConsistency'] = list(top_5_finishers_set.difference(top_5_fastest_pace_set.union(top_5_consistent_set)))

        return race_insights

    except Exception as e:
        print(f"An error occurred for {year} {grand_prix}: {e}")
        print("Skipping this race.")
        return None


# --- Main Execution for Season-long Analysis ---
if __name__ == "__main__":
    YEAR = 2025
    SESSION_TYPE = 'Race'
    MAX_ROUND = 12 # Analyze up to round 12

    schedule = fastf1.get_event_schedule(YEAR)
    today = pd.to_datetime('today').date()
    completed_races = schedule.loc[(schedule['EventDate'].dt.date < today) & (schedule['RoundNumber'] >= 1) & (schedule['RoundNumber'] <= MAX_ROUND)]

    all_season_insights = []

    if completed_races.empty:
        print(f"No completed races found for {YEAR} up to round {MAX_ROUND}.")
    else:
        for index, event in completed_races.iterrows():
            grand_prix_name = event['EventName']
            insights = analyze_race_performance(YEAR, grand_prix_name, SESSION_TYPE)
            if insights:
                all_season_insights.append(insights)

    print("\n==================================================")
    print(f"SEASON-LONG INSIGHTS SUMMARY ({YEAR} F1 Season - Rounds 1 to {MAX_ROUND})")
    print("==================================================")

    if not all_season_insights:
        print("No insights collected for the season.")
    else:
        # Overall Winner Performance
        winner_fastest_count = 0
        winner_consistent_count = 0
        winner_dominant_count = 0
        total_races_analyzed = len(all_season_insights)

        # Initialize all_dominant_performers_season with all drivers who were dominant in the first race
        # or an empty set if no races or no dominant performers in the first race
        all_dominant_performers_season = set(all_season_insights[0]['DominantPerformers']) if all_season_insights and all_season_insights[0]['DominantPerformers'] else set()

        # Track frequency for other categories
        all_fast_but_not_finished = {} # Driver: count
        all_consistent_but_not_finished = {} # Driver: count
        all_result_without_pace_consistency = {} # Driver: count

        for insight in all_season_insights:
            if insight['Winner'] == insight['OverallFastest']:
                winner_fastest_count += 1
            if insight['Winner'] == insight['OverallMostConsistent']:
                winner_consistent_count += 1
            if insight['Winner'] == insight['OverallFastest'] and insight['Winner'] == insight['OverallMostConsistent']:
                winner_dominant_count += 1

            # Update all_dominant_performers_season by intersecting with current race's dominant performers
            if insight['DominantPerformers'] is not None:
                all_dominant_performers_season = all_dominant_performers_season.intersection(set(insight['DominantPerformers']))
            else:
                # If a race has no dominant performers, then no one can be dominant across all races
                all_dominant_performers_season = set()
                break

            for driver in insight['FastButNotFinished']:
                all_fast_but_not_finished[driver] = all_fast_but_not_finished.get(driver, 0) + 1
            for driver in insight['ConsistentButNotFinished']:
                all_consistent_but_not_finished[driver] = all_consistent_but_not_finished.get(driver, 0) + 1
            for driver in insight['ResultWithoutPaceConsistency']:
                all_result_without_pace_consistency[driver] = all_result_without_pace_consistency.get(driver, 0) + 1

        print(f"\n--- Winner Performance Trends ---")
        print(f"Out of {total_races_analyzed} races analyzed:")
        print(f"  - Race winner was also the fastest driver (average pace): {winner_fastest_count} time(s)")
        print(f"  - Race winner was also the most consistent driver: {winner_consistent_count} time(s)")
        print(f"  - Race winner was both fastest AND most consistent (dominant performance): {winner_dominant_count} time(s)")

        print(f"\n--- Season-Wide Cross-Category Insights ---")
        if all_dominant_performers_season:
            print(f"Drivers dominant across ALL analyzed races (Top 5 in Finish, Pace, & Consistency in every race): {', '.join(all_dominant_performers_season)}")
        else:
            print("No drivers were consistently in Top 5 across all three categories throughout ALL analyzed races.")

        print("\nDrivers frequently fastest but not finishing in Top 5 (potential bad luck/strategy issues):")
        if all_fast_but_not_finished:
            for driver, count in sorted(all_fast_but_not_finished.items(), key=lambda item: item[1], reverse=True):
                print(f"  - {driver}: {count} time(s)")
        else:
            print("  None")

        print("\nDrivers frequently most consistent but not finishing in Top 5 (potential car limitations/other factors):")
        if all_consistent_but_not_finished:
            for driver, count in sorted(all_consistent_but_not_finished.items(), key=lambda item: item[1], reverse=True):
                print(f"  - {driver}: {count} time(s)")
        else:
            print("  None")

        print("\nDrivers frequently finishing in Top 5 without top pace/consistency (strategy masterclass/attrition):")
        if all_result_without_pace_consistency:
            for driver, count in sorted(all_result_without_pace_consistency.items(), key=lambda item: item[1], reverse=True):
                print(f"  - {driver}: {count} time(s)")
        else:
            print("  None")

        print("\n==================================================")