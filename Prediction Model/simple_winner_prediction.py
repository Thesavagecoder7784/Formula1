import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np

# Enable the cache for faster data loading on subsequent runs
fastf1.Cache.enable_cache('cache/')

# Set up plotting style (optional, as this script is more data-focused)
fastf1.plotting.setup_mpl()

def get_race_data(year, round_num):
    """
    Helper function to get qualifying and race results, and race pace data for a given race.
    Returns a dictionary with driver performance data.
    """
    data = {}
    try:
        # Get event schedule to find Grand Prix name
        schedule = fastf1.get_event_schedule(year)
        event = schedule.loc[schedule['RoundNumber'] == round_num].iloc[0]
        grand_prix = event['EventName']

        # Load Qualifying session
        quali_session = fastf1.get_session(year, grand_prix, 'Qualifying')
        quali_session.load(telemetry=False, weather=False, messages=False)

        # Load Race session
        race_session = fastf1.get_session(year, grand_prix, 'Race')
        race_session.load(telemetry=False, weather=False, messages=False)

        # Get all drivers who participated in the race
        all_drivers = race_session.drivers

        # Get fastest race pace for normalization
        all_race_laps = race_session.laps.pick_accurate().pick_track_status('1') # Only green flag laps
        if not all_race_laps.empty:
            all_race_laps['LapTime(s)'] = all_race_laps['LapTime'].dt.total_seconds()
            fastest_race_pace_overall = all_race_laps['LapTime(s)'].min()
        else:
            fastest_race_pace_overall = None

        for driver_id in all_drivers:
            driver_code = race_session.get_driver(driver_id)['Abbreviation']

            # Qualifying Position
            quali_pos = None
            driver_quali_results = quali_session.results.loc[quali_session.results['Abbreviation'] == driver_code]
            if not driver_quali_results.empty:
                quali_pos = driver_quali_results.iloc[0]['Position']

            # Race Position
            race_pos = None
            driver_race_results = race_session.results.loc[race_session.results['Abbreviation'] == driver_code]
            if not driver_race_results.empty:
                race_pos = driver_race_results.iloc[0]['Position']

            # Race Pace (median of valid laps)
            race_pace = None
            driver_race_laps = race_session.laps.pick_driver(driver_code).pick_accurate().pick_track_status('1')
            if not driver_race_laps.empty:
                driver_race_laps['LapTime(s)'] = driver_race_laps['LapTime'].dt.total_seconds()
                race_pace = driver_race_laps['LapTime(s)'].median()

            # Relative Race Pace
            race_pace_relative = None
            if race_pace is not None and fastest_race_pace_overall is not None:
                race_pace_relative = ((race_pace - fastest_race_pace_overall) / fastest_race_pace_overall) * 100

            data[driver_code] = {
                'QualiPosition': quali_pos,
                'RacePosition': race_pos,
                'RacePaceRelative': race_pace_relative
            }
    except Exception as e:
        print(f"Error loading data for {year} Round {round_num}: {e}")
    return data

def get_next_race_to_predict(year):
    """
    Finds the next race to predict based on the current date.
    """
    schedule = fastf1.get_event_schedule(year)
    today = pd.to_datetime('today')
    
    # Find the first event that is in the future
    future_events = schedule[schedule['EventDate'] > today]
    
    if future_events.empty:
        print(f"No future races found for the {year} season.")
        return None, None
        
    next_race = future_events.iloc[0]
    return next_race['RoundNumber'], next_race['EventName']

def predict_winner_simple(year, grand_prix_round_to_predict, event_name):
    """
    Predicts the race winner based on season-to-date average performance
    (Qualifying Position, Race Position, Relative Race Pace) for races prior to the target round.
    Introduces recency weighting for performance metrics.

    Args:
        year (int): The year of the F1 season.
        grand_prix_round_to_predict (int): The round number of the Grand Prix to predict (1-indexed).
        event_name (str): The name of the event to predict.
    """
    print(f"\n--- Predicting Winner for {year} {event_name} (Round {grand_prix_round_to_predict}) ---")

    all_drivers_performance = {}

    # Collect data from previous races in the season
    for round_num in range(1, grand_prix_round_to_predict):
        print(f"Collecting data from {year} Round {round_num}...")
        race_data = get_race_data(year, round_num)
        for driver_code, perf_data in race_data.items():
            if driver_code not in all_drivers_performance:
                all_drivers_performance[driver_code] = {
                    'QualiPositions': [],
                    'RacePositions': [],
                    'RacePaceRelatives': []
                }
            
            # Store performance with the round number for weighting
            if perf_data['QualiPosition'] is not None:
                all_drivers_performance[driver_code]['QualiPositions'].append((round_num, perf_data['QualiPosition']))
            if perf_data['RacePosition'] is not None:
                all_drivers_performance[driver_code]['RacePositions'].append((round_num, perf_data['RacePosition']))
            if perf_data['RacePaceRelative'] is not None:
                all_drivers_performance[driver_code]['RacePaceRelatives'].append((round_num, perf_data['RacePaceRelative']))

    driver_scores = []
    for driver_code, data in all_drivers_performance.items():
        
        # Weighted average calculation - more recent races get higher weight
        def weighted_avg(perf_list):
            if not perf_list:
                return np.nan
            total_weight = sum(item[0] for item in perf_list) # sum of round numbers
            weighted_sum = sum(item[1] * item[0] for item in perf_list) # value * round_num
            return weighted_sum / total_weight

        avg_quali_pos = weighted_avg(data['QualiPositions'])
        avg_race_pos = weighted_avg(data['RacePositions'])
        avg_race_pace_rel = weighted_avg(data['RacePaceRelatives'])

        # Simple scoring: lower is better for positions and relative pace.
        score = 0
        weights = {'quali_pos': 0.4, 'race_pos': 0.4, 'race_pace': 0.2}

        if not np.isnan(avg_quali_pos):
            score += avg_quali_pos * weights['quali_pos']
        else:
            score += 20 * weights['quali_pos'] # Penalize missing quali data

        if not np.isnan(avg_race_pos):
            score += avg_race_pos * weights['race_pos']
        else:
            score += 20 * weights['race_pos'] # Penalize missing race data

        if not np.isnan(avg_race_pace_rel):
            score += avg_race_pace_rel * weights['race_pace']
        else:
            score += 10 * weights['race_pace'] # Penalize missing pace data

        driver_scores.append({
            'Driver': driver_code,
            'AvgQualiPos': avg_quali_pos,
            'AvgRacePos': avg_race_pos,
            'AvgRacePaceRel': avg_race_pace_rel,
            'PredictionScore': score
        })

    if not driver_scores:
        print("No sufficient data to make a prediction.")
        return

    predictions_df = pd.DataFrame(driver_scores)
    predictions_df.sort_values(by='PredictionScore', ascending=True, inplace=True)

    print("\n--- Prediction Results (Top 5) ---")
    print(predictions_df.head(5).to_string(index=False))

    if not predictions_df.empty:
        print(f"\nPredicted Winner for {year} {event_name}: {predictions_df.iloc[0]['Driver']}")

# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration for Prediction ---
    # Set the year and round number for the race you want to predict.
    # If you set YEAR_TO_PREDICT or ROUND_TO_PREDICT to None, the script will find the next upcoming race automatically.
    YEAR_TO_PREDICT = 2025      # Example: 2023, or None
    ROUND_TO_PREDICT = 12       # Example: 10 for the 10th race, or None

    year = None
    race_round = None
    event_name = None

    if YEAR_TO_PREDICT is not None and ROUND_TO_PREDICT is not None:
        year = YEAR_TO_PREDICT
        race_round = ROUND_TO_PREDICT
        try:
            schedule = fastf1.get_event_schedule(year)
            event = schedule.loc[schedule['RoundNumber'] == race_round]
            if not event.empty:
                event_name = event.iloc[0]['EventName']
            else:
                print(f"Error: Round {race_round} not found for the {year} season.")
                exit()
        except Exception as e:
            print(f"An error occurred while fetching event details for {year} Round {race_round}: {e}")
            exit()
    else:
        # Automatically find the next race to predict
        year = pd.to_datetime('today').year
        race_round, event_name = get_next_race_to_predict(year)

    # --- Run Prediction ---
    if race_round and event_name:
        if race_round > 1:
            predict_winner_simple(year=year, grand_prix_round_to_predict=race_round, event_name=event_name)
        else:
            print("Cannot make a prediction for the first race of the season (Round 1).")
    else:
        print("Could not determine the race to predict. Check configuration or season schedule.")
