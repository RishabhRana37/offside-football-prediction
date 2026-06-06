import os
import pickle
import pandas as pd
import numpy as np

def main():
    workspace_dir = "/Users/rana/OFF SIDE"
    train_path = os.path.join(workspace_dir, 'train.csv')
    
    print("Loading training data...")
    # Load only the required columns to save memory and run fast
    cols = [
        'name_y', 'age', 'foot', 'position', 'sub_position', 
        'country_of_citizenship', 'home_club_name', 'market_value_before_match',
        'highest_market_value_in_eur', 'international_caps', 'international_goals',
        'avg_xG', 'avg_xA', 'avg_shots', 'minutes_ratio'
    ]
    
    train_df = pd.read_csv(train_path, usecols=cols)
    
    print("Grouping by player name to build profiles...")
    # Clean up names
    train_df = train_df.dropna(subset=['name_y'])
    
    # Calculate median/first for each player
    player_groups = train_df.groupby('name_y')
    
    player_profiles = {}
    total_players = len(player_groups)
    print(f"Processing {total_players} unique players...")
    
    # We can aggregate efficiently using pandas
    agg_dict = {
        'age': 'median',
        'foot': lambda x: x.mode().iloc[0] if not x.mode().empty else 'right',
        'position': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Attack',
        'sub_position': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Center-Forward',
        'country_of_citizenship': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
        'home_club_name': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
        'market_value_before_match': 'median',
        'highest_market_value_in_eur': 'median',
        'international_caps': 'median',
        'international_goals': 'median',
        'avg_xG': 'median',
        'avg_xA': 'median',
        'avg_shots': 'median',
        'minutes_ratio': 'median'
    }
    
    agg_df = player_groups.agg(agg_dict)
    
    # Fill any remaining NaNs with logical defaults
    agg_df['age'] = agg_df['age'].fillna(26.0)
    agg_df['market_value_before_match'] = agg_df['market_value_before_match'].fillna(1000000.0)
    agg_df['highest_market_value_in_eur'] = agg_df['highest_market_value_in_eur'].fillna(2000000.0)
    agg_df['international_caps'] = agg_df['international_caps'].fillna(0.0)
    agg_df['international_goals'] = agg_df['international_goals'].fillna(0.0)
    agg_df['avg_xG'] = agg_df['avg_xG'].fillna(0.1)
    agg_df['avg_xA'] = agg_df['avg_xA'].fillna(0.05)
    agg_df['avg_shots'] = agg_df['avg_shots'].fillna(1.0)
    agg_df['minutes_ratio'] = agg_df['minutes_ratio'].fillna(0.5)
    
    # Convert dataframe to dictionary
    player_profiles = agg_df.to_dict(orient='index')
    
    print("Saving player profiles...")
    profiles_path = os.path.join(workspace_dir, 'player_profiles.pkl')
    with open(profiles_path, 'wb') as f:
        pickle.dump(player_profiles, f)
        
    print("Successfully generated player_profiles.pkl!")

if __name__ == '__main__':
    main()
