import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import model as m

# Change this to just encode the day out of 365, and the year as an integer
def anonymize_data(df):
    length = len(df)
    num_zeros = length // 2
    num_ones = length - num_zeros

    array_zeros = np.zeros(num_zeros, dtype=int)
    array_ones = np.ones(num_ones, dtype=int)

    # Concatenate the arrays
    winners = np.concatenate((array_zeros, array_ones))
    np.random.shuffle(winners)

    df['winner'] = winners

    player_cols = ['id', 'hand', 'age', 'rank_points']
    stat_cols = ['first_serve_pt', 'first_serve_won', 'second_serve_won', 'double_faults', 'aces', 'break_points_saved', 'break_points_faced', 'return_first_serve_pt_won', 'return_second_serve_won', 'bp_converted', 'bp_opportunities', 'first_serve_pt', 'first_serve_won', 'second_serve_won', 'double_faults', 'break_points_saved', 'break_points_faced', 'return_first_serve_pt_won', 'return_second_serve_won', 'bp_converted', 'bp_opportunities', 'h2h', 'h2h', 'match_difficulty']

    player0_wins = df[df['winner'] == 0]
    player1_wins = df[df['winner'] == 1]

    # Rename the columns
    player0_wins = player0_wins.rename(columns={f'winner_{col}': f'player0_{col}' for col in player_cols})
    player0_wins = player0_wins.rename(columns={f'w_{col}': f'player0_{col}' for col in stat_cols})
    player0_wins = player0_wins.rename(columns={f'loser_{col}': f'player1_{col}' for col in player_cols})
    player0_wins = player0_wins.rename(columns={f'l_{col}': f'player1_{col}' for col in stat_cols})

    player1_wins = player1_wins.rename(columns={f'winner_{col}': f'player1_{col}' for col in player_cols})
    player1_wins = player1_wins.rename(columns={f'w_{col}': f'player1_{col}' for col in stat_cols})
    player1_wins = player1_wins.rename(columns={f'loser_{col}': f'player0_{col}' for col in player_cols})
    player1_wins = player1_wins.rename(columns={f'l_{col}': f'player0_{col}' for col in stat_cols})

    anonymized_df = pd.concat([player0_wins, player1_wins])
    anonymized_df.set_index(['tourney_id', 'match_num'], inplace=True)
    anonymized_df.sort_index(inplace=True)
    anonymized_df.reset_index(inplace=True)
    anonymized_df.to_csv('./data/atp_matches_1991-2023_processed.csv', index=False, encoding='utf-8')
    return anonymized_df

def encode_dates(df):
    df['day_sin'] = np.sin(2 * np.pi * df['day']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/365)
    return df

def encode_data(df):
    # Encode data
    # Update hand to a boolean value (0 = R, 1 = L, 2= unknown)
    df['winner_hand'] = df['winner_hand'].map({'R': 0, 'L': 1, 'U': 2})
    df['loser_hand'] = df['loser_hand'].map({'R': 0, 'L': 1, 'U':2})

    # Update surface to a one hot encoding
    df = pd.get_dummies(df, columns=['surface'])

    # Update round to a label encoding
    df['round'] = df['round'].map({'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7})
    
    # Update tourney_level to a one hot encoding
    df = pd.get_dummies(df, columns=['tourney_level'])

    # Columns that need extra processing
    # tourney_date
    # First, convert tourney_date to a datetime object
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')

    df['day'] = df.tourney_date.dt.day
    df['year'] = df.tourney_date.dt.year

    df = encode_dates(df)

    #Drop the original columns
    df = df.drop(['day', 'tourney_date'], axis=1)
    # df.to_csv('./data/atp_matches_1991-2023_processed.csv', index=False, encoding='utf-8')

    return df

def test_train_split_by_year(data):
    ratio = 0
    year = 2022

    while ratio < 0.2:
        train = data[data['year'] < year]
        test = data[data['year'] >= year]
        test_count = len(test)
        total = len(data)
        ratio = test_count / total
        year -= 1

    train.to_csv('./data/test_train_split/train.csv', index=False, encoding='utf-8')
    test.to_csv('./data/test_train_split/test.csv', index=False, encoding='utf-8')

def test_train_split(x,y, seed=42):
    index = ['tourney_id', 'match_num']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)

    train.set_index(index, inplace=True)
    test.set_index(index, inplace=True)
    train.sort_index(inplace=True)
    test.sort_index(inplace=True)
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)

    train.drop(columns=index, inplace=True)
    test.drop(columns=index, inplace=True)

    train.to_csv('./data/test_train_split/train.csv', index=False, encoding='utf-8')
    test.to_csv('./data/test_train_split/test.csv', index=False, encoding='utf-8')

    # y_test.to_csv('./data/y_test.csv', index=False, encoding='utf-8')
    return x_train, x_test, y_train, y_test


def preprocess_data():
    #TODO: Add loser aces back once preprocessing is done
    new_cols = ['w_first_serve_pt', 'w_first_serve_won', 'w_second_serve_won', 'w_double_faults', 'w_aces', 'w_break_points_saved', 'w_break_points_faced', 'w_return_first_serve_pt_won', 'w_return_second_serve_won', 'w_bp_converted', 'w_bp_opportunities', 'l_first_serve_pt', 'l_first_serve_won', 'l_second_serve_won', 'l_aces', 'l_double_faults', 'l_break_points_saved', 'l_break_points_faced', 'l_return_first_serve_pt_won', 'l_return_second_serve_won', 'l_bp_converted', 'l_bp_opportunities', 'w_h2h', 'l_h2h', 'w_match_difficulty', 'l_match_difficulty']
    old_cols = ['tourney_id', 'tourney_date', 'match_num', 'surface', 'draw_size', 'tourney_level', 'winner_hand', 'loser_hand', 'best_of', 'round', 'winner_age', 'loser_age',  'winner_rank_points', 'loser_rank_points', 'winner_id', 'loser_id']
    features = new_cols + old_cols
    df = pd.read_csv('data/atp_matches_1991-2023_with_stats_3.csv', usecols=features)
    df = df.dropna(subset=features+['winner_rank_points', 'loser_rank_points'])
    df = encode_data(df)
    df = anonymize_data(df)
    x = df.iloc[:, :-1]
    y = df['winner']
    x_train, x_test, y_train, y_test = test_train_split(x,y)
    return x_train, x_test, y_train, y_test

def get_config():
    config_path = "./src/config.json"
    invalid = False

    if not os.path.isfile(config_path):
        print('Invalid path to config file!')
        exit(1)

    with open(config_path) as json_file:
        config = json.load(json_file)

        if 'model' not in config:
            print('Missing required argument (model)!')
            invalid = True

        if 'cv' not in config:
            print('Missing required argument (cv)!')
            invalid = True
        
        if 'test' not in config:
            print('Missing required argument (test)!')
            invalid = True
        
        # if 'epochs' not in config:
        #     print('Missing required argument (epochs)!')
        #     invalid = True

        # if 'batch' not in config:
        #     print('Missing required argument (batch)!')
        #     invalid = True

        if invalid:
            exit(1)

        config = argparse.Namespace(**config)
        return bool(config.test), bool(config.cv), config.model, bool(config.time_split)
