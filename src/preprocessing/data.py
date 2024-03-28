from collections import deque
import time
import pandas as pd
import numpy as np

class CircularBuffer:
    def __init__(self, max_length):
        self.buffer = deque(maxlen=max_length)

    def add(self, item):
        self.buffer.append(item)

    def get_buffer(self):
        return list(self.buffer)

tourney_levels = {
    'A': 1,
    'M': 2,
    'G': 3
}

def calculate_stats(row, last_10_matches, is_winner):
    total_first_serve_in = 0
    total_first_serve_won = 0
    total_second_serve_won = 0
    total_double_faults = 0
    total_aces = 0
    total_break_points_saved = 0
    total_break_points_faced = 0
    total_service_games = 0
    total_service_points = 0
    total_other_serve_pts = 0
    total_other_first_serve_in = 0
    total_other_first_serve_won = 0
    total_other_second_serve_won = 0
    total_other_break_points_faced = 0
    total_other_break_points_saved = 0
    total_other_service_games = 0
    total_other_double_faults = 0

    current_match_level = row['tourney_level']
    current_match_level_rating = tourney_levels.get(current_match_level, 0)
    sum_weights = 0

    for i, match in enumerate(last_10_matches):
        level_rating = tourney_levels.get(match['level'], 0)
        recency_weight = 1

        index = i + 1

        tourney_level_weight = 1 if current_match_level_rating == 0 or level_rating == 0 else 2 ** (level_rating - current_match_level_rating)
        sum_weights += tourney_level_weight

        if index < 3:
            recency_weight = 0.5
        elif index < 6:
            recency_weight = 0.75
        elif index < 9:
            recency_weight = 0.9

        total_first_serve_in += match['first_serve_in'] * recency_weight * tourney_level_weight
        total_first_serve_won += match['first_serve_won'] * recency_weight * tourney_level_weight
        total_second_serve_won += match['second_serve_won']* recency_weight * tourney_level_weight
        total_double_faults += match['double_faults']* recency_weight * tourney_level_weight
        total_aces += match['aces']* recency_weight * tourney_level_weight
        total_break_points_saved += match['break_points_saved']* recency_weight * tourney_level_weight
        total_break_points_faced += match['break_points_faced']* recency_weight * tourney_level_weight
        total_service_games += match['service_games']* recency_weight * tourney_level_weight
        total_service_points += match['service_points']* recency_weight * tourney_level_weight
        total_other_serve_pts += match['other_serve_pts']* recency_weight * tourney_level_weight
        total_other_first_serve_in += match['other_first_serve_in']* recency_weight * tourney_level_weight
        total_other_first_serve_won += match['other_first_serve_won']* recency_weight * tourney_level_weight
        total_other_second_serve_won += match['other_second_serve_won']* recency_weight * tourney_level_weight
        total_other_break_points_faced += match['other_break_points_faced']* recency_weight * tourney_level_weight
        total_other_break_points_saved += match['other_break_points_saved']* recency_weight * tourney_level_weight
        total_other_service_games += match['other_service_games']* recency_weight * tourney_level_weight
        total_other_double_faults += match['other_double_faults']* recency_weight * tourney_level_weight

    first_serve_pt =   total_first_serve_in / total_service_points if total_service_points > 0 else np.nan
    first_serve_won = total_first_serve_won / total_first_serve_in if total_first_serve_in > 0 else np.nan
    second_serve_won = total_second_serve_won / (total_service_points - total_first_serve_in - total_double_faults) if (total_service_points - total_first_serve_in - total_double_faults) > 0 else np.nan
    double_faults = total_double_faults / total_service_points if total_service_points > 0 else np.nan
    aces = total_aces / total_service_points if total_service_points > 0 else np.nan
    break_points_saved = total_break_points_saved / total_break_points_faced if total_break_points_faced > 0 else np.nan
    break_points_faced = total_break_points_faced / total_service_games if total_service_games > 0 else np.nan
    return_first_serve_pt_won = 1 - (total_other_first_serve_won / total_other_first_serve_in) if total_other_first_serve_in > 0 else np.nan
    return_second_serve_won = 1 - (total_other_second_serve_won / (total_other_serve_pts - total_other_first_serve_in - total_other_double_faults)) if (total_other_serve_pts - total_other_first_serve_in - total_other_double_faults) > 0 else np.nan
    bp_converted = (total_other_break_points_faced - total_other_break_points_saved) / total_other_break_points_faced if total_other_break_points_faced > 0 else np.nan
    bp_opportunities = total_other_break_points_faced / total_other_service_games if total_other_service_games > 0 else np.nan

    prefix = 'w_' if is_winner else 'l_'
    row[prefix + 'first_serve_pt'] = first_serve_pt
    row[prefix + 'first_serve_won'] = first_serve_won
    row[prefix + 'second_serve_won'] = second_serve_won
    row[prefix + 'double_faults'] = double_faults
    row[prefix + 'aces'] = aces
    row[prefix + 'break_points_saved'] = break_points_saved
    row[prefix + 'break_points_faced'] = break_points_faced
    row[prefix + 'return_first_serve_pt_won'] = return_first_serve_pt_won
    row[prefix + 'return_second_serve_won'] = return_second_serve_won
    row[prefix + 'bp_converted'] = bp_converted
    row[prefix + 'bp_opportunities'] = bp_opportunities
    row[prefix + 'match_difficulty'] = sum_weights

def process_match(row, stats,player,h2h):
    winner_id = row['winner_id']
    loser_id = row['loser_id']
    is_winner = winner_id == player
    prefix = 'w_' if is_winner else 'l_'
    other_prefix = 'l_' if is_winner else 'w_'

    row[prefix + 'h2h'] = h2h.get(loser_id if is_winner else winner_id, 0)

    if is_winner:
        if loser_id in h2h:
            h2h[loser_id] += 1
        else:
            h2h[loser_id] = 1

    matches = stats.get_buffer()

    calculate_stats(row, matches, is_winner)

    current_match = {
        'first_serve_in': row[prefix + '1stIn'],
        'first_serve_won': row[prefix + '1stWon'],
        'second_serve_won': row[prefix + '2ndWon'],
        'double_faults': row[prefix + 'df'],
        'aces': row[prefix + 'ace'],
        'break_points_saved': row[prefix + 'bpSaved'],
        'break_points_faced': row[prefix + 'bpFaced'],
        'service_games': row[prefix + 'SvGms'],
        'service_points': row[prefix + 'svpt'],
        'other_serve_pts': row[other_prefix + 'svpt'],
        'other_first_serve_in': row[other_prefix +'1stIn'],
        'other_first_serve_won': row[other_prefix +'1stWon'],
        'other_second_serve_won': row[other_prefix +'2ndWon'],
        'other_break_points_faced': row[other_prefix +'bpFaced'],
        'other_break_points_saved': row[other_prefix +'bpSaved'],
        'other_service_games': row[other_prefix +'SvGms'],
        'other_double_faults': row[other_prefix + 'df'],
        'level': row['tourney_level']
    }

    stats.add(current_match)
    return row

def main():
    data_path = './data/atp_matches_1991-2023.csv'
    all_matches = pd.read_csv(data_path)

    all_matches = all_matches.dropna(subset=['winner_id', 'loser_id', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_df', 'w_ace', 'w_bpSaved', 'w_bpFaced', 'w_SvGms', 'w_svpt', 'l_svpt', 'l_1stWon', 'l_2ndWon', 'l_bpFaced', 'l_bpSaved', 'l_SvGms'])
    all_matches = all_matches[all_matches['score'].str.contains('RET') == False]
    all_matches.set_index(['tourney_id', 'match_num'], inplace=True, drop=False)

    new_cols = ['w_first_serve_pt', 'w_first_serve_won', 'w_second_serve_won', 'w_double_faults', 'w_aces', 'w_break_points_saved', 'w_break_points_faced', 'w_return_first_serve_pt_won', 'w_return_second_serve_won', 'w_bp_converted', 'w_bp_opportunities', 'l_first_serve_pt', 'l_first_serve_won', 'l_second_serve_won', 'l_double_faults', 'l_aces', 'l_break_points_saved', 'l_break_points_faced', 'l_return_first_serve_pt_won', 'l_return_second_serve_won', 'l_bp_converted', 'l_bp_opportunities', 'w_h2h', 'l_h2h', 'w_match_difficulty', 'l_match_difficulty']

    old_cols = ['w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_df', 'w_ace', 'w_bpSaved', 'w_bpFaced', 'w_SvGms', 'w_svpt', 'l_svpt', 'l_1stWon', 'l_2ndWon', 'l_bpFaced', 'l_bpSaved', 'l_SvGms', 'l_ace', 'l_df', 'l_1stIn']
    
    for col in new_cols:
        all_matches[col] = np.nan

    all_ids = pd.concat([all_matches['winner_id'], all_matches['loser_id']])
    player_counts = all_ids.value_counts()

    player_counts = player_counts[player_counts > 10]

    players = player_counts.index.unique().tolist()

    filtered_df = all_matches[all_matches['winner_id'].isin(players) | all_matches['loser_id'].isin(players)]

    pd.options.mode.copy_on_write = "warn"

    start = time.time()
    for idx, player in enumerate(players):
        player_matches = filtered_df[(filtered_df['winner_id'] == player) | (filtered_df['loser_id'] == player)]
        stats = CircularBuffer(10)
        h2h = {}

        print(f'{idx+1}/{len(players)}')

        player_matches = player_matches.apply(process_match, axis=1, args=(stats, player, h2h))
        player_matches.set_index(['tourney_id', 'match_num'], inplace=True, drop=True)
        all_matches.update(player_matches)

    end = time.time()
    print((end - start) / 60)
    all_matches.dropna(subset=new_cols, inplace=True)
    all_matches.drop(columns=old_cols, inplace=True)
    all_matches.to_csv('./data/atp_matches_1991-2023_with_stats_3.csv')

def check_data():
    data_path = './data/atp_matches_1991-2023_with_stats.csv'
    all_matches = pd.read_csv(data_path)
    rafa_novak = all_matches[(all_matches['winner_id'] == 104745) & (all_matches['loser_id'] == 104925) | (all_matches['winner_id'] == 104925) & (all_matches['loser_id'] == 104745)]
    rafa_novak.to_csv('./data/rafa_novak.csv')