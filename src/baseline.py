import pandas as pd

df = pd.read_csv('./data/rank/test.csv')

df.dropna(subset=['player0_rank', 'player1_rank'], inplace=True)

correct = 0
total = len(df)

for index, row in df.iterrows():
    winner = row['winner']
    player0_rank = row['player0_rank']
    player1_rank = row['player1_rank']
    if player0_rank < player1_rank and winner == 0:
        correct += 1
    elif player1_rank < player0_rank and winner == 1:
        correct += 1

print("ACCURACY:", correct / total, sep='\n')