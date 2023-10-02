import sys
sys.path.append('.')
import pandas as pd

def split_hard_data(csv_path, percentile_low, percentile_high, num_mutant):
    df = pd.read_csv(csv_path)
    print(df)
    scores_GFP = df['GFP'].tolist()
    scores_stability = df['stability'].tolist()
    scores_GFP.sort()
    # sort scores_stability in descending order
    scores_stability.sort(reverse=True)
    ref_score_GFP_low = scores_GFP[int(len(scores_GFP)*percentile_low)]
    ref_score_stability_low = scores_stability[int(len(scores_stability)*percentile_low)]
    ref_score_GFP_high = scores_GFP[int(len(scores_GFP)*percentile_high)]
    ref_score_stability_high = scores_stability[int(len(scores_stability)*percentile_high)]
    print('GFP:', scores_GFP[0], scores_GFP[-1], len(df[(ref_score_GFP_low <= df['GFP']) & (df['GFP'] <= ref_score_GFP_high)]))
    # input()
    print('stability:', scores_stability[0], scores_stability[-1], len(df[(ref_score_stability_high <= df['stability']) & (df['stability'] <= ref_score_stability_low)]))
    # input()
    print(ref_score_GFP_low, ref_score_GFP_high, ref_score_stability_low, ref_score_stability_high)
    filtered_df = df[(ref_score_GFP_low <= df['GFP']) & (df['GFP'] <= ref_score_GFP_high) & (ref_score_stability_high <= df['stability']) & (df['stability'] <= ref_score_stability_low)]
    print(filtered_df)
    filtered_df.to_csv(f'data/GFP_stability_percentile_{percentile_low}_{percentile_high}.csv', index=False)
    

if __name__ == '__main__':
    percentile_low = 0.2
    percentile_high = 0.4
    num_mutant = 3
    split_hard_data('data/ground_truth_GFP_stability.csv', percentile_low, percentile_high, num_mutant=num_mutant)
