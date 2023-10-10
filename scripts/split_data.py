import sys
sys.path.append('.')
import pandas as pd

def split_hard_data(csv_path, percentile_low, percentile_high, num_mutant):
    df = pd.read_csv(csv_path)
    # print(df)
    scores_GFP = df['gb1'].tolist()
    scores_stability = df['ddg'].tolist()
    stability_max, stability_min = max(scores_stability), min(scores_stability)
    scores_GFP.sort()
    print(f'max fitness: {scores_GFP[int(len(scores_GFP)*0.99)]}, 20%: {scores_GFP[int(len(scores_GFP)*0.99)]*0.2} 40%: {scores_GFP[int(len(scores_GFP)*0.99)]*0.4}')
    # sort scores_stability in descending order
    scores_stability.sort(reverse=True)
    ref_score_GFP_low = scores_GFP[int(len(scores_GFP)*percentile_low)]
    ref_score_stability_low = scores_stability[int(len(scores_stability)*percentile_low)]
    ref_score_GFP_high = scores_GFP[int(len(scores_GFP)*percentile_high)]
    ref_score_stability_high = scores_stability[int(len(scores_stability)*percentile_high)]
    nornmalize = lambda x: (x - stability_min) / (stability_max - stability_min)
    print(ref_score_stability_low, ref_score_stability_high, nornmalize(ref_score_stability_low), nornmalize(ref_score_stability_high))
    print(scores_stability[0], nornmalize(scores_stability[0]))
    print(scores_stability[-1], nornmalize(scores_stability[-1]))
    print(scores_stability[int(0.5*len(scores_stability))], nornmalize(scores_stability[int(0.5*len(scores_stability))]))
    print(scores_stability[int(0.1*len(scores_stability))], nornmalize(scores_stability[int(0.1*len(scores_stability))]))
    input()
    print('gb1:', scores_GFP[0], scores_GFP[-1], len(df[(ref_score_GFP_low <= df['gb1']) & (df['gb1'] <= ref_score_GFP_high)]))
    # input()
    print('ddg:', scores_stability[0], scores_stability[-1], len(df[(ref_score_stability_high <= df['ddg']) & (df['ddg'] <= ref_score_stability_low)]))
    # input()
    print(ref_score_GFP_low, ref_score_GFP_high, ref_score_stability_low, ref_score_stability_high)
    filtered_df = df[(ref_score_GFP_low <= df['gb1']) & (df['gb1'] <= ref_score_GFP_high) & (ref_score_stability_high <= df['ddg']) & (df['ddg'] <= ref_score_stability_low)]
    print(filtered_df)
    filtered_df.to_csv(f'data/gb1/gb1_ddg_percentile_{percentile_low}_{percentile_high}.csv', index=False)

def split_hard_data_value(csv_path, percentile_low, percentile_high, num_mutant):
    df = pd.read_csv(csv_path)
    # print(df)
    scores_GB1 = df['gb1'].tolist()
    scores_ddg = df['ddg'].tolist()
    stability_max, stability_min = max(scores_ddg), min(scores_ddg)
    scores_GB1.sort()
    print(f'max fitness: {scores_GB1[int(len(scores_GB1)*0.99)]}, 20%value: {scores_GB1[int(len(scores_GB1)*0.99)]*0.2} 40%value: {scores_GB1[int(len(scores_GB1)*0.99)]*0.4}')
    print(f'20%pos: {scores_GB1[int(len(scores_GB1)*0.2)]}, 40%pos: {scores_GB1[int(len(scores_GB1)*0.4)]}')
    # sort scores_stability in descending order
    scores_ddg.sort(reverse=True)
    ref_score_GFP_low = scores_GB1[int(len(scores_GB1)*0.99)] * percentile_low
    ref_score_stability_low = scores_ddg[0] - (scores_ddg[0] - scores_ddg[int(len(scores_ddg)*0.99)]) * percentile_low
    ref_score_GFP_high = scores_GB1[int(len(scores_GB1)*0.99)] * percentile_high
    ref_score_stability_high = scores_ddg[0] - (scores_ddg[0] - scores_ddg[int(len(scores_ddg)*0.99)]) * percentile_high
    normalize = lambda x: (x - stability_min) / (stability_max - stability_min)
    print(ref_score_stability_low, ref_score_stability_high, normalize(ref_score_stability_low), normalize(ref_score_stability_high))
    # print(scores_stability[0], nornmalize(scores_stability[0]))
    # print(scores_stability[-1], nornmalize(scores_stability[-1]))
    # print(scores_stability[int(0.5*len(scores_stability))], nornmalize(scores_stability[int(0.5*len(scores_stability))]))
    # print(scores_stability[int(0.1*len(scores_stability))], nornmalize(scores_stability[int(0.1*len(scores_stability))]))
    # input()
    print('gb1:', scores_GB1[0], scores_GB1[-1], len(df[(ref_score_GFP_low <= df['gb1']) & (df['gb1'] <= ref_score_GFP_high)]))
    gb1_df = df[(ref_score_GFP_low <= df['gb1']) & (df['gb1'] <= ref_score_GFP_high)]
    gb1_df_ddg = gb1_df['ddg'].tolist()
    gb1_df_ddg.sort(reverse=True)
    print(gb1_df_ddg[0], gb1_df_ddg[-1])
    print('gb1_df_ddg:', (scores_ddg[0] - gb1_df_ddg[0]) / (scores_ddg[0] - scores_ddg[int(len(scores_ddg)*0.99)]), (scores_ddg[0] - gb1_df_ddg[-1]) / (scores_ddg[0] - scores_ddg[int(len(scores_ddg)*0.99)]))
    # input()
    print('ddg:', scores_ddg[0], scores_ddg[-1], len(df[(ref_score_stability_high <= df['ddg']) & (df['ddg'] <= ref_score_stability_low)]))
    # input()
    print(ref_score_GFP_low, ref_score_GFP_high, ref_score_stability_low, ref_score_stability_high)
    filtered_df = df[(ref_score_GFP_low <= df['gb1']) & (df['gb1'] <= ref_score_GFP_high) & (ref_score_stability_high <= df['ddg']) & (df['ddg'] <= ref_score_stability_low)]
    print(filtered_df, len(filtered_df))
    gb1_df_ddg = filtered_df['ddg'].tolist()
    gb1_df_ddg.sort(reverse=True)
    # print(gb1_df_ddg[0], gb1_df_ddg[-1])
    # print('gb1_df_ddg:', (scores_ddg[0] - gb1_df_ddg[0]) / (scores_ddg[0] - scores_ddg[int(len(scores_ddg)*0.99)]), (scores_ddg[0] - gb1_df_ddg[-1]) / (scores_ddg[0] - scores_ddg[int(len(scores_ddg)*0.99)]))
    # filtered_df.to_csv(f'data/gb1/gb1_ddg_percentile_{percentile_low}_{percentile_high}_99value.csv', index=False)


if __name__ == '__main__':
    percentile_low = 0.0
    percentile_high = 0.4
    num_mutant = 3
    split_hard_data_value('data/gb1/ground_truth_gb1_ddg.csv', percentile_low, percentile_high, num_mutant=num_mutant)
