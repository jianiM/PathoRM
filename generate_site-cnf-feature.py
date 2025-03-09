import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import jaccard_score


file_path = "D:/PathoRM/m6a/m6A_information.xlsx"   
df = pd.read_excel(file_path)


def compute_cumulative_frequency(seq):
   
    base_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}  
    cumulative_freq = []
    
    for i, base in enumerate(seq, start=1):
        base_counts[base] += 1
        freq = base_counts[base] / i
        cumulative_freq.append(freq)
    
    return np.array(cumulative_freq)


ref_seqs = df["Reference Sequence"].tolist() 
alt_seqs = df["Alternative Sequence"].tolist()

cumu_prob_feature = np.zeros((len(ref_seqs),130))
cumu_prob_feature_df = pd.DataFrame(cumu_prob_feature)
cumu_prob_feature_df.to_excel("D:/PathoRM/m6a/cumu_prob_feature.xlsx")
