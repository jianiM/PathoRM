import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import jaccard_score


file_path = r"D:\PathoRM\m6a\m6A_information.xlsx"    # 修改为你的Excel文件路径
df = pd.read_excel(file_path)


base_encoding = {
    'A': (1, 1, 1),
    'C': (0, 0, 1),
    'G': (1, 0, 0),
    'T': (0, 1, 0)
}


def encode_sequence(seq):
    return np.array([base_encoding[base] for base in seq])

df["Merged_Sequence"] = df["Reference Sequence"] + df["Alternative Sequence"]

merged_seqs = df["Reference Sequence"] + df["Alternative Sequence"]
merged_seqs = merged_seqs.tolist()

encoded_seqs = np.zeros((len(merged_seqs),390))

for i in range(len(merged_seqs)):
    seq = merged_seqs[i]
    encoded_seq = encode_sequence(seq).flatten()
    encoded_seqs[i] = encoded_seq

def calculate_jaccard(seq1, seq2):
    seq1_flat = seq1.flatten()
    seq2_flat = seq2.flatten()
    return jaccard_score(seq1_flat, seq2_flat)

num_sequences = len(encoded_seqs)
jaccard_matrix = np.zeros((num_sequences, num_sequences))

for i in range(num_sequences):
    for j in range(i, num_sequences): 
        similarity = calculate_jaccard(encoded_seqs[i], encoded_seqs[j])
        jaccard_matrix[i, j] = similarity
        jaccard_matrix[j, i] = similarity  
            
jaccard_df = pd.DataFrame(jaccard_matrix)
jaccard_df.to_excel("D:\PathoRM\m6a\jaccard_similarity.xlsx")



