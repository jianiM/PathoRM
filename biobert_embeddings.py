from transformers import AutoModel, AutoTokenizer
import pandas as pd
import torch

model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


gwas_path = "/home/amber/datasets/m6a/disease_trait.xlsx"
df = pd.read_excel(gwas_path) 

embeddings_dict = {}

for index, row in df.iterrows():
    gwas_id = row['gwas_id']
    gwas_trait = row['gwas_trait']

    inputs = tokenizer(gwas_trait, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    embeddings_dict[gwas_id] = embedding
torch.save(embeddings_dict, "/home/amber/mulga/extracting_disease_semantic_embeddings/disease_embeddings_dict.pt")


