from transformers import BertModel, BertTokenizer
import torch
import pandas as pd 

# Load pre-trained BioBERT model and tokenizer
model_name = "dmis-lab/biobert-v1.1"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Load the Excel file
excel_file_path = "/home/amber/mulga/m6a_experiments/extracting_site_semantic_features/site_information.xlsx"
df = pd.read_excel(excel_file_path)
#column_names = df.columns.tolist()
# Create an empty dictionary to store embeddings
site_seq = df["seq"].values[1]
print('site_seq:',site_seq)
inputs = tokenizer(site_seq, return_tensors="pt")
with torch.no_grad():
        outputs = model(**inputs,output_attentions=True)    
        print('output:',outputs)
        #embeddings = outputs.last_hidden_state[:, 0, :].numpy()

# embeddings_dict = {}

# for index, row in df.iterrows():
#     site_id = row['site_id']
#     site_seq = row['seq']
#     inputs = tokenizer(site_seq, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)    
#     embeddings = outputs.last_hidden_state[:, 0, :].numpy()
#     embeddings_dict[site_id] = embeddings

# print('embeddings_dict:',embeddings_dict)
# torch.save(embeddings_dict, "/home/amber/mulga/m6a_experiments/extracting_site_semantic_features/rna_embeddings.pt")






## Your RNA sequence
#rna_sequence = "GTATAACATCATTAAACAATTAAATTCTATAAGCTGTATTAATTCTTGGAGTTATGAAATTTTAA"
#
## Tokenize and convert to IDs
#inputs = tokenizer(rna_sequence, return_tensors="pt")
#input_ids = inputs["input_ids"]
#
## Generate embeddings
#with torch.no_grad():
#    outputs = model(**inputs)
#
## Extract the embeddings for the [CLS] token
#embeddings = outputs.last_hidden_state[:, 0, :].numpy()
#
#print(embeddings)
