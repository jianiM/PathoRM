# PathoRM

## Resources:
+ README.md: this file.
....
....

## Datasets 
### m7GDA
m7GDA was downloded and curated from m7GHub database,768 associations among 741 m7G sites and 177 diseases.
+ m7G_information.xlsx (includes site id, reference seqs and alternative seqs) 
+ m7G_disease_mat.xlsx (association matrix between m7G sites and diseases)
+ m7G_disease_map_list.xlsx (association mapping list of m7G sites and diseases )
+ disease_trait.xlsx (description of disease)

### m6ADA
m6ADA was downloded and curated from RMVar database, 2860 associations among131 pathogenic m6A sites and 1338 diseases.
+ m6A_information.xlsx (includes site id, reference seqs and alternative seqs) 
+ m6A_disease_mat.xlsx (association matrix between m6A sites and diseases)
+ m6A_disease_map_list.xlsx (association mapping list of m6A sites and diseases )
+ disease_trait.xlsx (description of disease)


## Step-by-step running  

### 0 Prepare conda enviroment and install Python libraries needed
+ conda create -n bio python=3.10
+ source activate bio 
+ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
+ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
+ pip install torch-geometric
+ pip install transformers torch

dependencies: 
   + python == 3.9.16 
   + torch == 1.13.1+cu116
   + torch-geometric == 2.2.0 
   + torch-scatter == 2.1.0+pt113cu116
   + torch-sparse == 0.6.16+pt113cu116
   + torchvision == 0.14.1+cu116
   + numpy == 1.24.2 
   + pandas == 1.5.3
   + scikit-learn == 1.2.2 
   + scipy == 1.10.1
   + networkx == 3.0
   + matplotlib == 3.7.1 
   + seaborn == 0.12.2

### Usage 
+ Feature Generation
+ generate the feature embeddings of rna methylation sites and diseases 
+ Running:
+    python extracting_site_features.py ---> rna_embeddings.pt 
+    python biobert_embeddings.py ---> disease_embeddings_dict.pt

### RNA methylation site-diseases association prediction with PathoRM
+ predict the associations of m7G/m6A sites-diseases with PathoRM-GCN,-GIN,-GraphSAGE
+ Running: 
+    python main-gcn.py 
+    python main-gin.py
+    python main-graphsage.py
+    --->A_hat_list.py(predicted m7G/m6A sites-diseases association matrix)
+    --->prediciton metrics including auroc, aupr, f1, acc,recall, specificity, precision 
