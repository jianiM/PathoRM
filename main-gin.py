# -*- coding: utf-8 -*-

"""
Created on Sun May 29 17:55:57 2022
@author: Jiani Ma
task: m6a-disease association prediction
"""
import os
import numpy as np 
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from sklearn.model_selection import KFold
from utils import * 
from data_reading import read_data
from models.gin import *
import torch 
import torch.nn as nn
import torch.nn.functional as F
from train_test_split import kf_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import argparse
seed = 1024
np.random.seed(seed)
torch.manual_seed(seed)


# Check if GPU (CUDA) is available and set random seed for CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="This is a template of machine learning developping source code.")
    parser.add_argument('-hgcn_dim', '--hgcn_dim_topofallfeature', type=int, nargs='?', default=200,help='defining the size of hidden layer of GCN.')
    parser.add_argument('-dropout', '--dropout_topofallfeature', type=float, nargs='?', default=0.5,help='ratio of drop the graph nodes.') 
    return parser.parse_args()


def site_feature_concate(file_path):
    data_dict = torch.load(file_path)
    matrix = np.array(list(data_dict.values()))
    new_matrix = np.squeeze(matrix, axis=1)
    return new_matrix 


def disease_feature_concate(file_path):
    data_dict = torch.load(file_path)
    matrix = np.array(list(data_dict.values()))
    return matrix 


if __name__=="__main__":
    args = parse_args()    
    device = 'cuda:0'
    hgcn_dim = args.hgcn_dim_topofallfeature
    dropout = args.dropout_topofallfeature
   
    lr = 0.0001
    topk = 1  
    epoch_num = 200
    
    data_folder = "/home/amber/datasets/m6a/"
    drug_sim_path = "./m6a_jaccard_sim.xlsx"
    target_sim_path = "./disease_sim_mat.xlsx"
    DTI_path = "m6a_disease_mat.xlsx"
    
    """
    data reading
    """
    SR,SD,A_orig,A_orig_arr,known_sample = read_data(data_folder,drug_sim_path,target_sim_path,DTI_path)    

    """
    globalize the drug affinity matrix and target affinity matrix 
    """    
    A_unknown_mask = 1 - A_orig    
    drug_num = A_orig.shape[0]
    target_num = A_orig.shape[1]
    A_orig_list = A_orig.flatten()     
    drug_dissimmat = get_drug_dissimmat(SR,topk)

    """
    performing k-fold
    """
    n_splits = 10
    train_all, test_all = kf_split(known_sample,n_splits)    
   
    """
    all those unknown samples are negative samples:
        find their indice in A_orig 
        negative_index_arr : numpy format
        negative_index : tensor format 
    """
    negtive_index_arr = np.where(A_orig_arr==0)[0]
    negative_index = torch.LongTensor(negtive_index_arr)   # all negative samples
    test_auc = np.zeros(n_splits)
    test_aupr = np.zeros(n_splits)
    test_f1_score = np.zeros(n_splits)
    test_accuracy = np.zeros(n_splits)
    test_recall = np.zeros(n_splits)
    test_specificity = np.zeros(n_splits)
    test_precision = np.zeros(n_splits)
    """
    feature concate 
    """
    site_feature_path = "/home/amber/mulga/m6a_experiments/extracting_site_semantic_features/rna_embeddings.pt"
    site_mat = site_feature_concate(site_feature_path) 
    disease_feature_path = "/home/amber/mulga/m6a_experiments/extracting_disease_semantic_embeddings/disease_embeddings_dict.pt" 
    disease_mat = disease_feature_concate(disease_feature_path)
    H = np.concatenate((site_mat, disease_mat), axis=0)
    H = torch.FloatTensor(H).to(device)   
    
    for fold_int in range(n_splits):
        print('fold_int:',fold_int)        
        A_train_id = train_all[fold_int]
        A_test_id = test_all[fold_int]    
        A_train = known_sample[A_train_id]
        A_test = known_sample[A_test_id]        
        A_train_tensor = torch.LongTensor(A_train)
        A_test_tensor = torch.LongTensor(A_test)
        A_train_list = np.zeros_like(A_orig_arr)
        A_train_list[A_train] = 1        
        A_test_list = np.zeros_like(A_orig_arr)
        A_test_list[A_test] = 1                                
        A_train_mask = A_train_list.reshape((A_orig.shape[0],A_orig.shape[1]))
        A_test_mask = A_test_list.reshape((A_orig.shape[0],A_orig.shape[1]))
        A_unknown_mask = 1 - A_orig            
        A_train_mat = A_train_mask    

        G = Construct_G(A_train_mat,SR,SD).to(device)
        # sample the negative samples         
        train_neg_mask_candidate = get_negative_samples(A_train_mask,drug_dissimmat)
        train_neg_mask = np.multiply(train_neg_mask_candidate, A_unknown_mask)
        train_negative_index = np.where(train_neg_mask.flatten() ==1)[0]
        training_negative_index = torch.tensor(train_negative_index)

        train_W = torch.randn(hgcn_dim, hgcn_dim).to(device)  
        train_W = nn.init.xavier_normal_(train_W)        
        # initizalize the model 

        gin_model = GIN_autoencoder(input_dim=H.size(1), hidden_dim=200, num_layers=2, train_W=train_W).to(device)
        gin_optimizer = torch.optim.Adam(list(gin_model.parameters()), lr=lr)
        fgm = FGM_GIN(gin_model)
        gin_model.train()
        for epoch in range(epoch_num):                             
            A_hat = gin_model(H, G, drug_num, target_num)                    
            A_hat_list = A_hat.view(1, -1)            
            train_sample = A_hat_list[0][A_train_tensor]                        
            train_score = torch.sigmoid(train_sample)                    
            nega_sample = A_hat_list[0][training_negative_index]
            nega_score = torch.sigmoid(nega_sample)
                 
            loss = loss_function(train_score, nega_score, drug_num, target_num)            

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

            gin_optimizer.zero_grad()
            loss.backward()

            fgm.attack(epsilon=1.0)  
            A_hat_adv = gin_model(H, G, drug_num, target_num)
            A_hat_list_adv = A_hat_adv.view(1, -1)
            train_sample_adv = A_hat_list_adv[0][A_train_tensor]
            train_score_adv = torch.sigmoid(train_sample_adv)
            nega_sample_adv = A_hat_list_adv[0][training_negative_index]
            nega_score_adv = torch.sigmoid(nega_sample_adv)

            loss_adv = loss_function(train_score_adv, nega_score_adv, drug_num, target_num)
            loss_adv.backward()
            fgm.restore()
            gin_optimizer.step()


        gin_model.eval()  
        test_neg_mask_candidate = get_negative_samples(A_test_mask,drug_dissimmat)
        test_neg_mask = np.multiply(test_neg_mask_candidate, A_unknown_mask)
        test_negative_index = np.where(test_neg_mask.flatten() ==1)[0]
        test_negative_index = torch.tensor(test_negative_index)        
        positive_samples = A_hat_list[0][A_test_tensor].detach().cpu().numpy()
        negative_samples = A_hat_list[0][test_negative_index].detach().cpu().numpy()    
        positive_labels = np.ones_like(positive_samples)
        negative_labels = np.zeros_like(negative_samples)            
        labels = np.hstack((positive_labels,negative_labels))
        scores = np.hstack((positive_samples,negative_samples))
        TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision = get_metric(labels,scores)  
        test_auc[fold_int] = auc
        test_aupr[fold_int] = aupr
        test_f1_score[fold_int] = f1_score
        test_accuracy[fold_int] = accuracy
        test_recall[fold_int] = recall
        test_specificity[fold_int] = specificity
        test_precision[fold_int] = precision  
        print('TP:',TP)
        print('FP:',FP)
        print('FN:',FN)
        print('TN:',TN)
        print('fpr:',fpr)
        print('tpr:',tpr)
        print('test_auc:',auc)
        print('test_aupr:',aupr)
        print('f1_score:',f1_score)
        print('accuracy:',accuracy)
        print('recall:',recall)
        print('specificity:',specificity)
        print('precision:',precision)
   
    mean_auroc = np.mean(test_auc)
    mean_aupr = np.mean(test_aupr)
    mean_f1 = np.mean(test_f1_score)
    mean_acc = np.mean(test_accuracy)  
    mean_recall = np.mean(test_recall)
    mean_specificity = np.mean(test_specificity)
    mean_precision = np.mean(test_precision)
    print('mean_auroc:',mean_auroc)
    print('mean_aupr:',mean_aupr)
    print('mean_f1:',mean_f1)
    print('mean_acc:',mean_acc)
    print('mean_recall:',mean_recall)
    print('mean_specificity:',mean_specificity)
    print('mean_precision:',mean_precision)
    std_auc = np.std(test_auc)
    std_aupr = np.std(test_aupr)
    std_f1 = np.std(test_f1_score)
    std_acc = np.std(test_accuracy)
    std_recall = np.std(test_recall)
    std_specificity = np.std(test_specificity)
    std_precision = np.std(test_precision)
    print('std_auc:',std_auc)
    print('std_aupr:',std_aupr)
    print('std_f1:',std_f1)
    print('std_acc:',std_acc)
    print('std_recall:',std_recall)
    print('std_specificity:',std_specificity)
    print('std_precision:',std_precision)
