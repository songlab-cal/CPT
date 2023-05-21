import numpy as np
import pandas as pd
import unittest
from copy import deepcopy
from sklearn import linear_model

# Feature lists
eve_feats = ['EVE_scores_ASM', 'evolutionary_index_ASM']
eve_msa_feats = ['EVE_gap_freq', 'EVE_wt_freq', 'EVE_mut_freq']
msa_feats = ['100verte_gap_freq', '100verte_wt_freq', '100verte_mut_freq',
             '30mammal_gap_freq', '30mammal_wt_freq', '30mammal_mut_freq']
struct_feats = ['conditioned_wt_score', 'conditioned_mt_score']
mpnn_feats = ['vanilla_protmpnn_norm', 'ca_protmpnn_norm', 'ca_only_protmpnn_norm']
aa_feats = ['PP1', 'PP2', 'PP3', 
            'KF1', 'KF2', 'KF3', 'KF4', 'KF5', 'KF6', 'KF7', 'KF8', 'KF9', 'KF10', 
            'Z1', 'Z2', 'Z3', 'Z4', 'Z5',
            'F1', 'F2', 'F3', 'F4', 'F5', 'F6',
            'T1', 'T2', 'T3', 'T4', 'T5',
            'VHSE1', 'VHSE2', 'VHSE3', 'VHSE4', 'VHSE5', 'VHSE6', 'VHSE7', 'VHSE8',
            'ProtFP1', 'ProtFP2', 'ProtFP3', 'ProtFP4', 'ProtFP5', 'ProtFP6', 'ProtFP7', 'ProtFP8',
            'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6', 'ST7', 'ST8',
            'BLOSUM1', 'BLOSUM2', 'BLOSUM3', 'BLOSUM4', 'BLOSUM5', 'BLOSUM6', 'BLOSUM7', 'BLOSUM8', 'BLOSUM9', 'BLOSUM10',
            'MSWHIM1', 'MSWHIM2', 'MSWHIM3']

class CPTClassifier():
    '''
    Cross-Protein Transfer classifier
    '''
    def __init__(self, scale = True, l2 = 0, n_jobs = -1, ensemble = False, **kwargs):
        if l2 == 0:
            self.model_ = linear_model.LogisticRegression(penalty = 'none', class_weight = 'balanced',
                                                          solver = 'newton-cg', n_jobs = n_jobs, **kwargs)
        else:
            self.model_ = linear_model.LogisticRegression(penalty = 'l2', class_weight = 'balanced', C = l2,
                                                          solver = 'liblinear', n_jobs = n_jobs, **kwargs)
        self.fit_ = {}
        self.ensemble_ = ensemble
        self.scale_ = scale
        self.mean_ = 0
        self.var_ = 1
                
    def train(self, feat_mat_all, label, all_genes = None, feat_list = None):
        
        feat_mat = deepcopy(feat_mat_all)
        
        # Re-initialize fitted models
        self.fit_ = {}
        self.qnorm_fit_ = {}
        
        # By default use all genes and all features
        if all_genes == None:
            unittest.TestCase().assertCountEqual(feat_mat.keys(), label.keys(), 
                                                 'Gene lists of feature matrices and labels do not match.')
            all_genes = feat_mat.keys()
        
        if feat_list != None:
            for gene in all_genes:
                feat_mat[gene] = feat_mat[gene][feat_list]

        train_genes = all_genes
        
        if self.scale_ == True:
            
            feat_to_scale = feat_mat[all_genes[0]].columns.difference(msa_feats)
            
            if not feat_to_scale.empty:
                gene_stats = {'mean': {},
                          'var': {},
                          'n': {}}
                    
                for gene in train_genes:
                    gene_stats['mean'][gene] = feat_mat[gene][feat_to_scale].mean(axis = 0)
                    gene_stats['var'][gene] = feat_mat[gene][feat_to_scale].var(axis = 0)
                    gene_stats['n'][gene] = feat_mat[gene][feat_to_scale].shape[0]
                gene_stats = pd.DataFrame(gene_stats)
            
                self.mean_ = gene_stats.apply(lambda x: x['mean'], axis = 1).mean(axis = 0)
                self.var_ = gene_stats.apply(lambda x: x['var'] * (x['n']-1)/x['n'] + 
                                             (x['mean'] - self.mean_)**2, axis = 1).mean(axis = 0)
                self.var_[self.var_ == 0] = 1
                self.var_[self.var_.isna()] = 1
                for gene in train_genes:
                    feat_mat[gene][feat_to_scale] = feat_mat[gene][feat_to_scale] / np.sqrt(self.var_)
        
        # Fit 100 models with 70% of the positions each
        if self.ensemble_ == True:
            for gene in train_genes:
                pos = [int(ind[1:-1]) for ind in feat_mat[gene].index.tolist()]
                self.fit_[gene] = []
                for i in range(100):
                    random.seed(i+1)
                    sample_pos = random.sample(np.unique(pos).tolist(), int(0.7 * len(np.unique(pos))))
                    sample_ind = [i for i,p in enumerate(pos) if p in sample_pos]
                    train_x = feat_mat[gene].iloc[sample_ind, :]
                    train_y = label[gene].iloc[sample_ind, :]
                    self.model_.fit(train_x, train_y)
                    self.fit_[gene].append(deepcopy(self.model_))
    
        # Fit one model
        else:
            for gene in train_genes:
                self.model_.fit(feat_mat[gene], label[gene])
                self.fit_[gene] = deepcopy(self.model_)
        return self
        
    def predict(self, feat_mat_all):
        feat_mat = deepcopy(feat_mat_all)
        pred = {}
        if self.scale_ == True:

            feat_to_scale = feat_mat.columns.difference(msa_feats)

            if not feat_to_scale.empty:
                feat_mat[feat_to_scale] = feat_mat[feat_to_scale] / np.sqrt(self.var_)      
                
        for gene in self.fit_:
            if self.ensemble_ == True:
                pred[gene] = pd.DataFrame([m.predict_proba(feat_mat)[:, 1]
                                           for m in self.fit_[gene]]).mean(axis = 0).values
            else:
                pred[gene] = self.fit_[gene].predict_proba(feat_mat)[:, 1]
        return(pd.DataFrame(pred).mean(axis = 1))

class CPTRegressor():
    '''
    Cross-Protein Transfer regressor
    '''
    def __init__(self, scale = True, l2 = 0, n_jobs = -1, ensemble = False, **kwargs):
        if l2 == 0:
            self.model_ = linear_model.LinearRegression(n_jobs = n_jobs, **kwargs)
        else:
            self.model_ = linear_model.LinearRegression(alpha = l2, n_jobs = n_jobs, **kwargs)
        self.fit_ = {}
        self.ensemble_ = ensemble
        self.scale_ = scale
        self.mean_ = 0
        self.var_ = 1
        
    def train(self, feat_mat_all, label, all_genes = None, feat_list = None):
        
        feat_mat = deepcopy(feat_mat_all)
        
        # Re-initialize fitted models
        self.fit_ = {}
        self.qnorm_fit_ = {}
        
        # By default use all genes and all features
        if all_genes == None:
            unittest.TestCase().assertCountEqual(feat_mat.keys(), label.keys(), 
                                                 'Gene lists of feature matrices and labels do not match.')
            all_genes = feat_mat.keys()
        
        if feat_list != None:
            for gene in all_genes:
                feat_mat[gene] = feat_mat[gene][feat_list]

        train_genes = all_genes
        
        if self.scale_ == True:
            
            feat_to_scale = feat_mat[all_genes[0]].columns.difference(msa_feats)
            
            if not feat_to_scale.empty:
                gene_stats = {'mean': {},
                          'var': {},
                          'n': {}}
                    
                for gene in train_genes:
                    gene_stats['mean'][gene] = feat_mat[gene][feat_to_scale].mean(axis = 0)
                    gene_stats['var'][gene] = feat_mat[gene][feat_to_scale].var(axis = 0)
                    gene_stats['n'][gene] = feat_mat[gene][feat_to_scale].shape[0]
                gene_stats = pd.DataFrame(gene_stats)
            
                self.mean_ = gene_stats.apply(lambda x: x['mean'], axis = 1).mean(axis = 0)
                self.var_ = gene_stats.apply(lambda x: x['var'] * (x['n']-1)/x['n'] + 
                                             (x['mean'] - self.mean_)**2, axis = 1).mean(axis = 0)
                self.var_[self.var_ == 0] = 1
                self.var_[self.var_.isna()] = 1
                for gene in train_genes:
                    feat_mat[gene][feat_to_scale] = feat_mat[gene][feat_to_scale] / np.sqrt(self.var_)
                                    
        # Fit 100 models with 70% of the positions each
        if self.ensemble_ == True:
            for gene in train_genes:
                pos = [int(ind[1:-1]) for ind in feat_mat[gene].index.tolist()]
                self.fit_[gene] = []
                for i in range(100):
                    random.seed(i+1)
                    sample_pos = random.sample(np.unique(pos).tolist(), int(0.7 * len(np.unique(pos))))
                    sample_ind = [i for i,p in enumerate(pos) if p in sample_pos]
                    train_x = feat_mat[gene].iloc[sample_ind, :]
                    train_y = label[gene].iloc[sample_ind, :]
                    self.model_.fit(train_x, train_y)
                    self.fit_[gene].append(deepcopy(self.model_))

        # Fit one model
        else:
            for gene in train_genes:
                self.model_.fit(feat_mat[gene], label[gene])
                self.fit_[gene] = deepcopy(self.model_)
        return self
        
    def predict(self, feat_mat_all):
        feat_mat = deepcopy(feat_mat_all)
        pred = {}
        if self.scale_ == True:
            
            feat_to_scale = feat_mat.columns.difference(msa_feats)

            if not feat_to_scale.empty:
                feat_mat[feat_to_scale] = feat_mat[feat_to_scale] / np.sqrt(self.var_)   
                
        for gene in self.fit_:
            if self.ensemble_ == True:
                pred[gene] = pd.DataFrame([m.predict(feat_mat).flatten()
                                           for m in self.fit_[gene]]).mean(axis = 0).values
            else:
                pred[gene] = self.fit_[gene].predict(feat_mat).flatten()
        return(pd.DataFrame(pred).mean(axis = 1))
