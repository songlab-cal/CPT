from .model import *
from copy import deepcopy
from tqdm import tqdm
import itertools
import random
import os
from sklearn.impute import KNNImputer
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
from scipy.special import logit


def load_data(all_genes, binarize, quantile = 0.4, read_label = True, cross_impute = False):
    '''
    Load DMS experiment data
    
    Parameters
    ----------
    all_genes : list of str
        A list of gene names to load
    binarize : bool, optional
        A boolean indicator of whether to binarize the labels or not
    quantile : float, optional
        A 0-0.5 float to indicate the threshold for binarization. Default value 0.4 will label samples below 
        lower 0.4 quantile as negative (0) and samples above upper 0.4 quantile as positive (1).
    read_label: bool, optional
        A boolean indicator of whether to load labels together or only the feature matrices
    cross_impute: bool, optional
        A boolean indicator of whether to load the cross-gene imputed version of the feature matrices.
    
    Returns
    -------
    feat_mat : dict
        A dictionary where keys are gene names and values are corresponding feature matrices
    label : dict
        A dictionary where keys are gene names and values are corresponding label vectors.
    '''
    feat_mat = {} # feature matrices / x
    label = {} # y

    # Get the absolute path of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Read feature matrices
    for gene in all_genes:
        if cross_impute == True:
            feat_mat[gene] = pd.read_csv(os.path.join(dir_path, '../data/feat_mat_xgi_' + gene + '.csv'))
        else:
            feat_mat[gene] = pd.read_csv(os.path.join(dir_path, '../data/feat_mat_' + gene + '.csv'))
        
        feat_mat[gene] = feat_mat[gene].set_index('mutant')
    
        if read_label == True:
            label[gene] = pd.read_csv(os.path.join(dir_path, '../data/label_' + gene + '.csv'))
            label[gene] = label[gene].set_index('mutant')
            if binarize == True:
                if quantile == 0.5:
                    label[gene]['screenscore'] = pd.qcut(label[gene]['screenscore'], [0, 0.5, 1], labels = [0, 1])
                else:
                    label[gene]['screenscore'] = pd.qcut(label[gene]['screenscore'], [0, quantile, 1-quantile, 1], labels = [0, -1, 1])
                feat_mat[gene] = feat_mat[gene].loc[label[gene]['screenscore'] != -1]
                label[gene] = label[gene].loc[label[gene]['screenscore'] != -1]
                print(f"Loaded {label[gene]['screenscore'].value_counts().loc[1]} positive variants and " \
                      f"{label[gene]['screenscore'].value_counts().loc[0]} negative variants in {gene}")
            else:   
                print(f"Loaded {label[gene].shape[0]} variants in {gene}")
    if read_label == False:
        label = None        
        
    return(feat_mat, label)

class WeightedImputer():
    def __init__(self, nn = 10):
        self.imputer_ = KNNImputer(n_neighbors = nn)
        self.feat_impute_ = None
        self.feat_weights_ = None
        
    def fit(self, feat_mat, feat_impute, feat_weights):
        '''
        Fit KNN imputer
        '''
        if feat_weights.isna().any():
            raise ValueError("Feature weights should not contain NA")
        feat_weights.loc[feat_impute] = 1 # The feature to be imputed has weight 1

        feats = feat_weights.index.to_list()
        x_impute = feat_mat[feats]
        mean_impute = x_impute.mean()
        std_impute = x_impute.std()
        std_impute[std_impute == 0] = 1
        x_impute = (x_impute - mean_impute) / std_impute * feat_weights # The features are weighted by their corraltion with feat_impute
        self.imputer_ = self.imputer_.fit(x_impute)
        self.feat_weights_ = feat_weights
        self.feat_impute_ = feat_impute
        return(self)
    
    def transform(self, feat_mat):
        '''
        Impute missing values with fitted imputer
        '''
        self.feat_weights_.loc[self.feat_impute_] = 1 # The feature to be imputed has weight 1

        feats = self.feat_weights_.index.to_list()
        x_impute = feat_mat[feats]
        mean_impute = x_impute.mean()
        std_impute = x_impute.std()
        mean_impute[mean_impute.isna()] = 0
        std_impute[std_impute.isna()] = 1
        std_impute[std_impute == 0] = 1
        x_impute = (x_impute - mean_impute) / std_impute * self.feat_weights_        
        x_impute = pd.DataFrame(self.imputer_.transform(x_impute),
                                columns = x_impute.columns,
                                index = x_impute.index)
        feat_mat[self.feat_impute_] = (x_impute / self.feat_weights_ * std_impute + mean_impute)[self.feat_impute_]
        return(feat_mat)
    
    def fit_transform(self, feat_mat, feat_impute, feat_weights):
        '''
        Fit KNN imputer and impute missing features

        Parameters
        ----------
        feat_mat : pandas DataFrame
            The dataframe to be imputed
        feat_impute : string
            The name of the feature to be imputed.
        feat_weights : pandas Series
            A series whose indices are the features used to fit the KNN imputer and the values are
            correlations with the feature to be imputed which are used as weights when calculating distance
        nn : int
            Number of nearest neighbors for imputation
        Returns
        -------
        feat_mat : pandas DataFrame
            The dataframe after imputation
        '''
        if feat_weights.isna().any():
            raise ValueError("Feature weights should not contain NA")
        feat_weights.loc[feat_impute] = 1 # The feature to be imputed has weight 1

        feats = feat_weights.index.to_list()
        x_impute = feat_mat[feats]
        mean_impute = x_impute.mean()
        std_impute = x_impute.std()
        std_impute[std_impute == 0] = 1
        x_impute = (x_impute - mean_impute) / std_impute * feat_weights # The features are weighted by their corraltion with feat_impute
        
        self.imputer_ = self.imputer_.fit(x_impute)
        self.feat_weights_ = feat_weights
        self.feat_impute_ = feat_impute
        
        x_impute = pd.DataFrame(self.imputer_.transform(x_impute),
                                columns = x_impute.columns,
                                index = x_impute.index)
        feat_mat[self.feat_impute_] = (x_impute / feat_weights * std_impute + mean_impute)[self.feat_impute_]
        
        return(feat_mat)


def cross_validate(feat_mat_all, label, all_genes = None, feat_list = None, model = 'classifier',
                   method = 'spearman', feat_performance = False, scale = True, l2 = 0, **kwargs):
    '''
    Cross validation split by gene. Test performance of on each gene with the model trained on all the other genes.
     
    Parameters
    ----------
    feat_mat_all : dict
        A dictionary whose keys are gene names and values are feature matrices; output of load_data()
    labels : dict
        A dictionary whose keys are gene names and values are labels by DMS experiments; output of load_data()
    all_genes : list of str
        A list of genes to do the test
    feat_list : list 
        A list of features to consider, by default use all the features
    feat_auroc : bool
        A boolean indicator of whether to calculate AUROC based on individual features
    scale: bool
        A boolean indicator of whether to normalize the data by weighted standardization
    l2: float
        L2 regularization parameter, by default 0, i.e. no regularization
        
    Returns
    -------
    auroc: pandas DataFrame
        A matrix of AUROC based on features or model predictions for each gene
    '''
    
    feat_mat = deepcopy(feat_mat_all)
    
    # By default use all genes and all features
    if all_genes == None:
        unittest.TestCase().assertCountEqual(feat_mat.keys(), label.keys(), 
                                             'Gene lists of feature matrices and labels do not match.')
        all_genes = feat_mat.keys()
    
    if feat_list == None:
        feat_list = feat_mat[all_genes[0]].columns.to_list()
    
    if model == 'classifier':
        cpt = CPTClassifier(scale = scale, l2 = l2, **kwargs)
        auroc = {}
        for test_gene in all_genes:
            test_genes = [test_gene]
            train_genes = list(set(all_genes) - set(test_genes))

            cpt.train(feat_mat, label, train_genes, feat_list)
            pred = cpt.predict(feat_mat[test_gene][feat_list])

            auroc[test_gene] = {}
            # Calculate AUROC by model prediction
            auroc[test_gene]['transfer_model'] = round(
                roc_auc_score(label[test_gene]['screenscore'], pred), 5)
        
            # Calculate AUORC by individual features
            if feat_performance == True:
                for feat in feat_mat_all[test_gene].columns.values:
                    auroc[test_gene][feat] = round(
                        roc_auc_score(label[test_gene]['screenscore'], feat_mat_all[test_gene][feat]), 5)
                    auroc[test_gene][feat] = max(auroc[test_gene][feat], 1 - auroc[test_gene][feat])
        auroc = pd.DataFrame(auroc)
        auroc['Avg'] = auroc.mean(axis = 1).round(5).abs()
        return(auroc.sort_values('Avg', ascending = False))
        
    elif model == 'regressor':
        cpt = CPTRegressor(scale = scale, l2 = l2, **kwargs)
        corr = {}
        for test_gene in all_genes:
            test_genes = [test_gene]
            train_genes = list(set(all_genes) - set(test_genes))

            cpt.train(feat_mat, label, train_genes, feat_list)
            pred = cpt.predict(feat_mat[test_gene][feat_list])
            
            corr[test_gene] = {}
            # Calculate correlation between model prediction and labels
            if method == 'spearman':
                corr[test_gene]['transfer_model'] = round(
                    stats.spearmanr(pred, label[test_gene])[0], 5)
            elif method == 'pearson':
                corr[test_gene]['transfer_model'] = round(
                    stats.pearsonr(pred.ravel(), label[test_gene].values.ravel())[0], 5)

            # Calculate correlation between individual features and labels
            if feat_performance == True:
                for feat in feat_mat_all[test_gene].columns.values:
                    if method == 'spearman':
                        corr[test_gene][feat] = round(
                            stats.spearmanr(feat_mat_all[test_gene][feat], label[test_gene])[0], 5)
                    elif method == 'pearson':
                        tmp = stats.pearsonr(feat_mat_all[test_gene][feat], label[test_gene])[0]
                        if pd.isna(tmp):
                            tmp = [0]
                        corr[test_gene][feat] = round(tmp[0], 5)
                
        corr = pd.DataFrame(corr)
        corr['Avg'] = corr.mean(axis = 1).round(5).abs()
        return(corr.sort_values('Avg', ascending = False))

    else:
        raise ValueError("'model' argument should error be 'classifier' or 'regressor'")
    
def powerset(l):
    '''
    Generate the power set of a given set/list
    '''
    pset = set()
    for n in range(len(l) + 1):
        for sset in itertools.combinations(l, n):
            pset.add(sset)
    return pset
 
def select_feat_by_category(feat_mat, label, all_genes, model = 'classifier', tol = 1e-3, verbose = True,
                            base_feat = ['esm_1v', 'evolutionary_index_ASM'], with_struct = True):
    '''
    Select features by category using DMS results
    '''
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
    
    feat_category = {}
    feat_category['100verte_msa'] = ['100verte_gap_freq', '100verte_wt_freq', '100verte_mut_freq']
    feat_category['30mammal_msa'] = ['30mammal_gap_freq', '30mammal_wt_freq', '30mammal_mut_freq']
    feat_category['mpnn'] = ['vanilla_protmpnn_norm', 'ca_protmpnn_norm', 'ca_only_protmpnn_norm']

    if with_struct == True:
        feat_category['struct'] = ['conditioned_wt_score', 'conditioned_mt_score']
    feat_selected = deepcopy(base_feat)
    
    # For complex features, search exhaustively
    for cat in feat_category:
        # Select at least one feature in each catagory
        pset_feats = powerset(feat_category[cat]) - {()}
        if verbose:
            print('Selecting features in the category', cat)
        best_metric = 0
        for feat_list in tqdm(pset_feats, disable = not verbose):
            metric = cross_validate(feat_mat, label, all_genes, feat_selected + list(feat_list),
                                    model = model).loc['transfer_model', 'Avg']
            if metric > best_metric:
                best_metric = metric
                best_feats = feat_list
        if verbose:
            print('Best features:', best_feats, 'Avg metric on DMS:', best_metric)
        feat_selected += list(best_feats)
    
    # For aa descriptor features, use feed-forward selection
    n_feat = 20
    aa_feats_selected = []
    metric_selected = []
    last_best_metric = 0
    for i in range(n_feat):
        best_feat = ''
        best_metric = 0
        for feat in aa_feats:
            if feat not in aa_feats_selected:
                feat_list = feat_selected + aa_feats_selected + [feat]
                metric = cross_validate(feat_mat, label, all_genes, feat_list,
                                        model = model).loc['transfer_model', 'Avg']
                if best_metric < metric:
                    best_metric = metric
                    best_feat = feat
        if (best_metric - last_best_metric) / last_best_metric < tol:
            break
        else:
            last_best_metric = best_metric
            aa_feats_selected.append(best_feat)
            metric_selected.append(best_metric)
        if verbose:
            print(i, aa_feats_selected, best_metric)
    feat_selected += aa_feats_selected
    print('All selected features:', feat_selected)
    return(feat_selected)

def random_feat_by_category(seed):
    '''
    Randomly select features
    '''
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
    
    feat_category = {}
    feat_category['verte_msa'] = ['100verte_gap_freq', '100verte_wt_freq', '100verte_mut_freq']
    feat_category['mammal_msa'] = ['30mammal_gap_freq', '30mammal_wt_freq', '30mammal_mut_freq']
    # feat_category['mpnn'] = ['vanilla_protmpnn', 'ca_protmpnn', 'ca_only_protmpnn']
    feat_category['mpnn'] = ['vanilla_protmpnn_norm', 'ca_protmpnn_norm', 'ca_only_protmpnn_norm']
    feat_category['struct'] = ['conditioned_wt_score', 'conditioned_mt_score']

    feat_selected = ['esm_1v', 'EVE_scores_ASM']
    
    random.seed(seed)
    for cat in feat_category:
        # For complex features, search exhaustively
        pset_feats = powerset(feat_category[cat]) - {()}
        best_feats = random.choice(list(pset_feats))
        feat_selected += list(best_feats)
    
    # For aa descriptor features, use feed-forward selection
    aa_feat1 = random.choice(aa_feats)
    aa_feat2 = random.choice(list(set(aa_feats) - set([aa_feat1])))
    aa_feat3 = random.choice(list(set(aa_feats) - set([aa_feat1, aa_feat2])))
    
    feat_selected += [aa_feat1, aa_feat2, aa_feat3]
    return(feat_selected)


def get_per_gene_auroc(label, pred, min_pos_sample = 4, min_neg_sample = 4, verbose = True):
    '''
    Calculate per gene AUROC
    '''
    df = label.reset_index()['mutant'].str.split(pat = '_', expand = True)
    df.columns = ['gene', 'mutant']
    df.index = label.index
    df['label'] = label.values
    if not (df.index == pred.index).all():
        raise AssertionError("Mutants in label and pred do not match.")
    df = df.merge(pred, left_index = True, right_index = True)
    grouping = df.groupby('gene').filter(lambda x: (x['label'] == 0.0).sum() >= min_neg_sample and 
                                      (x['label'] == 1.0).sum() >= min_pos_sample).groupby('gene')
    if verbose == True:
        print(f"{len(grouping)} genes used to calculate per gene AUROC out of {len(df.groupby('gene'))} genes")
        print('Calculating...')
    group_dict = {}
    for i in tqdm(grouping, disable = not verbose):
        (group_label, gdf) = i
        group_dict[group_label] = [gdf.shape[0],
                                   (gdf['label'] == 0.0).sum(),
                                   (gdf['label'] == 1.0).sum()] + \
                                  [roc_auc_score(gdf['label'], gdf[feat]) for feat in pred.columns]
    roc_by_gene = pd.DataFrame(group_dict, index = ['n', 'n_pos', 'n_neg'] + pred.columns.to_list())
    roc_by_gene.insert(loc = 0, column = 'Avg_per_gene_AUROC', value = roc_by_gene.mean(axis = 1))
    roc_by_gene = roc_by_gene.astype(float)
    return(roc_by_gene)
