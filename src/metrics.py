root_dir = '../..'
src_dir = 'src'

import os
import sys

sys.path.append(os.path.join(root_dir, src_dir))

from training import TrainingCorpus

def compute_fscore(p, r, beta):
    try:
        return (1 + beta**2) * p * r/(beta**2 * p + r)
    except:
        return 0
    
def compute_metrics(entity_id, gt_dict, terms, tidy_text=True):
    size = terms.shape[0]
    target_tokens = gt_dict[entity_id]
    if tidy_text:
        target_tokens = TrainingCorpus.tokenize(target_tokens.lower())
    target_tokens_len = len(target_tokens)
    
    tokens_count = 0
    tokens_found = []
    
    precision_list = []
    recall_list = []
    f1score_list = []
    f2score_list = []
    f05score_list = []
    
    for i in range(size):
        selected_terms = terms[i].split()
        
        for term in selected_terms:
            if term in target_tokens:
                if term not in tokens_found:
                    tokens_found.append(term)
                    tokens_count += 1
            else:
                tokens_count += 1
        
        precision = len(tokens_found)/tokens_count
        precision_list.append(precision)
        
        recall = len(tokens_found)/target_tokens_len
        recall_list.append(recall)

        f1score = compute_fscore(precision, recall, 1)
        f1score_list.append(f1score)
        
        f2score = compute_fscore(precision, recall, 2)
        f2score_list.append(f2score)
        
        f05score = compute_fscore(precision, recall, 0.5)
        f05score_list.append(f05score)
    
    metrics_dict = {}
    metrics_dict['entity'] = entity_id
    metrics_dict['k_recall'] = None
    metrics_dict['max_f1score_at_k'] = [0]
    metrics_dict['argmax_f1score'] = None
    metrics_dict['max_f2score_at_k'] = [0]
    metrics_dict['max_f05score_at_k'] = [0]
    metrics_dict['max_recall'] = None
    metrics_dict['max_precision'] = None
    metrics_dict['max_recall_at_k'] = [0]
    metrics_dict['max_precision_at_k'] = [0]
    
    # compute k_recall
    k_recall_threshold = 0.9
    not_found = True
    i = 0
    while i < size and not_found:
        if recall_list[i] > 0.9:
            not_found = False
            metrics_dict['k_recall'] = i + 1 # starts from 0
        i += 1
    
    # compute max_f1score_at_k
    max_f1score = -1
    for i in range(size):
        if f1score_list[i] > max_f1score:
            max_f1score = f1score_list[i]
        metrics_dict['max_f1score_at_k'].append(max_f1score)
    
    # compute argmax_f1score
    metrics_dict['argmax_f1score'] = max(list(range(size)), key=lambda i: f1score_list[i]) + 1 # starts from zero
    
    # compute max_f2score_at_k
    max_f2score = -1
    for i in range(size):
        if f2score_list[i] > max_f2score:
            max_f2score = f2score_list[i]
        metrics_dict['max_f2score_at_k'].append(max_f2score) 
    
    # compute max_f05score_at_k
    max_f05score = -1
    for i in range(size):
        if f05score_list[i] > max_f05score:
            max_f05score = f05score_list[i]
        metrics_dict['max_f05score_at_k'].append(max_f05score)
    
    # compute max_recall
    p_threshold = 0.9
    selected_idxs = [idx for idx, p in enumerate(precision_list) if p > p_threshold]
    if selected_idxs:
        metrics_dict['max_recall'] = max([recall_list[idx] for idx in selected_idxs])
    
    # compute max_precision
    r_threshold = 0.9
    selected_idxs = [idx for idx, r in enumerate(recall_list) if r > r_threshold]
    if selected_idxs:
        metrics_dict['max_precision'] = max([precision_list[idx] for idx in selected_idxs])
        
    # compute max_recall_at_k
    max_recall = -1
    for i in range(size):
        if recall_list[i] > max_recall:
            max_recall = recall_list[i]
        metrics_dict['max_recall_at_k'].append(max_recall)
    
    # compute max_precision_at_k
    max_precision = -1
    for i in range(size):
        if precision_list[i] > max_precision:
            max_precision = precision_list[i]
        metrics_dict['max_precision_at_k'].append(max_precision)
    
    return metrics_dict 