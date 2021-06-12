import numpy as np
from sklearn.model_selection import StratifiedKFold


def df_to_dataset(X, Y, nfolds, shuffle=True):
    bags_nm = np.asarray(Y['pid'], dtype=str)
    bags_label = np.asarray(Y['mor'], dtype='uint8')
    ins_fea = np.asarray(X)

    ins_idx_of_input = {}
    for id, bag_nm in enumerate(bags_nm):
        if bag_nm in ins_idx_of_input:
            ins_idx_of_input[bag_nm].append(id)
        else:
            ins_idx_of_input[bag_nm] = [id]
    bags_fea = []
    for bag_nm, ins_idxs in ins_idx_of_input.items():
        bag_fea = ([], [], [])
        for ins_idx in ins_idxs:
            bag_fea[0].append(ins_fea[ins_idx])
            bag_fea[1].append(bags_label[ins_idx])
            bag_fea[2].append(bag_nm)
        bags_fea.append(bag_fea)

    skf = StratifiedKFold(n_splits=nfolds, shuffle=shuffle, random_state=1234)
    datasets = []
    bags_list = [item[1] for item in bags_fea]
    bag_label = [item[0] for item in bags_list]
    for train_idx, test_idx in skf.split(bags_fea, bag_label):
        dataset = {}
        dataset['train'] = [bags_fea[ibag] for ibag in train_idx]
        dataset['test'] = [bags_fea[ibag] for ibag in test_idx]
        datasets.append(dataset)

    return datasets
