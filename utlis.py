import torch as th
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import tqdm

def datasets(kg_path):
    norm_dd_mat = th.load(kg_path['norm_dd_mat']).float()
    fp = th.load(kg_path['fp']).float()
    norm_dt_mat = th.load(kg_path['norm_dt_mat'])

    indices = th.load(kg_path['ppi_indices']).int()
    values = th.load(kg_path['ppi_values'])
    sparse_ppi = th.sparse_coo_tensor(indices, values).coalesce()

    val_dataset = np.load(kg_path['val_dataset'], allow_pickle=True).item()
    test_dataset = np.load(kg_path['test_dataset'], allow_pickle=True).item()

    return fp, norm_dd_mat, norm_dt_mat, sparse_ppi, val_dataset, test_dataset

def tc_datasets(kg_path):
    fp = th.load(kg_path['fp']).float()
    norm_dt_mat = th.load(kg_path['norm_dt_mat'])

    indices = th.load(kg_path['ppi_indices']).int()
    values = th.load(kg_path['ppi_values'])
    sparse_ppi = th.sparse_coo_tensor(indices, values).coalesce()

    val_dataset = np.load(kg_path['val_dataset'], allow_pickle=True).item()
    test_dataset = np.load(kg_path['test_dataset'], allow_pickle=True).item()

    return fp, norm_dt_mat, sparse_ppi, val_dataset, test_dataset

def arange_batch(kg_path, batch_size):
    rs_train_dataset = np.load(kg_path['rs_train_dataset'], allow_pickle=True).item()
    x_data = th.tensor(range(964))
    y_data = th.ones(964)
    dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=False)
    batchTrain_dataset = {}
    for i, data in enumerate(tqdm(train_loader)):
        # print('第%d组batch' % i)
        batchTrain_dataset[i] = {}
        batchTrain_dataset[i]['sample'] = []
        batchTrain_dataset[i]['side effect'] = []
        batchTrain_dataset[i]['label'] = []
        c_idx, _ = data
        c_idx = c_idx.tolist()
        for c in c_idx:
            batchTrain_dataset[i]['sample'].append(rs_train_dataset[c]['sample'])
            batchTrain_dataset[i]['side effect'] += rs_train_dataset[c]['side effect']
            batchTrain_dataset[i]['label'] += rs_train_dataset[c]['label']
        batchTrain_dataset[i]['sample'] = th.cat(batchTrain_dataset[i]['sample'], dim=0)
    return batchTrain_dataset

def tc_arange_batch(t, rs_train_dataset, batch_size):
    x_data = th.tensor(range(964))
    y_data = th.ones(964)
    dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=False)
    batchTrain_dataset = {}
    for i, data in enumerate(tqdm(train_loader)):
        # print('第%d组batch' % i)
        batchTrain_dataset[i] = {}
        batchTrain_dataset[i]['sample'] = []
        batchTrain_dataset[i]['side effect'] = []
        batchTrain_dataset[i]['label'] = []
        c_idx, _ = data
        c_idx = c_idx.tolist()
        for c in c_idx:
            batchTrain_dataset[i]['sample'].append(rs_train_dataset[t][c]['sample'])
            batchTrain_dataset[i]['side effect'] += rs_train_dataset[t][c]['side effect']
            batchTrain_dataset[i]['label'] += rs_train_dataset[t][c]['label']
        batchTrain_dataset[i]['sample'] = th.cat(batchTrain_dataset[i]['sample'], dim=0)
    return batchTrain_dataset

def get_path(NEGA_NUM, data_type, ratio = {'va': 0.1, 'te': 0.1, 'tr': 0.8}):
    kg_path = {}
    kg_path['name'] = data_type
    kg_path['basepath'] = '../basedata/'
    kg_path['data_path'] = '../{}/'.format(data_type)

    # ratio = {'va':0.2, 'te':0.3, 'tr':0.5}
    kg_path['rs_train_dataset'] = '{}{}-{} nega_num={}//train dataset.npy'.format(kg_path['data_path'],
                                                                                               ratio['va'],
                                                                                               ratio['te'],
                                                                                               NEGA_NUM)
    kg_path['val_dataset'] = '{}{}-{} nega_num={}//validation dataset.npy'.format(kg_path['data_path'],
                                                                                  ratio['va'],
                                                                                  ratio['te'],
                                                                                  NEGA_NUM)
    kg_path['test_dataset'] = '{}{}-{} nega_num={}//test dataset.npy'.format(kg_path['data_path'],
                                                                             ratio['va'],
                                                                             ratio['te'],
                                                                             NEGA_NUM)

    kg_path['norm_dd_mat'] = '{}{}-{} nega_num={}//norm adj_mat.pth'.format(kg_path['basepath'],
                                                                            ratio['va'],
                                                                            ratio['te'],
                                                                            NEGA_NUM)

    kg_path['fp'] = '{}fingerprint645_tensor.pth'.format(kg_path['basepath'])
    kg_path['norm_dt_mat'] = '{}norm_drug_target_adj.pth'.format(kg_path['basepath'])

    kg_path['ppi_indices'] = '{}sparse norm ppi adj indices.pth'.format(kg_path['basepath'])
    kg_path['ppi_values'] = '{}sparse norm ppi adj values.pth'.format(kg_path['basepath'])
    kg_path['dp_indices'] = '{}sparse norm d-p adj indices.pth'.format(kg_path['basepath'])
    kg_path['dp_values'] = '{}sparse norm d-p adj values.pth'.format(kg_path['basepath'])
    return kg_path

def get_path_tc(NEGA_NUM, data_type, task):
    kg_path = {}
    kg_path['name'] = data_type
    kg_path['basepath'] = '../basedata/'
    kg_path['data_path'] = '../{}/'.format(data_type)

    kg_path['train_dataset'] = '{}{} nega_num={}//rs org batch train dataset.npy'.format(kg_path['data_path'], task,
                                                                                            NEGA_NUM)
    kg_path['val_dataset'] = '{}{} nega_num={}//validation dataset.npy'.format(kg_path['data_path'], task, NEGA_NUM)
    kg_path['test_dataset'] = '{}{} nega_num={}//test dataset.npy'.format(kg_path['data_path'], task, NEGA_NUM)

    kg_path['fp'] = '{}fingerprint645_tensor.pth'.format(kg_path['basepath'])
    kg_path['norm_dt_mat'] = '{}norm_drug_target_adj.pth'.format(kg_path['basepath'])

    kg_path['ppi_indices'] = '{}sparse norm ppi adj indices.pth'.format(kg_path['basepath'])
    kg_path['ppi_values'] = '{}sparse norm ppi adj values.pth'.format(kg_path['basepath'])
    kg_path['dp_indices'] = '{}sparse norm d-p adj indices.pth'.format(kg_path['basepath'])
    kg_path['dp_values'] = '{}sparse norm d-p adj values.pth'.format(kg_path['basepath'])
    kg_path['sample'] = '{}sample.txt'.format(kg_path['basepath'])
    kg_path['train_pair_mark'] = '{}{} nega_num={}//train_pair_mark.npy'.format(kg_path['data_path'], task, NEGA_NUM)
    kg_path['node_mark'] = '{}{} nega_num={}//node_mark.npy'.format(kg_path['data_path'], task, NEGA_NUM)
    return kg_path

def tc_input_mat(sample, trpair_mark, drug_num=645):
    k_sam = sample[trpair_mark]
    k_sam = th.from_numpy(k_sam).long()
    dd_mat = th.zeros((drug_num, drug_num))
    dd_mat[k_sam[:, 0], k_sam[:, 1]] = 1.0
    dd_mat[k_sam[:, 1], k_sam[:, 0]] = 1.0
    d_deg = th.sum(dd_mat, dim=0)
    d_deg = th.pow(d_deg.float(), -0.5)
    d_deg[th.isinf(d_deg)] = 0.0
    dd_mat = th.mm(th.mm(th.diag(d_deg), dd_mat.float()), th.diag(d_deg))
    return dd_mat