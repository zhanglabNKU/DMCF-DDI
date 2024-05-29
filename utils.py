import torch
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
    return kg_path


def get_path_cs(NEGA_NUM, task, dise):
    kg_path = {}
    kg_path['name'] = f'case study for {dise}'
    kg_path['basepath'] = '../basedata/'

    kg_path['data_path'] = '../Case Study/{}/'.format(dise)

    kg_path['rs_train_dataset'] = '{}{} nega_num={}//rs org batch train dataset.npy'.format(kg_path['data_path'], task,
                                                                                            NEGA_NUM)
    kg_path['val_dataset'] = '{}{} nega_num={}//validation dataset.npy'.format(kg_path['data_path'], task, NEGA_NUM)
    kg_path['test_dataset'] = '{}{} nega_num={}//test dataset.npy'.format(kg_path['data_path'], task, NEGA_NUM)

    kg_path['fp'] = '{}fingerprint645_tensor.pth'.format(kg_path['basepath'])
    kg_path['norm_dt_mat'] = '{}norm_drug_target_adj.pth'.format(kg_path['basepath'])

    kg_path['ppi_indices'] = '{}sparse norm ppi adj indices.pth'.format(kg_path['basepath'])
    kg_path['ppi_values'] = '{}sparse norm ppi adj values.pth'.format(kg_path['basepath'])
    kg_path['sample'] = '{}sample.txt'.format(kg_path['basepath'])
    kg_path['train_pair_mark'] = '{}{} nega_num={}//train_pair_mark.npy'.format(kg_path['data_path'], task, NEGA_NUM)
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


def floor(x):
    return torch.div(x, 1, rounding_mode='trunc')

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def split_edges(data,args):
    set_random_seed(args.seed)
    row, col = data.edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    n_v= floor(args.val_ratio * row.size(0)).int() #number of validation positive edges
    n_t=floor(args.test_ratio * row.size(0)).int() #number of test positive edges
    #split positive edges
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v+n_t], col[n_v:n_v+n_t]
    data.test_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v+n_t:], col[n_v+n_t:]
    data.train_pos = torch.stack([r, c], dim=0)

    #sample negative edges
    if args.practical_neg_sample == False:
        # If practical_neg_sample == False, the sampled negative edges
        # in the training and validation set aware the test set

        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample all the negative edges and split into val, test, train negs
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.randperm(neg_row.size(0))[:row.size(0)]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v + n_t:], neg_col[n_v + n_t:]
        data.train_neg = torch.stack([row, col], dim=0)

    else:

        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample the test negative edges first
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.randperm(neg_row.size(0))[:n_t]
        neg_row, neg_col = neg_row[perm], neg_col[perm]
        data.test_neg = torch.stack([neg_row, neg_col], dim=0)

        # Sample the train and val negative edges with only knowing
        # the train positive edges
        row, col = data.train_pos
        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample the train and validation negative edges
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()

        n_tot = n_v + data.train_pos.size(1)
        perm = torch.randperm(neg_row.size(0))[:n_tot]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:], neg_col[n_v:]
        data.train_neg = torch.stack([row, col], dim=0)

    return data


def split_edges_ic(edge_index, trte_edges, tete_edges, num_nodes, trtr_num, trte_num, tete_num, tr_nodes, te_nodes):
    # Sample train negative edges
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]

    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0
    neg_adj_mask[tr_nodes.unsqueeze(-1), te_nodes.unsqueeze(0)] = 0
    neg_adj_mask[te_nodes.unsqueeze(-1), tr_nodes.unsqueeze(0)] = 0
    neg_adj_mask[te_nodes.unsqueeze(-1), te_nodes.unsqueeze(0)] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:row.size(0)]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    row, col = neg_row[:trtr_num], neg_col[:trtr_num]
    train_neg = torch.stack([row, col], dim=0)

    # Sample train-test negative edges
    row, col = trte_edges
    mask = row < col
    row, col = row[mask], col[mask]

    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[tr_nodes.unsqueeze(-1), tr_nodes.unsqueeze(0)] = 0
    neg_adj_mask[te_nodes.unsqueeze(-1), te_nodes.unsqueeze(0)] = 0
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:row.size(0)]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    row, col = neg_row[:trte_num], neg_col[:trte_num]
    trte_neg = torch.stack([row, col], dim=0)

    # Sample test-test negative edges
    row, col = tete_edges
    mask = row < col
    row, col = row[mask], col[mask]

    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[tr_nodes.unsqueeze(-1), tr_nodes.unsqueeze(0)] = 0
    neg_adj_mask[tr_nodes.unsqueeze(-1), te_nodes.unsqueeze(0)] = 0
    neg_adj_mask[te_nodes.unsqueeze(-1), tr_nodes.unsqueeze(0)] = 0
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:row.size(0)]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    row, col = neg_row[:tete_num], neg_col[:tete_num]
    tete_neg = torch.stack([row, col], dim=0)
    return train_neg, trte_neg, tete_neg


def get_index(edges, test_nodes, node_num):
    # 将所有药物分成五折，其中一折参与的所有药物组合都放在测试集里
    # 当然包含train-test, test-test，分开计算

    test_node_mask = torch.zeros(node_num, dtype=torch.bool)
    # test_nodes = torch.from_numpy(test_nodes).long()
    test_node_mask[test_nodes] = True   # 如果是测试节点就是True

    test_edge_mask = test_node_mask[edges[0]] | test_node_mask[edges[1]]  # edge里面只要有一个是test就是测试样本, tr-te, te-te

    task_mask = torch.ones(test_edge_mask.sum(), dtype=torch.long)  # 有多少测试样本,就做多长的mask
    test_edges = edges[:, test_edge_mask]

    task_mask[test_node_mask[test_edges[0]] & test_node_mask[test_edges[1]]] = 2

    trtr_edges = edges[:, ~test_edge_mask]
    trte_edges = test_edges[:, task_mask == 1]
    tete_edges = test_edges[:, task_mask == 2]
    print('Train data: ', trtr_edges.shape)
    print('Test data Train-Test: ', trte_edges.shape)
    print('Test data Test -Test: ', tete_edges.shape)
    return trtr_edges, trte_edges, tete_edges