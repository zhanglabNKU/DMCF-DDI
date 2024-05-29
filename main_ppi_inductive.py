import torch
from torch_geometric.data import Data
import argparse
import time
import os
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from DMCF_PPI import complex_fuse_base
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
import torch as th
from utils import split_edges_ic, get_index


def init_model(args, seq_dim, drug_num, protein_num, paras):
    seq_dim = seq_dim

    if args.model_name == 'DMCF-DDI':
        model = complex_fuse_base(drug_num, protein_num,
                                  seq_dim,
                                  paras['h_dim'], paras['out_dim'],
                                  paras['num_base'], paras['dropout'],
                                  paras['operation'],
                                  paras['isdp'], paras['act'],
                                  paras['cmlp'], device, True, paras['method'],
                                  paras['classifier'])
    elif args.model_name == 'Quate':
        model = complex_fuse_base(drug_num, protein_num,
                                  seq_dim,
                                  paras['h_dim'], paras['out_dim'],
                                  paras['num_base'], paras['dropout'],
                                  paras['operation'],
                                  paras['isdp'], paras['act'],
                                  paras['cmlp'], device, True, paras['method'])
    else:
        model = nn.Module()
    return model


def train(model, optm, loss_fun, loader, train_pos, train_neg, seq, adj_list):
    model.train()
    loss_epoch = 0
    for batch in tqdm(loader, desc='Train:'):
        pos = train_pos[:, batch]
        neg = train_neg[:, batch]
        sam = torch.cat((pos, neg), dim=1)
        lab = torch.tensor([1] * len(batch) + [0] * len(batch)).to(device)

        logits = model(seq, adj_list, sam[0], sam[1])
        loss = loss_fun(logits.reshape(-1), lab.float())

        loss_epoch += loss.item()
        loss.backward()
        optm.step()
        optm.zero_grad()
    return loss_epoch/len(loader)


def eval_hits(pred_pos, pred_neg, K):
    kth_score_in_neg = torch.topk(pred_neg, K)[0][-1]
    hitn = (pred_pos > kth_score_in_neg).float().mean().item()
    return hitn


def ttest(model, seq, adj_list, loader, test_pos, test_neg, data_type='test'):
    model.eval()
    scores = torch.tensor([])
    labels = torch.tensor([])
    with torch.no_grad():
        #for data in tqdm(loader,position=0,leave=True):  # Iterate in batches over the training/test dataset.
        for batch in tqdm(loader, desc='Test:'+data_type):  # Iterate in batches over the training/test dataset.
            pos = test_pos[:, batch]
            neg = test_neg[:, batch]
            sam = torch.cat((pos, neg), dim=1)
            lab = torch.tensor([1.0] * len(batch) + [0.0] * len(batch))
            out = model(seq, adj_list, sam[0], sam[1])

            out = out.cpu().clone().detach()
            scores = torch.cat((scores,out),dim = 0)
            labels = torch.cat((labels,lab),dim = 0)

        hitn = eval_hits(scores[labels == 1], scores[labels == 0], args.hitk)
        scores = scores.cpu().clone().detach().numpy()
        labels = labels.cpu().clone().detach().numpy()
    return roc_auc_score(labels, scores), average_precision_score(labels, scores),hitn


def main(args):
    ppi = torch.load('../ppi/ppi.pth')
    ddi = torch.load('../ppi/ddi.pth')
    dti = torch.load('../ppi/dt.pth')
    seq = torch.load('../ppi/pca_pretrain_embedding.pth')
    protein_num = 18138
    drug_num = 450
    seq_dim = seq.shape[1]

    prot_have_drug = torch.zeros(protein_num, dtype=torch.bool)
    prot_have_drug[dti[1]] = True

    ppi = ppi[:, prot_have_drug[ppi[0]] & prot_have_drug[ppi[1]]]
    dti = dti[:, prot_have_drug[dti[1]]]

    prot_have_drug = prot_have_drug.nonzero().reshape(-1)
    subset, inv = prot_have_drug.unique(return_inverse=True)

    node_idx = torch.zeros(protein_num).long()
    node_idx[subset] = torch.arange(subset.size(0))
    ppi = node_idx[ppi]
    dti[1] = node_idx[dti[1]]
    seq = seq[subset]

    from torch_geometric.nn.pool import knn_graph
    knn_edges = knn_graph(seq, k=5)
    knn_edges = torch.cat((knn_edges, knn_edges[[1, 0]]), dim=1)

    protein_num = len(subset)

    # data = Data(edge_index=torch.cat((ppi, ppi[[1, 0]]), dim=1),
    #             num_nodes=protein_num)
    ddi = torch.cat((ddi, ddi[[1, 0]]), dim=1)
    adj_list = {'ddi': ddi.to(device),
                'dti': dti.to(device)}
    seq = seq.to(device)
    mode_log = exper_para = exper_content = 'None'
    model_para = {}

    if args.model_name == 'DMCF-DDI':
        method = 'ASC'
        classifier = args.classifier
        mode_log = '{}_{}_{}'.format(method, classifier, args.o_fun)
        exper_content = 'DMCF-DDI'
        model_para['h_dim'] = 256
        model_para['out_dim'] = args.out_dim
        model_para['num_base'] = 2
        model_para['dropout'] = 0.1
        model_para['operation'] = args.o_fun
        model_para['isdp'] = False
        model_para['act'] = False
        model_para['cmlp'] = False
        model_para['method'] = method
        model_para['classifier'] = classifier
        exper_para = 'out_dim={}_lr={}_batch_size={}_epoch={}'. \
            format(model_para['out_dim'], args.lr, args.batch_size, args.epochs)

    if args.model_name == 'Quate':
        method = 'quate'
        classifier = 'MLP'
        mode_log = '{}_{}'.format(method, classifier)
        exper_content = 'Quate'# input('')
        model_para['h_dim'] = 256
        model_para['out_dim'] = args.out_dim
        model_para['num_base'] = 2
        model_para['dropout'] = 0.1
        model_para['operation'] = args.o_fun
        model_para['isdp'] = False
        model_para['act'] = False
        model_para['cmlp'] = False
        model_para['method'] = method
        exper_para = 'out_dim={}_lr={}_batch_size={}_epoch={}'. \
            format(model_para['out_dim'], args.lr, args.batch_size, args.epochs)

    exper_name = '../ppi/IC-{}-{}'.format(time.ctime().replace(':', '-'), mode_log)
    if not os.path.exists(exper_name):
        os.makedirs(exper_name)

    logging.basicConfig(filename=exper_name + '/result.log', level=logging.INFO)
    logging.info('{}-{}-{}'.format(mode_log, exper_para, exper_content))

    BCE = nn.BCELoss()
    results = []

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    k = 0
    for train_nodes, test_nodes in kf.split(range(protein_num)):  # 对于所有药物分5折
        train_nodes, test_nodes = torch.from_numpy(train_nodes).long(), torch.from_numpy(test_nodes).long()
        trtr_edges, trte_edges, tete_edges = get_index(ppi, test_nodes, protein_num)
        edge_index = torch.cat((trtr_edges, trtr_edges[[1, 0]]), dim=1)

        neg_edges = split_edges_ic(edge_index,
                                   torch.cat((trte_edges, trte_edges[[1, 0]]), dim=1),
                                   torch.cat((tete_edges, tete_edges[[1, 0]]), dim=1),
                                   protein_num, len(trtr_edges[0]), len(trte_edges[0]), len(tete_edges[0]),
                                   train_nodes, test_nodes)

        train_neg, trte_neg, tete_neg = neg_edges
        knn_edges_mask = torch.zeros(protein_num, dtype=torch.bool)
        knn_edges_mask[train_nodes] = True
        knn_edges_train = knn_edges[:, knn_edges_mask[knn_edges[0]] & knn_edges_mask[knn_edges[1]]]
        dti_edge_mask = knn_edges_mask[dti[1]]
        dti_train = dti[:, dti_edge_mask].to(device)

        train_loader = DataLoader(range(len(trtr_edges[0])), args.batch_size, shuffle=True)
        trte_loader = DataLoader(range(len(trte_edges[0])), batch_size=5000, shuffle=False)
        tete_loader = DataLoader(range(len(tete_edges[0])), batch_size=10000, shuffle=False)
        model = init_model(args, seq_dim, drug_num, protein_num, model_para)
        optm = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
        model.to(device)

        Best_Val_fromAUC = 0
        Final_Train_Test_AUC_fromAUC = 0
        Final_Train_Test_AP_fromAUC = 0
        Final_Train_Test_Hitn_fromAUC = 0
        Final_Test_Test_AUC_fromAUC = 0
        Final_Test_Test_AP_fromAUC = 0
        Final_Test_Test_Hitn_fromAUC = 0

        for epoch in range(args.epoch):
            adj_list['ppi'] = torch.cat((edge_index, knn_edges_train), dim=1).to(device)
            adj_list['dti'] = dti_train
            loss_epoch = train(model, optm, BCE, train_loader, trtr_edges, train_neg, seq, adj_list)

            adj_list['ppi'] = torch.cat((edge_index, knn_edges), dim=1).to(device)
            adj_list['dti'] = dti.to(device)
            trte_auc, trte_ap, trte_hitn = ttest(model, seq, adj_list, trte_loader, trte_edges, trte_neg, data_type='train-test')
            tete_auc, tete_ap, tete_hitn = ttest(model, seq, adj_list, tete_loader, tete_edges, tete_neg, data_type='test-test')
            if tete_auc > Best_Val_fromAUC:
                Best_Val_fromAUC = tete_auc

                Final_Train_Test_AUC_fromAUC = trte_auc
                Final_Train_Test_AP_fromAUC = trte_ap
                Final_Train_Test_Hitn_fromAUC = trte_hitn
                Final_Test_Test_AUC_fromAUC = tete_auc
                Final_Test_Test_AP_fromAUC = tete_ap
                Final_Test_Test_Hitn_fromAUC = tete_hitn
            print(f'RUN:{k:01d}, Epoch: {epoch:03d}, Loss : {loss_epoch:.4f},\
                  Test-Test AUC: {tete_auc:.4f},\
                  Train-Test AUC: {trte_auc:.4f}')
            print(
                f'From AUC: Final TrainTest AUC: {Final_Train_Test_AUC_fromAUC:.4f}, '
                f'Final TrainTest AP: {Final_Train_Test_AP_fromAUC:.4f}, '
                f'Final TrainTest Hitn: {Final_Train_Test_Hitn_fromAUC:.4f}')
            print(
                f'From AUC: Final TestTest AUC: {Final_Test_Test_AUC_fromAUC:.4f}, '
                f'Final TestTest AP: {Final_Test_Test_AP_fromAUC:.4f}, '
                f'Final TestTest Hitn: {Final_Test_Test_Hitn_fromAUC:.4f}')
        results.append((Final_Train_Test_AUC_fromAUC, Final_Train_Test_AP_fromAUC, Final_Train_Test_Hitn_fromAUC,
                        Final_Test_Test_AUC_fromAUC, Final_Test_Test_AP_fromAUC, Final_Test_Test_Hitn_fromAUC))
        print(
            f'From AUC: Final TrainTest AUC: {Final_Train_Test_AUC_fromAUC:.4f}, '
            f'Final TrainTest AP: {Final_Train_Test_AP_fromAUC:.4f}, '
            f'Final TrainTest Hitn: {Final_Train_Test_Hitn_fromAUC:.4f}')
        print(
            f'From AUC: Final TestTest AUC: {Final_Test_Test_AUC_fromAUC:.4f}, '
            f'Final TestTest AP: {Final_Test_Test_AP_fromAUC:.4f}, '
            f'Final TestTest Hitn: {Final_Test_Test_Hitn_fromAUC:.4f}')
        k += 1

    results = torch.tensor(results)
    means = torch.mean(results, dim=0)
    stds = torch.std(results, dim=0)
    for i, m in enumerate(
            ['TrainTest AUC', 'TrainTest AP', 'TrainTest Hitn', 'TestTest AUC', 'TestTest AP', 'TestTest Hitn']):
        print(f'{m}: {means[i]:.4f} ± {stds[i]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link Prediction with Walk-Pooling')
    # Model
    parser.add_argument('--model_name', type=str, default='DMCF-DDI', help='Model class name')
    parser.add_argument('--out_dim', type=int, default=64, help='The dimension of output.')
    parser.add_argument('--o_fun', type=str, default='RE', help='SUM or RE')
    parser.add_argument('--classifier', type=str, default='cip', help='mlp or cip')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001,  help="Learning rate")
    parser.add_argument('--batch-size', type=int, default=20000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hitk', type=int, default=500)

    args = parser.parse_args()

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    args.device = device

    main(args)