import torch
import numpy as np
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
import matplotlib.pyplot as plt
import math
import torch as th
from utils import split_edges
from matplotlib import colors


def init_model(args, drug_num, protein_num, seq_dim, paras):
    seq_dim = seq_dim

    if args.model_name == 'DMCF-DDI':
        model = complex_fuse_base(drug_num, protein_num,
                                  seq_dim,
                                  paras['h_dim'], paras['out_dim'],
                                  paras['num_base'], paras['dropout'],
                                  paras['operation'],
                                  paras['isdp'], paras['act'],
                                  paras['cmlp'], device, False, paras['method'],
                                  paras['classifier'])
    elif args.model_name == 'Quate':
        model = complex_fuse_base(drug_num, protein_num,
                                  seq_dim,
                                  paras['h_dim'], paras['out_dim'],
                                  paras['num_base'], paras['dropout'],
                                  paras['operation'],
                                  paras['isdp'], paras['act'],
                                  paras['cmlp'], device, False, paras['method'])
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


def get_output(model, seq, adj_list, loader, test_pos, data_type='test'):
    model.eval()
    scores = torch.tensor([])
    with torch.no_grad():
        #for data in tqdm(loader,position=0,leave=True):  # Iterate in batches over the training/test dataset.
        for batch in tqdm(loader, desc='Test:'+data_type):  # Iterate in batches over the training/test dataset.
            pos = test_pos[:, batch]
            out = model(seq, adj_list, pos[0], pos[1])
            out = out.cpu().clone().detach()
            scores = torch.cat((scores, out), dim=0)
        scores = scores.cpu().clone().detach()
    return scores


def get_modulus_phase(R, I, out_dim):
    z_mod = np.linalg.norm(np.concatenate((R.reshape(-1, 1, out_dim), I.reshape(-1, 1, out_dim)), axis=1), axis=1)
    pha = np.arctan(I/R)
    pha += math.pi*((R < 0) & (I >=0)).astype('float32')
    pha -= math.pi*((R < 0) & (I < 0)).astype('float32')
    return z_mod, pha


def main(args):
    ppi = torch.load('../ppi/ppi.pth')
    ddi = torch.load('../ppi/ddi.pth')
    dti = torch.load('../ppi/dt.pth')
    protein_num = 18138
    drug_num = 450

    # seq = torch.load('../ppi/pretrain_embedding.pth')
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components = 300)
    # pca.fit(seq)
    # seq = pca.transform(seq)
    # from sklearn.preprocessing import minmax_scale
    # seq = minmax_scale(seq, axis=0)
    # torch.save(torch.from_numpy(seq).float(), '../ppi/pca_pretrain_embedding.pth')

    seq = torch.load('../ppi/pca_pretrain_embedding.pth')
    seq = seq.to(device)
    seq_dim = seq.shape[1]

    data = Data(edge_index=torch.cat((ppi, ppi[[1, 0]]), dim=1),
                num_nodes=protein_num)
    ddi = torch.cat((ddi, ddi[[1, 0]]), dim=1)
    adj_list = {'ddi': ddi.to(device),
                'dti': dti.to(device)}

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

    exper_name = '../ppi/{}-{}'.format(time.ctime().replace(':', '-'), mode_log)
    if not os.path.exists(exper_name):
        os.makedirs(exper_name)

    logging.basicConfig(filename=exper_name + '/result.log', level=logging.INFO)
    logging.info('{}-{}-{}'.format(mode_log, exper_para, exper_content))

    BCE = nn.BCELoss()
    results = []
    for seed in range(5):
        args.seed = seed
        data = split_edges(data, args)
        ppi = torch.cat((data.train_pos, data.train_pos[[1, 0]]), dim=1)
        adj_list['ppi'] = ppi.to(device)
        train_loader = DataLoader(range(len(data.train_pos[0])), args.batch_size, shuffle=True)
        valid_loader = DataLoader(range(len(data.val_pos[0])), batch_size=5000, shuffle=False)
        test_loader = DataLoader(range(len(data.test_pos[0])), batch_size=10000, shuffle=False)
        model = init_model(args, drug_num, protein_num, seq_dim, model_para)
        optm = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
        model.to(device)

        Best_Val_fromAUC = 0
        Final_Test_AUC_fromAUC = 0
        Final_Test_AP_fromAUC = 0
        Final_Test_Hitn_fromAUC = 0

        for epoch in range(args.epoch):
            loss_epoch = train(model, optm, BCE, train_loader, data.train_pos, data.train_neg, seq, adj_list)
            val_auc, val_ap, val_hitn = ttest(model, seq, adj_list, valid_loader, data.val_pos, data.val_neg, data_type='valid')
            test_auc, test_ap, test_hitn = ttest(model, seq, adj_list, test_loader, data.test_pos, data.test_neg, data_type='test')
            if val_auc > Best_Val_fromAUC:
                Best_Val_fromAUC = val_auc
                Final_Test_AUC_fromAUC = test_auc
                Final_Test_AP_fromAUC = test_ap
                Final_Test_Hitn_fromAUC = test_hitn
                optm_state = model.state_dict()
            print(f'RUN:{seed:01d}, Epoch: {epoch:03d}, Loss : {loss_epoch:.4f},\
                  Val AUC: {val_auc:.4f},\
                  Test AUC: {test_auc:.4f}, Picked AUC:{Final_Test_AUC_fromAUC:.4f}\
                  Test Hitn: {test_hitn:.4f}, Picked Hitn:{Final_Test_Hitn_fromAUC:.4f}')
            print(
                f'From AUC: Final Test AUC: {Final_Test_AUC_fromAUC:.4f}, '
                f'Final Test AP: {Final_Test_AP_fromAUC:.4f}, '
                f'Final Test Hitn: {Final_Test_Hitn_fromAUC:.4f}')

        results.append((Final_Test_AUC_fromAUC, Final_Test_AP_fromAUC, Final_Test_Hitn_fromAUC))

        torch.save(optm_state, exper_name + '/optm_state{}.pth'.format(seed))

        if args.model_name == 'DMCF-DDI' and False:
            model.load_state_dict(optm_state)
            pos_score = get_output(model, seq, adj_list, test_loader, data.test_pos, data_type='test')
            neg_score = get_output(model, seq, adj_list, test_loader, data.test_neg, data_type='test')

            _, pos_idx = torch.sort(-1 * pos_score)
            _, neg_idx = torch.sort(neg_score)

            pos_sam = data.test_pos[:, pos_idx[:80]]
            neg_sam = data.test_neg[:, neg_idx[:80]]

            pos_R, pos_I = model.get_R_Ima(seq, adj_list, pos_sam[0], pos_sam[1])
            neg_R, neg_I = model.get_R_Ima(seq, adj_list, neg_sam[0], neg_sam[1])

            pos_R, pos_I = pos_R.cpu().detach().numpy(), pos_I.cpu().detach().numpy()
            neg_R, neg_I = neg_R.cpu().detach().numpy(), neg_I.cpu().detach().numpy()

            pos_mod, pos_pha = get_modulus_phase(pos_R, pos_I, args.out_dim)
            neg_mod, neg_pha = get_modulus_phase(neg_R, neg_I, args.out_dim)
            plt.figure(dpi=300, figsize=(8, 6))
            plt.hist(pos_pha.reshape(-1), bins=60, density=True, label='positive drug pairs', alpha=0.7,
                     color="#15b01a")
            plt.hist(neg_pha.reshape(-1), bins=60, density=True, label='negative drug pairs', alpha=0.7,
                     color="#a55af4")
            plt.xticks([-1 * math.pi, -1 * math.pi / 2, 0, math.pi / 2, math.pi],
                       ['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.xlabel('phase', fontsize=20)
            plt.savefig('../ppi/phase hist.png', bbox_inches='tight', pad_inches=0.02, transparent=True)
            plt.show()

            plt.figure(dpi=300, figsize=(8, 6))
            plt.hist2d(pos_pha.reshape(-1), pos_mod.reshape(-1), bins=100, cmap="Spectral_r", norm=colors.LogNorm())
            plt.xticks([-1 * math.pi, -1 * math.pi / 2, 0, math.pi / 2, math.pi],
                       ['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
            plt.yticks(fontsize=20)
            plt.ylabel('modulus', fontsize=20)
            plt.xticks(fontsize=20)
            plt.xlabel('phase', fontsize=20)
            plt.colorbar(format='%.1f', ticks=range(11))
            plt.savefig('../ppi/positive hist2d hist.png', bbox_inches='tight', pad_inches=0.02, transparent=True)
            plt.show()

            plt.figure(dpi=300, figsize=(8, 6))
            plt.hist2d(neg_pha.reshape(-1), neg_mod.reshape(-1), bins=100, cmap="Spectral_r", norm=colors.LogNorm())
            plt.xticks([-1 * math.pi, -1 * math.pi / 2, 0, math.pi / 2, math.pi],
                       ['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
            plt.yticks(fontsize=20)
            plt.ylabel('modulus', fontsize=20)
            plt.xticks(fontsize=20)
            plt.xlabel('phase', fontsize=20)
            plt.colorbar(format='%.1f', ticks=range(11))
            plt.savefig('../ppi/negative hist2d hist.png', bbox_inches='tight', pad_inches=0.02, transparent=True)
            plt.show()
        print(
            f'From AUC: Final Test AUC: {Final_Test_AUC_fromAUC:.4f}, Final Test AP: {Final_Test_AP_fromAUC:.4f}, Final Test Hitn: {Final_Test_Hitn_fromAUC:.4f}')

    results = torch.tensor(results)
    means = torch.mean(results, dim=0)
    stds = torch.std(results, dim=0)
    for i, m in enumerate(['AUC', 'AP', 'Hitn']):
        print(f'{m}: {means[i]:.4f} ± {stds[i]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link Prediction with Walk-Pooling')
    # Dataset
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='0.1 ratio of test links')
    parser.add_argument('--val-ratio', type=float, default=0.05,
                        help='ratio of validation links. If using the splitted data from SEAL,\
                         it is the ratio on the observed links, othewise, it is the ratio on the whole links.')
    parser.add_argument('--practical-neg-sample', type=bool, default=False,
                        help='only see the train positive edges when sampling negative')
    # Model
    parser.add_argument('--model_name', type=str, default='DMCF-DDI',
                        help='Model class name')
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