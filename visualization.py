import torch as th
import numpy as np
from DMCF_DDI import complex_fuse_base
from utils import tc_datasets, get_path_cs, tc_input_mat
import argparse
from matplotlib import colors
import matplotlib.pyplot as plt
import math


def model_eval(ite, dev, exper_name, size,
               fp, dd_mat, dt_mat, ppi_mat,
               test_sample, test_SE,
               out_dim, ablation, drug_mark, o_fun, classifier):
    drug_node_num = dd_mat.shape[0]
    drug_node_id = th.arange(0, drug_node_num, 1).to(dev)

    protein_node_num = ppi_mat.shape[0]
    protein_node_id = th.arange(0, protein_node_num, 1).to(dev)

    target_num = dt_mat.shape[1]

    adj_list = [dd_mat, dt_mat, ppi_mat]

    model = complex_fuse_base(drug_node_num=drug_node_num,
                              protein_node_num=protein_node_num,
                              fp_dim=881,
                              h_dim=256,
                              out_dim=out_dim,
                              num_base=2,
                              target_num=target_num,
                              dropout=0.2,
                              num_rela=964,
                              ablation=ablation,
                              drug_mark=drug_mark,
                              o_fun=o_fun,
                              classifier=classifier)

    model_state_dic = th.load(exper_name + f'/model parameter {ite}.pth')
    model.load_state_dict(model_state_dic['featFusion'])
    model.to(dev)

    with th.no_grad():
        model.eval()
        if size > 1:
            test_output = []
            test_ite = int(test_sample.shape[0] / size)
            test_ite_list = []
            for t in range(size):
                if t == size - 1:
                    test_ite_list.append((test_ite * t, test_sample.shape[0]))
                else:
                    test_ite_list.append((test_ite * t, test_ite * (t + 1)))
            for j in test_ite_list:
                test_output_ = model(fp,
                                     drug_node_id,
                                     protein_node_id,
                                     adj_list,  # [d-d, d-t, p-p]
                                     idx1=test_sample[j[0]:j[1], 0].reshape(-1),
                                     idx2=test_sample[j[0]:j[1], 1].reshape(-1),
                                     idx3=test_SE[j[0]:j[1]])
                test_output.append(test_output_.reshape(1, -1))
            test_output = th.cat(test_output, dim=1).reshape(-1)
        else:
            test_output = model(fp,
                                drug_node_id,
                                protein_node_id,
                                adj_list,
                                idx1=test_sample[:, 0].reshape(-1),
                                idx2=test_sample[:, 1].reshape(-1),
                                idx3=test_SE)

    return test_output.cpu().detach(), model


def tc_mapk_(sample, output, label, side_effect):
    pos_sam = []
    pos_se = []
    neg_sam = []
    neg_se = []
    for se in range(964):
        if (side_effect == se).float().sum() > 0:
            # bool->float，如果全是FALSE，那么加和为0，只有大于0的情况下，有样本的情况下，才值得做这个标签的acc计算
            prob = output[side_effect == se]
            true = label[side_effect == se]
            tup = sample[side_effect == se]
            # true = th.nonzero(true).reshape(-1).tolist()  # 正样本的位置
            ind = th.sort(prob, descending=True)[1].tolist()
            if true[ind[0]] == 1:
                pos_sam.append(tup[ind[0]])
                pos_se.append(se)
                pos_sam.append(tup[ind[1]])
                pos_sam.append(tup[ind[2]])
            if true[ind[-1]] == 0:
                neg_sam.append(tup[ind[-1]])
                neg_sam.append(tup[ind[-2]])
                neg_sam.append(tup[ind[-3]])
                neg_se.append(se)

    return pos_sam, pos_se, neg_sam, neg_se


def get_modulus_phase(R, I):
    z_mod = np.linalg.norm(np.concatenate((R.reshape(-1, 1, out_dim), I.reshape(-1, 1, out_dim)), axis=1), axis=1)
    pha = np.arctan(I / R)
    pha += math.pi * ((R < 0) & (I >= 0)).astype('float32')
    pha -= math.pi * ((R < 0) & (I < 0)).astype('float32')
    return z_mod, pha

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dim', type=int, default=64, help='The dimension of output.')
    parser.add_argument('--dise', type=str, default='Pneumonia')
    parser.add_argument('--ite', type=int, default=0, help='The iterate number')
    parser.add_argument('--size', type=int, default=4, help='The batch size for validation set')
    parser.add_argument('--NEGA_NUM', type=str, default='1to1',
                        help='Num. of positive samples: Num. of negative samples')
    parser.add_argument('--model_path', type=str, default='./',
                        help='The path optm model exist.')
    parser.add_argument('--method', type=str, default='ASC', help='ASC, SC or quate')
    parser.add_argument('--cmlp', action='store_trues',  help='Whether use complex-valued linear transformation (Task A Yes, Task B No)')
    parser.add_argument('--o_fun', type=str, default='SUM', help='SUM or RE')
    parser.add_argument('--classifier', type=str, default='cip', help='mlp or cip')
    parser.add_argument('--date', type=str, default='23.xx.xx', help='Experiment date')
    args = parser.parse_args()

    dev = th.device('cuda' if th.cuda.is_available() else 'cpu')

    out_dim = args.out_dim  ##
    dise = args.dise
    ite = args.ite
    size = args.size

    task = 'TaskB'
    NEGA_NUM = args.NEGA_NUM
    model_path = args.model_path

    ablation = {'method': args.method,
                'cmlp': args.cmlp}
    o_fun = args.o_fun
    classifier = args.classifier

    kg_path = get_path_cs(NEGA_NUM, task, args.dise)

    fp, dt_mat, ppi_mat, val_dataset, test_dataset = tc_datasets(kg_path)
    fp, dt_mat, ppi_mat = fp.to(dev), dt_mat.to(dev), ppi_mat.to(dev)
    drug_mark = None

    sample = []
    file_sample = open(kg_path['sample']).readlines()
    for line in file_sample:
        sample.append((int(line.split(" ")[0]), int(line.split(" ")[1])))
    sample = np.array(sample)
    train_pair_mark = np.load(kg_path['train_pair_mark'], allow_pickle=True)
    dd_mat = tc_input_mat(sample, trpair_mark=train_pair_mark)
    dd_mat = dd_mat.cuda()

    test_dataset = np.load(kg_path['test_dataset'], allow_pickle=True).item()
    test_sample = test_dataset[ite]['sample'].to(dev)
    test_SE = th.tensor(test_dataset[ite]['side effect']).to(dev)
    test_label = th.tensor(test_dataset[ite]['label'])
    test_output, optm_model = model_eval(ite, dev, model_path, size,
                                         fp, dd_mat, dt_mat, ppi_mat,
                                         test_sample, test_SE,
                                         out_dim, ablation, drug_mark, o_fun, classifier)

    pos_sam, pos_se, neg_sam, neg_se = tc_mapk_(test_sample, test_output, test_label, test_SE)

    drug_node_num = dd_mat.shape[0]
    drug_node_id = th.arange(0, drug_node_num, 1).cuda()

    protein_node_num = ppi_mat.shape[0]
    protein_node_id = th.arange(0, protein_node_num, 1).cuda()

    target_num = dt_mat.shape[1]

    adj_list = [dd_mat, dt_mat, ppi_mat]

    pos_sam = th.vstack(pos_sam)
    neg_sam = th.vstack(neg_sam)
    pos_se = th.tensor(pos_se)
    neg_se = th.tensor(neg_se)
    pos_R, pos_I = optm_model.get_R_Ima(fp, drug_node_id, protein_node_id, adj_list, pos_sam[:, 0], pos_sam[:, 1])
    neg_R, neg_I = optm_model.get_R_Ima(fp, drug_node_id, protein_node_id, adj_list, neg_sam[:, 0], neg_sam[:, 1])

    pos_R, pos_I = pos_R.cpu().detach().numpy(), pos_I.cpu().detach().numpy()
    neg_R, neg_I = neg_R.cpu().detach().numpy(), neg_I.cpu().detach().numpy()

    pos_mod, pos_pha = get_modulus_phase(pos_R, pos_I)
    neg_mod, neg_pha = get_modulus_phase(neg_R, neg_I)

    plt.figure(dpi=300, figsize=(8, 6))
    plt.hist(pos_pha.reshape(-1), bins=60, density=True, label='positive drug pairs', alpha=0.7, color="#15b01a")
    plt.hist(neg_pha.reshape(-1), bins=60, density=True, label='negative drug pairs', alpha=0.7, color="#a55af4")
    plt.xticks([-1 * math.pi, -1 * math.pi / 2, 0, math.pi / 2, math.pi],
               ['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlabel('phase', fontsize=20)
    plt.savefig(f'../image/DMCF-{out_dim} phase hist.png', bbox_inches='tight', pad_inches=0.02, transparent=True)
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
    plt.savefig(f'../image/DMCF-{out_dim} positive hist2d hist.png', bbox_inches='tight', pad_inches=0.02,
                transparent=True)
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
    plt.savefig(f'../image/DMCF-{out_dim} negative hist2d hist.png', bbox_inches='tight', pad_inches=0.02,
                transparent=True)
    plt.show()