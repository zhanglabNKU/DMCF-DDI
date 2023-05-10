import torch as th
import numpy as np
import time
import torch.nn as nn
from DMCF_DDI import complex_fuse_base
import os
from utlis import datasets, arange_batch, get_path
from metric import class_auc, class_aupr, mapk_, class_acc
from tqdm import tqdm
import logging
import argparse

def epoch_train(ite, dev, exper_name, size,
                fp, dd_mat, dt_mat, ppi_mat, batchTrain_dataset,
                val_sample, val_SE, val_label,
                out_dim, lr, batch_size, ablation, drug_mark, o_fun, classifier):
    # ---------------------------------------------------#
    drug_node_num = dd_mat.shape[0]
    drug_node_id = th.arange(0, drug_node_num, 1).to(dev)

    protein_node_num = ppi_mat.shape[0]
    protein_node_id = th.arange(0, protein_node_num, 1).to(dev)

    target_num = dt_mat.shape[1]

    BCE = nn.BCELoss()
    adj_list = [dd_mat, dt_mat, ppi_mat]

    # ---------------------------------------------------#
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

    optm = th.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    model.to(dev)

    max_val_acc = 0
    optm_epoch = 0
    sub_acc = 0
    # ---------------------------------------------------#
    batch_num = len(batchTrain_dataset.keys())
    batch_idx = list(range(batch_num))
    # batch_idx = list(range(25))
    TS_time = time.time()
    for epoch in range(50):
        es_time = time.time()
        print('##################  Train  #####################')
        for i in tqdm(batch_idx):
            train_sample = batchTrain_dataset[i]['sample'].to(dev)
            train_SE = th.tensor(batchTrain_dataset[i]['side effect']).to(dev)
            train_label = th.tensor(batchTrain_dataset[i]['label']).float().to(dev)

            start_time = time.time()
            model.train()
            train_output = model(fp, drug_node_id,
                                 protein_node_id,
                                 adj_list,  # [d-d, d-t, p-p, d-p]
                                 idx1=train_sample[:, 0].reshape(-1),
                                 idx2=train_sample[:, 1].reshape(-1),
                                 idx3=train_SE)

            loss_ce = BCE(train_output.reshape(-1), train_label)
            loss = loss_ce

            if th.isnan(loss) or loss > 20:
                result_dic = {}
                return result_dic

            loss.backward()
            optm.step()
            optm.zero_grad()

            end_time = time.time()
        print('##################  Validation  #####################')
        with th.no_grad():
            model.eval()
            val_output = []
            val_ite = int(val_sample.shape[0] / size)
            val_ite_list = []
            for t in range(size):
                if t == size - 1:
                    val_ite_list.append((val_ite * t, val_sample.shape[0]))
                else:
                    val_ite_list.append((val_ite * t, val_ite * (t + 1)))

            for j in tqdm(val_ite_list):
                val_output_ = model(fp, drug_node_id,
                                    protein_node_id,
                                    adj_list,  # [d-d, d-t, p-p]
                                    idx1=val_sample[j[0]:j[1], 0].reshape(-1),
                                    idx2=val_sample[j[0]:j[1], 1].reshape(-1),
                                    idx3=val_SE[j[0]:j[1]])

                val_output.append(val_output_.reshape(1, -1))

        val_output = th.cat(val_output, dim=1).reshape(-1)  #
        val_mean_acc, val_acc_tensor = class_acc(val_output, val_label, val_SE)
        ee_time = time.time()
        print('Epoch: ', epoch,
              'valid metric: %.4f' % val_mean_acc,
              'epoch time: %.2f' % (ee_time - es_time))
        logging.info('Epoch: {}, validation acc: {:.4f}, epoch time: {:.2f}'.format(epoch,
                                                                                    val_mean_acc,
                                                                                    (ee_time - es_time)))

        if val_mean_acc > max_val_acc:
            print('##################  Save Model  #####################')
            sub_acc = val_mean_acc - max_val_acc
            max_val_acc = val_mean_acc
            optm_epoch = epoch

            optm_state = {'featFusion': model.state_dict(),
                          'optimizer': optm.state_dict(),
                          'epoch': epoch}
            th.save(optm_state, exper_name + f'/model parameter {ite}.pth')
        print('optm_epoch', optm_epoch,
              'max validation metric: %.4f' % max_val_acc,
              'gain by: %.4f' % sub_acc)
        logging.info('optm_epoch: {}, max validation acc: {:.4f}, gain by: {:.4f}'.format(optm_epoch,
                                                                                          max_val_acc,
                                                                                          sub_acc))
    TE_time = time.time()
    print('total time: %.2f' % (TE_time - TS_time),
          'optimal epoch: ', optm_epoch,
          'valid mean metric: %.4f' % max_val_acc,
          'metric: ', 'acc')
    logging.info('total time: {:.2f}, optimal epoch: {}, valid mean acc: {:.4f}'.format((TE_time - TS_time),
                                                                                        optm_epoch,
                                                                                        max_val_acc))
    result_dic = {'total time': (TE_time - TS_time),
                  'optm epoch': optm_epoch,
                  'optm val metric': max_val_acc,
                  'metric': 'acc'}
    return result_dic

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

 ###############################################
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
            for j in tqdm(test_ite_list):
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

    return test_output.cpu().detach()

def main(args):
    out_dim = args.out_dim
    lr = args.lr
    batch_size = args.batch_size
    size = args.size

    NEGA_NUM = args.NEGA_NUM

    ablation = {'method': args.method,
                'cmlp': args.cmlp}
    o_fun = args.o_fun
    classifier = args.classifier
    date = args.date
    dev = args.device
    method_log = '{}-{}-{}'.format(ablation['method'], o_fun, classifier)

    header = '../regular/'
    para = 'out_dim={}_lr={}_method={}_nega={}'.format(out_dim, lr, method_log, NEGA_NUM)
    exper_name = '{}-{}'.format(date, para)
    exper_name = header + exper_name

    if not os.path.exists(exper_name):
        os.makedirs(exper_name)

    logging.basicConfig(filename=exper_name + '/result.log', level=logging.INFO)
    logging.info("Regular Experiment:{}".format(exper_name))

    kg_path = get_path(NEGA_NUM, 'multi-label-regular')

    print('Loading....')
    fp, dd_mat, dt_mat, ppi_mat, val_dataset, test_dataset = datasets(kg_path)
    dd_mat, dt_mat, ppi_mat, fp = dd_mat.to(dev), dt_mat.to(dev), ppi_mat.to(dev), fp.to(dev)

    batchTrain_dataset = arange_batch(kg_path, batch_size)
    val_sample = val_dataset['sample'].to(dev)
    val_SE = th.tensor(val_dataset['side effect']).to(dev)
    val_label = th.tensor(val_dataset['label']).float().to(dev)

    drug_mark = None
    results = {}
    ite5_stime = time.time()
    for ite in range(5):
        results[ite] = epoch_train(ite, dev, exper_name, size,
                                   fp, dd_mat, dt_mat, ppi_mat, batchTrain_dataset,
                                   val_sample, val_SE, val_label,
                                   out_dim, lr, batch_size, ablation,
                                   drug_mark, o_fun, classifier)

    ite5_etime = time.time()
    print('Total time{:.2f}'.format(ite5_etime - ite5_stime))

    np.save(exper_name + '/results summary.npy', results)

    print(results)
    print(exper_name)

    print('---- Test ------------------------------')
    test_dataset = np.load(kg_path['test_dataset'], allow_pickle=True).item()

    test_sample = test_dataset['sample'].to(dev)
    test_SE = th.tensor(test_dataset['side effect']).to(dev)
    test_label = th.tensor(test_dataset['label'])

    test_acc_list, test_map_list, test_auc_list, test_aupr_list = [], [], [], []
    acc_tensor_list, map_tensor_list, auc_tensor_list, aupr_tensor_list = [], [], [], []

    for ite in range(5):
        print('{} Iteration'.format(ite))
        test_output = model_eval(ite, dev, exper_name, size,
                                 fp, dd_mat, dt_mat, ppi_mat,
                                 test_sample, test_SE,
                                 out_dim, ablation, drug_mark, o_fun, classifier)
        th.save(test_output, exper_name + f'/test_output {ite}.pth')
        test_acc, test_acc_tensor = class_acc(test_output, test_label.float(), test_SE)
        test_map, test_map_tensor = mapk_(test_output, test_label, test_SE, k=50)
        test_auc, test_auc_tensor = class_auc(test_output, test_label, test_SE)
        test_aupr, test_aupr_tensor = class_aupr(test_output, test_label, test_SE)
        print(
            'mean acc: {:.4f}\nMAP@50: {:.4f}\nmean auc: {:.4f}\nmean aupr: {:.4f}'.format(test_acc, test_map, test_auc,
                                                                                           test_aupr))
        print('mode \t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_acc, test_auc,
                                                             test_aupr, test_map))
        test_acc_list.append(test_acc)
        test_map_list.append(test_map)
        test_auc_list.append(test_auc)
        test_aupr_list.append(test_aupr)

        acc_tensor_list.append(test_acc_tensor)
        map_tensor_list.append(test_map_tensor)
        auc_tensor_list.append(test_auc_tensor)
        aupr_tensor_list.append(test_aupr_tensor)

    mean_test_acc = th.mean(th.tensor(test_acc_list))
    mean_test_map = th.mean(th.tensor(test_map_list))
    mean_test_auc = th.mean(th.tensor(test_auc_list))
    mean_test_aupr = th.mean(th.tensor(test_aupr_list))
    print('------5 times results-------')
    print('mean acc: {:.4f}\nmean auc: {:.4f}\nmean aupr: {:.4f}\nMAP@50: {:.4f}'.format(mean_test_acc,
                                                                                         mean_test_auc,
                                                                                         mean_test_aupr,
                                                                                         mean_test_map))
    logging.info('mean acc: {:.4f}\nmean auc: {:.4f}\nmean aupr: {:.4f}\nMAP@50: {:.4f}'.format(mean_test_acc,
                                                                                                mean_test_auc,
                                                                                                mean_test_aupr,
                                                                                                mean_test_map))

    var_test_acc = th.var(th.tensor(test_acc_list))
    var_test_map = th.var(th.tensor(test_map_list))
    var_test_auc = th.var(th.tensor(test_auc_list))
    var_test_aupr = th.var(th.tensor(test_aupr_list))
    print('------5 times results-------')
    print('var acc: {:.4f}\nvar auc: {:.4f}\nvar aupr: {:.4f}\nMAP@50: {:.4f}'.format(var_test_acc,
                                                                                      var_test_auc,
                                                                                      var_test_aupr,
                                                                                      var_test_map))
    logging.info('var acc: {:.4f}\nvar auc: {:.4f}\nvar aupr: {:.4f}\nMAP@50: {:.4f}'.format(var_test_acc,
                                                                                             var_test_auc,
                                                                                             var_test_aupr,
                                                                                             var_test_map))
    th.save(test_acc_list, exper_name + '/test_acc_list.pth')
    th.save(test_auc_list, exper_name + '/test_auc_list.pth')
    th.save(test_aupr_list, exper_name + '/test_aupr_list.pth')
    th.save(test_map_list, exper_name + '/test_map_list.pth')

    th.save(acc_tensor_list, exper_name + '/acc_tensor_list.pth')
    th.save(auc_tensor_list, exper_name + '/auc_tensor_list.pth')
    th.save(aupr_tensor_list, exper_name + '/aupr_tensor_list.pth')
    th.save(map_tensor_list, exper_name + '/map_tensor_list.pth')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dim', type=int, default=64, help='The dimension of output.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0005,  help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=5, help='The batch size for side effect')
    parser.add_argument('--size', type=int, default=4, help='The batch size for validation set')
    parser.add_argument('--NEGA_NUM', type=str, default='1to1',
                        help='Num. of positive samples: Num. of negative samples')
    parser.add_argument('--task', type=str, default='c2',
                        help='Task A: c2, Task B: c3')
    parser.add_argument('--method', type=str, default='ASC', help='ASC, SC or quate')
    parser.add_argument('--cmlp', type=bool, default=False,  help='Whether use complex-valued linear transformation')
    parser.add_argument('--o_fun', type=str, default='RE', help='SUM or RE')
    parser.add_argument('--classifier', type=str, default='cip', help='mlp or cip')
    parser.add_argument('--date', type=str, default='23.xx.xx', help='Experiment date')
    arg = parser.parse_args()

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    arg.device = device

    main(arg)


