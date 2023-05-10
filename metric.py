from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
import torch as th

def class_auc(output, label, side_effect):
    auc_tensor = []
    for se in range(964):
        auc = roc_auc_score(label[side_effect == se].numpy(), output[side_effect == se].numpy())
        auc_tensor.append(auc)
    auc_tensor = th.tensor(auc_tensor)
    mean_auc = th.mean(auc_tensor)
    return mean_auc, auc_tensor

def tc_class_auc(output, label, side_effect):
    auc_tensor = []
    for se in range(964):
        if (side_effect == se).float().sum() > 0:
            # bool->float，如果全是FALSE，那么加和为0，只有大于0的情况下，有样本的情况下，才值得做这个标签的acc计算
            auc = roc_auc_score(label[side_effect == se].numpy(), output[side_effect == se].numpy())
            auc_tensor.append(auc)
    auc_tensor = th.tensor(auc_tensor)
    mean_auc = th.mean(auc_tensor)
    return mean_auc, auc_tensor

def get_aupr(label, prob):
    precision, recall, _ = precision_recall_curve(label, prob)
    return auc(recall, precision)

def class_aupr(output, label, side_effect):
    aupr_tensor = []
    for se in range(964):
        aupr = get_aupr(label[side_effect == se].numpy(), output[side_effect == se].numpy())
        aupr_tensor.append(aupr)
    aupr_tensor = th.tensor(aupr_tensor)
    mean_aupr = th.mean(aupr_tensor)
    return mean_aupr, aupr_tensor

def tc_class_aupr(output, label, side_effect):
    aupr_tensor = []
    for se in range(964):
        if (side_effect == se).float().sum() > 0:
            # bool->float，如果全是FALSE，那么加和为0，只有大于0的情况下，有样本的情况下，才值得做这个标签的acc计算
            aupr = get_aupr(label[side_effect == se].numpy(), output[side_effect == se].numpy())
            aupr_tensor.append(aupr)
    aupr_tensor = th.tensor(aupr_tensor)
    mean_aupr = th.mean(aupr_tensor)
    return mean_aupr, aupr_tensor


def apk_(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average precision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    ## 取k
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        ## 判断是不是正样本
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk_(output, label, side_effect, k):
    ap_tensor = []
    for se in range(964):
        prob = output[side_effect == se]
        true = label[side_effect == se]
        true = th.nonzero(true).reshape(-1).tolist()  # 正样本的位置
        ind = th.sort(prob, descending=True)[1].tolist()
        ap_tensor.append(apk_(true, ind, k))

    ap_tensor = th.tensor(ap_tensor)
    mean_ap = th.mean(ap_tensor)
    return mean_ap, ap_tensor

def tc_mapk_(output, label, side_effect, k):
    ap_tensor = []
    for se in range(964):
        if (side_effect == se).float().sum() > 0:
            # bool->float，如果全是FALSE，那么加和为0，只有大于0的情况下，有样本的情况下，才值得做这个标签的acc计算
            prob = output[side_effect == se]
            true = label[side_effect == se]
            true = th.nonzero(true).reshape(-1).tolist()  # 正样本的位置
            ind = th.sort(prob, descending=True)[1].tolist()
            ap_tensor.append(apk_(true, ind, k))

    ap_tensor = th.tensor(ap_tensor)
    mean_ap = th.mean(ap_tensor)
    return mean_ap, ap_tensor

def Com_acc(output, lab):
    output = output.reshape(-1)
    lab = lab.reshape(-1)
    result = output.ge(0.5).float() == lab
    acc = result.float().mean()
    return acc


def class_acc(output, label, side_effect):
    acc_tensor = []
    for se in range(964):
        acc = Com_acc(output[side_effect == se], label[side_effect == se]).item()
        # print('side_effect==se sum',th.sum(side_effect == se),'Com_acc',acc)
        acc_tensor.append(acc)
    acc_tensor = th.tensor(acc_tensor)
    mean_acc = th.mean(acc_tensor)
    return mean_acc, acc_tensor

def tc_class_acc(output, label, side_effect):
    acc_tensor = []
    for se in range(964):
        if (side_effect == se).float().sum() > 0:
            # bool->float，如果全是FALSE，那么加和为0，只有大于0的情况下，有样本的情况下，才值得做这个标签的acc计算
            acc = Com_acc(output[side_effect == se], label[side_effect == se]).item()
            # print('side_effect==se sum',th.sum(side_effect == se),'Com_acc',acc)
            acc_tensor.append(acc)
    acc_tensor = th.tensor(acc_tensor)
    mean_acc = th.mean(acc_tensor)
    return mean_acc, acc_tensor