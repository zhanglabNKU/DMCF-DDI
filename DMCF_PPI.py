import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean

class graph_node_update(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_base):
        super(graph_node_update, self).__init__()
        self.gcn = GCNConv(in_channels=in_feat, out_channels=out_feat)
        self.lin = nn.Linear(in_features=in_feat, out_features=out_feat, bias=False)
        self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)

    def forward(self,
                adj,
                x):

        x1 = self.gcn(x, adj)
        x2 = self.lin(x)
        x = self.layer_norm_weight(x1 + x2 + 1e-6)
        return x

class gnn_based_seq(nn.Module):
    def __init__(self, h_dim, out_dim, seq_dim, num_base, dp, isdp, act, inductive=False):
        super(gnn_based_seq, self).__init__()
        self.seq_init = nn.Linear(seq_dim, h_dim, bias=False)

        self.node_update_layer1 = graph_node_update(h_dim, out_dim, num_base)
        self.node_update_layer2 = graph_node_update(out_dim, out_dim, num_base)

        self.dropout = nn.Dropout(dp)
        self.isdp = isdp
        self.act = act
    def forward(self,
                mat,
                seq):
        if self.act:
            x_seq = F.relu(self.seq_init(seq))
        else:
            x_seq = self.seq_init(seq)

        x = self.node_update_layer1(mat, x_seq)

        if self.act:
            x = F.relu(x)
        if self.isdp:
            x = self.dropout(x)

        x = self.node_update_layer2(mat, x)

        return x_seq, x

class rela_graph_node_update(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_base):
        super(rela_graph_node_update, self).__init__()

        self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=False)
        self.gcn_pp = GCNConv(in_channels=in_feat, out_channels=out_feat)
        self.lin_td = nn.Linear(in_features=in_feat, out_features=out_feat)
        self.lin_pr = nn.Linear(in_features=in_feat, out_features=out_feat)

        self.gcn_dd = GCNConv(in_channels=in_feat, out_channels=out_feat)
        self.lin_dt = nn.Linear(in_features=in_feat, out_features=out_feat)
        self.lin_dr = nn.Linear(in_features=in_feat, out_features=out_feat)

    def forward(self,
                node,
                adj,
                x):
        protein_num = x['prot'].shape[0]
        drug_num = x['drug'].shape[0]
        if node == 'protein':  # adj: [d-d, d-t], x: [drug, target]

            x1 = self.gcn_pp(x['prot'], adj['ppi'])
            x2 = scatter_mean(self.lin_td(x['drug'])[adj['dti'][0]], adj['dti'][1], dim=0, dim_size=protein_num)
            x3 = self.lin_pr(x['prot'])
            x_new = self.layer_norm_weight(x1 + x2 + x3 + 1e-6)

        elif node == 'drug': # adj: [p-p, t-d], x: [protein, drug]

            x1 = self.gcn_dd(x['drug'], adj['ddi'])
            x2 = scatter_mean(self.lin_dt(x['prot'])[adj['dti'][1]], adj['dti'][0], dim=0, dim_size=drug_num)
            x3 = self.lin_dr(x['drug'])
            x_new = self.layer_norm_weight(x1 + x2 + x3 + 1e-6)
        else:
            x_new = x
        return x_new

class multi_rela_graph_node_update(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_base,
                 dp):
        super(multi_rela_graph_node_update, self).__init__()
        self.node_update_layer = rela_graph_node_update(in_feat, out_feat, num_base)

        self.dropout = nn.Dropout(dp)
    def forward(self, x_drug, x_protein, adj_list):
        xs = {'prot':x_protein, 'drug':x_drug}
        x_drug_new = self.node_update_layer('drug', adj_list, xs)
        x_protein_new = self.node_update_layer('protein', adj_list, xs)
        return x_drug_new, x_protein_new

class gnn_based_pkg(nn.Module):
    def __init__(self, drug_node_num, protein_node_num, h_dim, out_dim,
                 num_base, dropout, isdp, act, device, inductive=False):
        super(gnn_based_pkg, self).__init__()

        self.drug_id = torch.arange(drug_node_num).to(device)
        self.prot_id = torch.arange(protein_node_num).to(device)
        self.drug_embedding = nn.Embedding(drug_node_num, h_dim)
        self.protein_node_num = protein_node_num
        self.inductive = inductive
        if not inductive:
            self.prot_embedding = nn.Embedding(protein_node_num, h_dim)

        self.node_update_layer1 = multi_rela_graph_node_update(h_dim, out_dim,
                                                               num_base, dropout)
        self.node_update_layer2 = multi_rela_graph_node_update(out_dim, out_dim,
                                                               num_base, dropout)
        self.dropout = nn.Dropout(dropout)
        self.isdp = isdp
        self.act = act

    def forward(self,
                adj_list # [d-d, d-t, p-p]
                ):
        x_drug_init = self.drug_embedding(self.drug_id)
        if not self.inductive:
            x_prot_init = self.prot_embedding(self.prot_id)
        else:
            x_prot_init = scatter_mean(x_drug_init[adj_list['dti'][0]], adj_list['dti'][1], dim=0,
                                       dim_size=self.protein_node_num)

        x_drug, x_prot = self.node_update_layer1(x_drug_init, x_prot_init, adj_list)
        if self.act:
            x_drug = F.relu(x_drug)
            x_prot = F.relu(x_prot)
        if self.isdp:
            x_drug = self.dropout(x_drug)
            x_prot = self.dropout(x_prot)

        x_drug, x_prot = self.node_update_layer2(x_drug, x_prot, adj_list)

        return x_prot_init, x_prot, x_drug

class gnn_encoder(nn.Module):
    def __init__(self, drug_node_num, protein_node_num, seq_dim, act,
                 h_dim, out_dim, num_base, dropout, isdp, device, inductive=False):
        super(gnn_encoder, self).__init__()
        self.gnn_seq = gnn_based_seq(h_dim, out_dim, seq_dim, num_base, dropout, isdp, act, inductive)
        self.gnn_pkg = gnn_based_pkg(drug_node_num, protein_node_num, h_dim, out_dim,
                                     num_base, dropout, isdp, act, device, inductive)

        self.leakrelu = nn.LeakyReLU(0.2)

    def forward(self,
                seq,
                adj_list):
        x_seq, x1 = self.gnn_seq(adj_list['ppi'], seq)
        x_prot_init, x2, x_drug = self.gnn_pkg(adj_list)

        return x_seq, x_prot_init, x1, x2, x_drug


class complex_fuse_base(nn.Module):
    def __init__(self, drug_node_num, protein_node_num, seq_dim,
                 h_dim, out_dim, num_base, dropout, operation,
                 isdp, act, cmlp, device, inductive=False, method='complex',
                 classifier='cip'):
        super(complex_fuse_base, self).__init__()
        self.cmlp = cmlp
        self.o_fun = operation
        self.encoder = gnn_encoder(drug_node_num, protein_node_num, seq_dim, act,
                                   h_dim, out_dim, num_base, dropout, isdp, device, inductive)

        self.seq_fc_layer = nn.Linear(h_dim, out_dim, bias=False)
        self.skip_fc_layer = nn.Linear(h_dim, out_dim, bias=False)

        self.layer_norm_weight = nn.LayerNorm(out_dim, elementwise_affine=False)
        self.layer_norm_weight_hdim = nn.LayerNorm(h_dim, elementwise_affine=False)

        if self.cmlp:

            self.rela_linear_r1 = nn.Parameter(torch.Tensor(out_dim, out_dim))
            self.rela_linear_r2 = nn.Parameter(torch.Tensor(out_dim, out_dim))
            self.rela_linear_i1 = nn.Parameter(torch.Tensor(out_dim, out_dim))
            self.rela_linear_i2 = nn.Parameter(torch.Tensor(out_dim, out_dim))
            nn.init.xavier_uniform_(self.rela_linear_r1,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.rela_linear_r2,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.rela_linear_i1,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.rela_linear_i2,
                                    gain=nn.init.calculate_gain('relu'))

        self.method = method
        if classifier == 'mlp':
            self.w_relation = nn.Parameter(torch.Tensor(1, 2 * out_dim))  # C,F
            nn.init.xavier_uniform_(self.w_relation,
                                    gain=nn.init.calculate_gain('sigmoid'))
            self.w_relation.data = self.w_relation.data.reshape(-1)

        if classifier == 'cip':
            self.w_relationR = nn.Parameter(torch.Tensor(1, out_dim))
            self.w_relationI = nn.Parameter(torch.Tensor(1, out_dim))
            nn.init.xavier_uniform_(self.w_relationR,
                                    gain=nn.init.calculate_gain('sigmoid'))
            nn.init.xavier_uniform_(self.w_relationI,
                                    gain=nn.init.calculate_gain('sigmoid'))
            self.w_relationR.data = self.w_relationR.data.reshape(-1)
            self.w_relationI.data = self.w_relationI.data.reshape(-1)

        if method == 'quate':
            self.mlp = nn.Parameter(torch.Tensor(4 * out_dim, out_dim))
            nn.init.xavier_uniform_(self.mlp,
                                    gain=nn.init.calculate_gain('relu'))
            self.w_relation = nn.Parameter(torch.Tensor(1, out_dim))  # C,F
            nn.init.xavier_uniform_(self.w_relation,
                                    gain=nn.init.calculate_gain('sigmoid'))
            self.w_relation.data = self.w_relation.data.reshape(-1)

        self.b_relation = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self,
                seq,
                adj_list,  # [d-d, d-t, p-p]
                idx1,
                idx2):

        drug_fp, drug_init, x1, x2, protfeat = self.encoder(seq, adj_list)

        output = None

        drug_fp = F.elu(self.layer_norm_weight_hdim(drug_fp))
        x_fp = F.elu(self.seq_fc_layer(drug_fp))
        x_drug_skip = self.skip_fc_layer(drug_init)

        if self.method == 'ASC':
            R1 = x1[idx1] + x_fp[idx2]
            Ima1 = x2[idx1] + x_drug_skip[idx2]
            R2 = x_drug_skip[idx1] + x2[idx2]
            Ima2 = x_fp[idx1] + x1[idx2]
            R1 = self.layer_norm_weight(R1)
            Ima1 = self.layer_norm_weight(Ima1)
            R2 = self.layer_norm_weight(R2)
            Ima2 = self.layer_norm_weight(Ima2)

            R, Ima = self.complex_mult(R1, Ima1, R2, Ima2)

            R = self.layer_norm_weight(R)
            Ima = self.layer_norm_weight(Ima)

            if self.classifier == 'mlp':
                lr = torch.cat((R, Ima), dim=1)
                output = torch.sigmoid(torch.sum(lr * self.w_relation, dim=1).reshape(-1) + \
                                       self.b_relation)
            if self.classifier == 'cip':
                r_, i_ = self.complex_inner(R, Ima, self.w_relationR, self.w_relationI)
                if self.o_fun == 'SUM':
                    output = torch.sigmoid(r_ + i_)
                if self.o_fun == 'RE':
                    output = torch.sigmoid(r_)

        elif self.method == 'quate':
            x1 = self.layer_norm_weight(x1)
            x2 = self.layer_norm_weight(x2)
            x_fp = self.layer_norm_weight(x_fp)
            x_drug_skip = self.layer_norm_weight(x_drug_skip)
            a, b, c, d = x1[idx1], x2[idx1], x_fp[idx1], x_drug_skip[idx1]
            p, q, u, v = x1[idx2], x2[idx2], x_fp[idx2], x_drug_skip[idx2]
            R = a * p - b * q - c * u - d * v
            Ima1 = a * q + b * p + c * v - d * u
            Ima2 = a * u - b * v + c * p + d * q
            Ima3 = a * v + b * u - c * q + d * p
            lr = torch.cat((R, Ima1, Ima2, Ima3), dim=1)
            lr = torch.mm(lr, self.mlp)
            output = torch.sigmoid(torch.sum(lr * self.w_relation, dim=1).reshape(-1) + \
                                   self.b_relation)

        return output

    def get_R_Ima(self, seq, adj_list,  idx1, idx2):
        drug_fp, drug_init, x1, x2, protfeat = self.encoder(seq, adj_list)
        drug_fp = F.elu(self.layer_norm_weight_hdim(drug_fp))
        x_fp = F.elu(self.seq_fc_layer(drug_fp))
        x_drug_skip = self.skip_fc_layer(drug_init)

        R1 = x1[idx1] + x_fp[idx2]
        Ima1 = x2[idx1] + x_drug_skip[idx2]
        R2 = x_drug_skip[idx1] + x2[idx2]
        Ima2 = x_fp[idx1] + x1[idx2]

        if self.cmlp:
            R1, Ima1 = self.complex_nn(R1, Ima1,
                                       self.rela_linear_r1, self.rela_linear_i1)
            R2, Ima2 = self.complex_nn(R2, Ima2,
                                       self.rela_linear_r2, self.rela_linear_i2)
        R1 = self.layer_norm_weight(R1)
        Ima1 = self.layer_norm_weight(Ima1)
        R2 = self.layer_norm_weight(R2)
        Ima2 = self.layer_norm_weight(Ima2)
        R, Ima = self.complex_mult(R1, Ima1, R2, Ima2)

        R = self.layer_norm_weight(R)
        Ima = self.layer_norm_weight(Ima)
        return R, Ima

    def complex_mult(self, R1, Ima1, R2, Ima2):
        R = R1 * R2 - Ima1 * Ima2
        Ima = R1 * Ima2 + Ima1 * R2
        return R, Ima

    def complex_nn(self, R, Ima, W_R, W_I):
        r_trans = torch.mm(R, W_R) - torch.mm(Ima, W_I)
        i_trans = torch.mm(R, W_I) + torch.mm(Ima, W_R)
        r_trans = F.relu(r_trans)
        i_trans = F.relu(i_trans)
        return r_trans, i_trans

    def complex_inner(self, R, Ima, w_r, w_i):
        r_ = torch.sum(R * w_r, dim=1) - torch.sum(Ima * (-1) * w_i, dim=1)
        i_ = torch.sum(R * (-1) * w_i, dim=1) + torch.sum(Ima * w_r, dim=1)
        return r_, i_