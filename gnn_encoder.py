import torch
import torch.nn as nn
import torch.nn.functional as F


def get_nan(var, var_name, phase):
    print('nan in ' + var_name + ' when ' + phase)
    print(var_name + ' nan num：', torch.sum(torch.isnan(var)))
    print(var_name + ' nan index：', torch.nonzero(torch.isnan(torch.sum(var, dim=1))))

class graph_node_update(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_base):
        super(graph_node_update, self).__init__()
        self.num_base = num_base
        if in_feat % self.num_base != 0 or out_feat % self.num_base != 0:
            raise ValueError(
                'Feature size must be a multiplier of num_bases (%d).'
                % self.num_bases
            )
        self.submat_in = in_feat // num_base
        self.submat_out = out_feat // num_base
        self.num_bases = num_base
        self.out_feat = out_feat

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(1, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))

        nn.init.xavier_uniform_(self.loop_weight,
                                gain=nn.init.calculate_gain('relu'))
        self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=False)

    def forward(self,
                adj,
                x):
        tmp_w = self.weight.view(self.num_bases, self.submat_in, self.submat_out)
        tmp_h = x.view(-1, self.num_bases, self.submat_in)
        x1 = torch.einsum('abc,bcd->abd', tmp_h, tmp_w)
        x1 = x1.reshape(-1, self.out_feat)
        x1 = torch.mm(adj, x1)
        x2 = torch.mm(x, self.loop_weight)
        x = self.layer_norm_weight(x1 + x2 + 1e-6)
        return x

class gnn_based_fp(nn.Module):
    def __init__(self, h_dim, out_dim, fp_dim, num_base, dp):
        super(gnn_based_fp, self).__init__()
        self.fp_init = nn.Linear(fp_dim, h_dim, bias=False)

        self.node_update_layer1 = graph_node_update(h_dim, out_dim, num_base)
        self.node_update_layer2 = graph_node_update(out_dim, out_dim, num_base)

        self.dropout = nn.Dropout(dp)
    def forward(self,
                mat,
                fp):
        x_fp = self.fp_init(fp)
        x = self.node_update_layer1(mat, x_fp)
        x = self.node_update_layer2(mat, x)

        return x_fp, x

class rela_graph_node_update(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_base):
        super(rela_graph_node_update, self).__init__()
        self.num_base = num_base
        if in_feat % self.num_base != 0 or out_feat % self.num_base != 0:
            raise ValueError(
                'Feature size must be a multiplier of num_bases (%d).'
                % self.num_bases
            )
        self.submat_in = in_feat // num_base
        self.submat_out = out_feat // num_base
        self.num_bases = num_base
        self.out_feat = out_feat

        # assuming in_feat and out_feat are both divisible by num_bases\
        self.weight1 = nn.Parameter(torch.Tensor(1, self.num_bases * self.submat_in * self.submat_out)) # drug-drug
        self.weight2 = nn.Parameter(torch.Tensor(1, self.num_bases * self.submat_in * self.submat_out)) # drug-protein
        self.weight3 = nn.Parameter(torch.Tensor(1, self.num_bases * self.submat_in * self.submat_out)) # protein-drug
        self.weight4 = nn.Parameter(torch.Tensor(1, self.num_bases * self.submat_in * self.submat_out)) # protein-protein
        nn.init.xavier_uniform_(self.weight1, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight2, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight3, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight4, gain=nn.init.calculate_gain('relu'))

        self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat),
                                        requires_grad=True)
        self.loop_weight_p = nn.Parameter(torch.Tensor(in_feat, out_feat),
                                        requires_grad=True)

        nn.init.xavier_uniform_(self.loop_weight,
                                gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.loop_weight_p,
        #                         gain=nn.init.calculate_gain('relu'))
        self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=False)

    def forward(self,
                node,
                adj,
                x):
        if node == 'drug':  # adj: [d-d, d-t], x: [drug, target]
            # drug-drug
            tmp_w1 = self.weight1.view(self.num_bases, self.submat_in, self.submat_out)
            tmp_h = x[0].view(-1, self.num_bases, self.submat_in)
            x1 = torch.einsum('abc,bcd->abd', tmp_h, tmp_w1)
            x1 = x1.reshape(-1, self.out_feat)
            if torch.sum(torch.isnan(x1)) > 0:
                get_nan(x1, 'x1', 'drug node update: drug-drug; before agg')
            x1 = torch.mm(self.dp(adj[0]), x1)
            if torch.sum(torch.isnan(x1)) > 0:
                get_nan(x1, 'x1', 'drug node update: drug-drug; after agg')

            # drug-target
            tmp_w2 = self.weight2.view(self.num_bases, self.submat_in, self.submat_out)
            tmp_h = x[1].view(-1, self.num_bases, self.submat_in)
            x2 = torch.einsum('abc,bcd->abd', tmp_h, tmp_w2)
            x2 = x2.reshape(-1, self.out_feat)
            if torch.sum(torch.isnan(x2)) > 0:
                get_nan(x2, 'x2', 'drug node update: drug-target; before agg')
            x2 = torch.mm(adj[1], x2)
            if torch.sum(torch.isnan(x2)) > 0:
                get_nan(x2, 'x2', 'drug node update: drug-target; after agg')

            # drug self
            x3 = torch.mm(x[0], self.loop_weight)
            if torch.sum(torch.isnan(x3)) > 0:
                get_nan(x3, 'x3', 'drug node update: self; after weight')
            x_new = self.layer_norm_weight(x1 + x2 + x3 + 1e-6)


        elif node == 'target':  # adj: [t-d], x: [drug, target]
            # target-drug
            tmp_w3 = self.weight3.view(self.num_bases, self.submat_in, self.submat_out)
            tmp_h = x[0].view(-1, self.num_bases, self.submat_in)
            x1 = torch.einsum('abc,bcd->abd', tmp_h, tmp_w3)
            x1 = x1.reshape(-1, self.out_feat)
            if torch.sum(torch.isnan(x1)) > 0:
                get_nan(x1, 'x1', 'target node update: target-drug; before agg')
            x1 = torch.mm(adj[0], x1)
            if torch.sum(torch.isnan(x1)) > 0:
                get_nan(x1, 'x1', 'target node update: target-drug; after agg')

            # target self
            x2 = torch.mm(x[1], self.loop_weight)
            x_new = self.layer_norm_weight(x1 + x2 + 1e-6)
        elif node == 'protein': # adj: [p-p, t-d], x: [protein, drug]
            # protein-protein
            tmp_w4 = self.weight4.view(self.num_bases, self.submat_in, self.submat_out)
            tmp_h = x[0].view(-1, self.num_bases, self.submat_in)
            x1 = torch.einsum('abc,bcd->abd', tmp_h, tmp_w4)
            x1 = x1.reshape(-1, self.out_feat)
            if torch.sum(torch.isnan(x1)) > 0:
                get_nan(x1, 'x1', 'protein node update: protein-protein; before agg')
            x1 = torch.matmul(adj[0], x1)
            if torch.sum(torch.isnan(x1)) > 0:
                get_nan(x1, 'x1', 'protein node update: protein-protein; after agg')

            # target-drug
            tmp_w3 = self.weight3.view(self.num_bases, self.submat_in, self.submat_out)
            tmp_h = x[1].view(-1, self.num_bases, self.submat_in)
            x2 = torch.einsum('abc,bcd->abd', tmp_h, tmp_w3)
            x2 = x2.reshape(-1, self.out_feat)
            if torch.sum(torch.isnan(x2)) > 0:
                get_nan(x2, 'x2', 'protein node update: target-drug; before agg')
            x2 = torch.mm(adj[1], x2)
            if torch.sum(torch.isnan(x2)) > 0:
                get_nan(x2, 'x2', 'protein node update: target-drug; after agg')
            # if torch.sum(torch.isnan(x2)) > 0:
            #     print('node == protein|| x2: ', x2)

            # protein self
            x3 = torch.mm(x[0], self.loop_weight)
            x_new = x1 + x3
            x_new[:adj[1].shape[0]] = x_new[:adj[1].shape[0]] + x2
            x_new = self.layer_norm_weight(x_new + 1e-6)
        else:
            x_new = x
        return x_new

class multi_rela_graph_node_update(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_base,
                 target_num,
                 dp):
        super(multi_rela_graph_node_update, self).__init__()
        self.target_num = target_num
        self.node_update_layer = rela_graph_node_update(in_feat, out_feat, num_base)

        self.dropout = nn.Dropout(dp)
    def forward(self, x_drug, x_protein, adj_list):
        x_target = x_protein[:self.target_num]
        x_drug_new = self.node_update_layer('drug',
                                            adj_list[:2],
                                            [x_drug, x_target])
        x_protein_new = self.node_update_layer('protein',
                                               [adj_list[2], adj_list[1].t()],
                                               [x_protein, x_drug])
        return x_drug_new, x_protein_new

class gnn_based_protein(nn.Module):
    def __init__(self, drug_node_num, protein_node_num, h_dim, out_dim,
                 num_base, target_num, dropout, drug_mark=None):
        super(gnn_based_protein, self).__init__()
        self.target_num = target_num
        self.drug_mark = drug_mark  # mark 掉不存在的drug
        self.drug_embedding = nn.Embedding(drug_node_num, h_dim)

        self.kgnode_embedding = nn.Embedding(protein_node_num, h_dim)

        self.node_update_layer1 = multi_rela_graph_node_update(h_dim, out_dim,
                                                               num_base, self.target_num, dropout)
        self.node_update_layer2 = multi_rela_graph_node_update(out_dim, out_dim,
                                                               num_base, self.target_num, dropout)
    def forward(self,
                drug_node_id,
                kg_node_id,
                adj_list # [d-d, d-t, p-p]
                ):
        x_drug_init = self.drug_embedding(drug_node_id)
        if self.drug_mark != None:
            x_drug_init[self.drug_mark] = 0.0
        x_kg = self.kgnode_embedding(kg_node_id)

        x_drug, x_kg = self.node_update_layer1(x_drug_init, x_kg, adj_list)

        x_drug, x_kg = self.node_update_layer2(x_drug, x_kg, adj_list)

        return x_drug_init, x_drug

class encoder(nn.Module):
    def __init__(self, drug_node_num, protein_node_num, fp_dim,
                 h_dim, out_dim, num_base, target_num, dropout, drug_mark=None):
        super(encoder, self).__init__()
        self.gnn_fp = gnn_based_fp(h_dim, out_dim, fp_dim, num_base, dropout)
        self.gnn_protein = gnn_based_protein(drug_node_num, protein_node_num,
                                             h_dim, out_dim, num_base, target_num,
                                             dropout, drug_mark)

        self.dp = nn.Dropout(dropout)
    def forward(self,
                fp,
                drug_node_id,
                kg_node_id,
                adj_list  # [d-d, d-t, p-p, d-p]
                ):
        drug_fp, x1 = self.gnn_fp(self.dp(adj_list[0]), fp)
        drug_init, x2 = self.gnn_protein(drug_node_id, kg_node_id, adj_list)

        return drug_fp, drug_init, x1, x2
