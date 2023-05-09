import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn_encoder import encoder

class complex_fuse_base(nn.Module):
    def __init__(self, drug_node_num, protein_node_num, fp_dim,
                 h_dim, out_dim, num_base, target_num, dropout,
                 num_rela, ablation, drug_mark, o_fun, classifier='cip'):
        super(complex_fuse_base, self).__init__()
        self.cmlp = ablation['cmlp']
        self.drug_mark = drug_mark
        self.encoder = encoder(drug_node_num, protein_node_num, fp_dim,
                               h_dim, out_dim, num_base, target_num, dropout,
                               drug_mark=self.drug_mark)
        self.fp_fc_layer = nn.Linear(h_dim, out_dim, bias=False)
        self.skip_fc_layer = nn.Linear(h_dim, out_dim, bias=False)

        self.layer_norm_weight = nn.LayerNorm(out_dim, elementwise_affine=False)
        self.layer_norm_weight_hdim = nn.LayerNorm(h_dim, elementwise_affine=False)

        self.ablation = ablation

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

        self.classifier = classifier
        self.o_fun = o_fun
        if classifier == 'mlp':
            self.w_relation = nn.Parameter(torch.Tensor(num_rela, 2 * out_dim))  # C,F
            nn.init.xavier_uniform_(self.w_relation,
                                    gain=nn.init.calculate_gain('sigmoid'))
        if classifier == 'cip':
            self.w_relationR = nn.Parameter(torch.Tensor(num_rela, out_dim))
            self.w_relationI = nn.Parameter(torch.Tensor(num_rela, out_dim))
            nn.init.xavier_uniform_(self.w_relationR,
                                    gain=nn.init.calculate_gain('sigmoid'))
            nn.init.xavier_uniform_(self.w_relationI,
                                    gain=nn.init.calculate_gain('sigmoid'))

        if ablation['method'] == 'quate':
            self.w_relation = nn.Parameter(torch.Tensor(num_rela, 4 * out_dim))  # C,F
            nn.init.xavier_uniform_(self.w_relation,
                                    gain=nn.init.calculate_gain('sigmoid'))

        self.b_relation = nn.Parameter(torch.zeros(num_rela), requires_grad=True)

    def forward(self,
                fp,
                drug_node_id,
                kg_node_id,
                adj_list,  # [d-d, d-t, p-p]
                idx1,
                idx2,
                idx3):
        drug_fp, drug_init, x1, x2 = self.encoder(fp, drug_node_id, kg_node_id, adj_list)

        drug_fp = F.elu(self.layer_norm_weight_hdim(drug_fp))
        x_fp = F.elu(self.fp_fc_layer(drug_fp))
        x_drug_skip = self.skip_fc_layer(drug_init)
        if self.ablation['method'] == 'ASC':
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

            if self.classifier == 'mlp':
                lr = torch.cat((R, Ima), dim=1)
                output = torch.sigmoid(torch.sum(lr * self.w_relation[idx3], dim=1).reshape(-1) + \
                                       self.b_relation[idx3])
            if self.classifier == 'cip':
                r_, i_ = self.complex_inner(R, Ima, self.w_relationR[idx3], self.w_relationI[idx3])
                if self.o_fun == 'SUM':
                    output = torch.sigmoid(r_ + i_)
                if self.o_fun == 'RE':
                    output = torch.sigmoid(r_)

        if self.ablation['method'] == 'SC':
            x1_intra = x1 + x_fp
            x2_intra = x2 + x_drug_skip
            R1 = x1_intra[idx1]  # h
            Ima1 = x2_intra[idx1]  # h
            R2 = x2_intra[idx2]  # t
            Ima2 = x1_intra[idx2]  # t
            R1 = self.layer_norm_weight(R1)
            Ima1 = self.layer_norm_weight(Ima1)
            R2 = self.layer_norm_weight(R2)
            Ima2 = self.layer_norm_weight(Ima2)

            R, Ima = self.complex_mult(R1, Ima1, R2, Ima2)

            R = self.layer_norm_weight(R)
            Ima = self.layer_norm_weight(Ima)

            if self.classifier == 'mlp':
                lr = torch.cat((R, Ima), dim=1)
                output = torch.sigmoid(torch.sum(lr * self.w_relation[idx3], dim=1).reshape(-1) + \
                                       self.b_relation[idx3])
            if self.classifier == 'cip':
                r_, i_ = self.complex_inner(R, Ima, self.w_relationR[idx3], self.w_relationI[idx3])
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
            output = torch.sigmoid(torch.sum(lr * self.w_relation[idx3], dim=1).reshape(-1) + \
                                   self.b_relation[idx3])

        return output

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