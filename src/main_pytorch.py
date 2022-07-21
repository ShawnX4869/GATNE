import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from torch.nn.parameter import Parameter

from utils import *


def get_batches(pairs, neighbors, batch_size):
    '''
    :param pairs: -- train_pairs: node对(node1, node2, layer_id)
    :param neighbors: neighbors'shape = (num_node, edge_type_count, neighbor_samples)
                      记录每个node所相连的指定数量的邻居nodelist
    :param batch_size:
    :return:torch.tensor(x), source
            torch.tensor(y), end
            torch.tensor(t), edge_type
            torch.tensor(neigh)=edge_type_count, neighbor_samples
    '''
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0]) #source node
            y.append(pairs[index][1]) #end node
            t.append(pairs[index][2]) #edge type
            neigh.append(neighbors[pairs[index][0]]) #source node的邻居
        yield torch.tensor(x), torch.tensor(y), torch.tensor(t), torch.tensor(neigh)


class GATNEModel(nn.Module):
    def __init__(
        self, num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a, features
    ):
        super(GATNEModel, self).__init__()
        self.num_nodes = num_nodes #节点数量
        self.embedding_size = embedding_size #每个节点输出的embedding纬度
        self.embedding_u_size = embedding_u_size #节点作为邻居初始化size
        self.edge_type_count = edge_type_count #类别数量
        self.dim_a = dim_a #中间隐层特征数量
        self.features = None

        if features is not None: #GATNE-I
            self.features = features
            feature_dim = self.features.shape[-1]
            self.embed_trans = Parameter(torch.FloatTensor(feature_dim, embedding_size))
            self.u_embed_trans = Parameter(torch.FloatTensor(edge_type_count, feature_dim, embedding_u_size))
        else: #初始化 base embedding
            self.node_embeddings = Parameter(torch.FloatTensor(num_nodes, embedding_size))
            self.node_type_embeddings = Parameter( #初始化  edge embedding
                torch.FloatTensor(num_nodes, edge_type_count, embedding_u_size)
            )
        self.trans_weights = Parameter( #定义Mr矩阵
            torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size)
        )
        self.trans_weights_s1 = Parameter( #Wr计算attention使用
           torch.FloatTensor(edge_type_count, embedding_u_size, dim_a)
        )
        #wr计算attention使用
        self.trans_weights_s2 = Parameter(torch.FloatTensor(edge_type_count, dim_a, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.features is not None:
            self.embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
            self.u_embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        else:
            self.node_embeddings.data.uniform_(-1.0, 1.0)
            self.node_type_embeddings.data.uniform_(-1.0, 1.0)
        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, train_inputs, train_types, node_neigh):
        '''
        :param train_inputs: torch.tensor(x),     shape=(batch_size, source_node)
        :param train_types:  torch.tensor(t),     shape=(batch_size, edge_type)
        :param node_neigh:   torch.tensor(neigh)  shape=(batch_size, edge_type_count, neighbour_num)
        :return:
        '''
        if self.features is None:
            node_embed = self.node_embeddings[train_inputs]  # batch_size, embedding_size
            node_embed_neighbors = self.node_type_embeddings[node_neigh]
            # batch_size, edge_type_count, neighbour_num, edge_type_count, embedding_u_size

        else:
            node_embed = torch.mm(self.features[train_inputs], self.embed_trans)
            # einsum 爱因斯坦求和表达https://blog.csdn.net/weixin_45101959/article/details/124483226
            # edge_type_count, feature_dim, embedding_u_size = akm
            # num_nodes, edge_type_count, neighbour_num, feature_dim = bijk
            node_embed_neighbors = torch.einsum('bijk,akm->bijam', self.features[node_neigh], self.u_embed_trans)
        # node_embed_tmp=(batch_size, edge_type_count, neighbour_num, embedding_u_size)
        node_embed_tmp = torch.diagonal(node_embed_neighbors, dim1=1, dim2=3).permute(0, 3, 1, 2)
        # node_embed_tmp=(batch_size, edge_type_count, embedding_u_size)
        node_type_embed = torch.sum(node_embed_tmp, dim=2)
        # trans_weights=(edge_type_count, embedding_u_size, embedding_size)
        # train_types = (batch_size)
        trans_w = self.trans_weights[train_types]
        # trans_w = (batch_size, embedding_u_size, embedding_size)
        # trans_weights_s1 = (edge_type_count, embedding_u_size, dim_a)
        trans_w_s1 = self.trans_weights_s1[train_types]
        # trans_w_s1 = (batch_size, embedding_u_size, dim_a)
        # trans_weights_s2= (edge_type_count, dim_a, 1)
        trans_w_s2 = self.trans_weights_s2[train_types]
        # trans_w_s2 = (batch_size, dim_a, 1)
        attention = F.softmax(
            torch.matmul(
                #                 (batch_size, edge_type_count, dim_a)  -->  batch_size, edge_type_count, 1
                torch.tanh(torch.matmul(node_type_embed, trans_w_s1)), trans_w_s2
            ).squeeze(2),
            dim=1,
        ).unsqueeze(1)  # batch_size, 1, edge_type_count
        # batch_size, edge_type_count, embedding_u_size ->  batch_size, 1, embedding_u_size
        node_type_embed = torch.matmul(attention, node_type_embed)
        # batch_size, embedding_size
        node_embed = node_embed + torch.matmul(node_type_embed, trans_w).squeeze(1)

        last_node_embed = F.normalize(node_embed, dim=1)

        return last_node_embed


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size):
        '''
        :param num_nodes:  list of all nodes
        :param num_sampled: 负样本采样数
        :param embedding_size:
        '''
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.sample_weights = F.normalize(
            torch.Tensor(
                [
                    (math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)
                    for k in range(num_nodes) # quest1on
                ]
            ),
            dim=0,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        '''
        :param input: source_node
        :param embs:
        :param label: end_node
        :return:
        '''
        n = input.shape[0] #batch_size
        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1)) #每个source乘以endnode的weight
        )
        #以输入的张量作为权重进行多分布采样
        negs = torch.multinomial(
            self.sample_weights, self.num_sampled * n, replacement=True
        ).view(n, self.num_sampled)
        # negs = (batch_size, num_sampled, embedding_size)
        noise = torch.neg(self.weights[negs]) #取负

        sum_log_sampled = torch.sum(                 #embs.unsqueeze=(batch_size, embedding_size, 1)
            torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1
        ).squeeze()

        loss = log_target + sum_log_sampled  #quest1on
        return -loss.sum() / n



def train_model(network_data, feature_dic):
    '''
    :param network_data: 训练集，key为边对类型，value为相连node对的list
    :param feature_dic: 特征字典，key为node序号，value为对应特征
    :return:
    '''

    #vocab:字典vocab[node]=Node(count,index)实例
    #index2word: 全node列表，顺序与vocab[node].index一致
    #train_pairs: node对(node1, node2, layer_id)
    vocab, index2word, train_pairs = generate(network_data, args.num_walks, args.walk_length,\
    args.schema, file_name, args.window_size, args.num_workers, args.walk_file)
    edge_types = list(network_data.keys())

    num_nodes = len(index2word)
    edge_type_count = len(edge_types)
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions
    embedding_u_size = args.edge_dim
    u_num = edge_type_count
    num_sampled = args.negative_samples
    dim_a = args.att_dim
    att_head = 1
    neighbor_samples = args.neighbor_samples

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    neighbors = generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples)
    #neighbors'shape = (num_node, edge_type_count, neighbor_samples)
    features = None
    if feature_dic is not None:
        feature_dim = len(list(feature_dic.values())[0])
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():
            if key in vocab:
                features[vocab[key].index, :] = np.array(value)
        features = torch.FloatTensor(features).to(device)

    model = GATNEModel(
        num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a, features
    )
    nsloss = NSLoss(num_nodes, num_sampled, embedding_size)

    model.to(device)
    nsloss.to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": nsloss.parameters()}], lr=1e-4
    )

    best_score = 0
    test_score = (0.0, 0.0, 0.0)
    patience = 0
    for epoch in range(epochs):
        random.shuffle(train_pairs)
        batches = get_batches(train_pairs, neighbors, batch_size)
        #each batch[0]= source node
        #each batch[1]= end node
        #each batch[2]= edge type
        #each batch[3]= source node's neighs for each type (shape=(edge_type_count, neighbor_samples))

        data_iter = tqdm(
            batches,
            desc="epoch %d" % (epoch),
            total=(len(train_pairs) + (batch_size - 1)) // batch_size,
            bar_format="{l_bar}{r_bar}",
        )
        avg_loss = 0.0

        for i, data in enumerate(data_iter):
            optimizer.zero_grad()
            embs = model(data[0].to(device), data[2].to(device), data[3].to(device),)
            loss = nsloss(data[0].to(device), embs, data[1].to(device))
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if i % 5000 == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                data_iter.write(str(post_fix))

        final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
        # final_model={edge_type: {}}
        for i in range(num_nodes):
            train_inputs = torch.tensor([i for _ in range(edge_type_count)]).to(device)
            train_types = torch.tensor(list(range(edge_type_count))).to(device)
            node_neigh = torch.tensor(
                [neighbors[i] for _ in range(edge_type_count)]
            ).to(device)
            node_emb = model(train_inputs, train_types, node_neigh) #推理
            for j in range(edge_type_count):
                final_model[edge_types[j]][index2word[i]] = (
                    node_emb[j].cpu().detach().numpy()
                )

        valid_aucs, valid_f1s, valid_prs = [], [], []
        test_aucs, test_f1s, test_prs = [], [], []
        for i in range(edge_type_count):
            if args.eval_type == "all" or edge_types[i] in args.eval_type.split(","):
                tmp_auc, tmp_f1, tmp_pr = evaluate(
                    final_model[edge_types[i]],
                    valid_true_data_by_edge[edge_types[i]], #node对
                    valid_false_data_by_edge[edge_types[i]],#node对
                )
                valid_aucs.append(tmp_auc)
                valid_f1s.append(tmp_f1)
                valid_prs.append(tmp_pr)

                tmp_auc, tmp_f1, tmp_pr = evaluate(
                    final_model[edge_types[i]],
                    testing_true_data_by_edge[edge_types[i]],
                    testing_false_data_by_edge[edge_types[i]],
                )
                test_aucs.append(tmp_auc)
                test_f1s.append(tmp_f1)
                test_prs.append(tmp_pr)
        print("valid auc:", np.mean(valid_aucs))
        print("valid pr:", np.mean(valid_prs))
        print("valid f1:", np.mean(valid_f1s))

        average_auc = np.mean(test_aucs)
        average_f1 = np.mean(test_f1s)
        average_pr = np.mean(test_prs)

        cur_score = np.mean(valid_aucs)
        if cur_score > best_score:
            best_score = cur_score
            test_score = (average_auc, average_f1, average_pr)
            patience = 0
        else:
            patience += 1
            if patience > args.patience:
                print("Early Stopping")
                break
    return test_score


if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    print(args)
    if args.features is not None:
        feature_dic = load_feature_data(args.features)
    else:
        feature_dic = None

    training_data_by_type = load_training_data(file_name + "/train.txt")
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(
        file_name + "/valid.txt"
    )
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(
        file_name + "/test.txt"
    )

    average_auc, average_f1, average_pr = train_model(training_data_by_type, feature_dic)

    print("Overall ROC-AUC:", average_auc)
    print("Overall PR-AUC", average_pr)
    print("Overall F1:", average_f1)
