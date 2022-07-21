import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from torch.nn.parameter import Parameter

# from utils import *
import random
import multiprocessing

from tqdm import tqdm

def walk(args):
    walk_length, start, schema = args
    # Simulate a random walk starting from start node.
    rand = random.Random()

    if schema:
        schema_items = schema.split('-')
        assert schema_items[0] == schema_items[-1]

    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        candidates = []
        for node in G[cur]:
            if schema == '' or node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                candidates.append(node)
        if candidates:
            walk.append(rand.choice(candidates))
        else:
            break
    return [str(node) for node in walk]

def initializer(init_G, init_node_type):
    global G
    G = init_G
    global node_type
    node_type = init_node_type

class RWGraph():
    def __init__(self, nx_G, node_type_arr=None, num_workers=16):
        self.G = nx_G #字典：每个node相连node的集合
        self.node_type = node_type_arr #随机游走规则
        self.num_workers = num_workers

    def node_list(self, nodes, num_walks):
        for loop in range(num_walks):
            for node in nodes:
                yield node

    def simulate_walks(self, num_walks, walk_length, schema=None):
        all_walks = []
        nodes = list(self.G.keys())
        random.shuffle(nodes)

        if schema is None: #pool.imap(函数，入参)
            with multiprocessing.Pool(self.num_workers, initializer=initializer, initargs=(self.G, self.node_type)) as pool:
                all_walks = list(pool.imap(walk, ((walk_length, node, '') \
                                                  for node in tqdm(self.node_list(nodes, num_walks))), chunksize=256))
        else:
            schema_list = schema.split(',')
            for schema_iter in schema_list:
                with multiprocessing.Pool(self.num_workers, initializer=initializer, initargs=(self.G, self.node_type)) as pool:
                    walks = list(pool.imap(walk, ((walk_length, node, schema_iter)\
                            for node in tqdm(self.node_list(nodes, num_walks)) if schema_iter.split('-')[0] == self.node_type[node]), chunksize=512))
                all_walks.extend(walks)
        # all_walks : list of [str(node) for node in walk]
        return all_walks


import argparse
import multiprocessing
from collections import defaultdict
from operator import index

import numpy as np
from six import iteritems
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)
from tqdm import tqdm

# from walk import RWGraph


class Vocab(object):

    def __init__(self, count, index):
        self.count = count
        self.index = index


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='data/amazon',
                        help='Input dataset path')

    parser.add_argument('--features', type=str, default=None,
                        help='Input node features')

    parser.add_argument('--walk-file', type=str, default=None,
                        help='Input random walks')

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch. Default is 100.')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='Number of batch_size. Default is 64.')

    parser.add_argument('--eval-type', type=str, default='all',
                        help='The edge type(s) for evaluation.')

    parser.add_argument('--schema', type=str, default=None,
                        help='The metapath schema (e.g., U-I-U,I-U-I).')

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 200.')

    parser.add_argument('--edge-dim', type=int, default=10,
                        help='Number of edge embedding dimensions. Default is 10.')

    parser.add_argument('--att-dim', type=int, default=20,
                        help='Number of attention dimensions. Default is 20.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--negative-samples', type=int, default=5,
                        help='Negative samples for optimization. Default is 5.')

    parser.add_argument('--neighbor-samples', type=int, default=10,
                        help='Neighbor samples for aggregation. Default is 10.')

    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.')

    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of workers for generating random walks. Default is 16.')

    return parser.parse_args()


def get_G_from_edges(edges):
    '''
    :param edges: node对的列表
    :return: edge_dict：每个node所相连的node集合(set)
    '''
    edge_dict = defaultdict(set)
    for edge in edges:
        u, v = str(edge[0]), str(edge[1])
        edge_dict[u].add(v)
        edge_dict[v].add(u)
    return edge_dict


def load_training_data(f_name):
    '''
    :param f_name: 训练文件
    :return: edge_data_by_type字典，key为边的类型；value为相连的(node1,node2)节点对
    '''
    print('We are loading data from:', f_name)
    edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print('Total training nodes: ' + str(len(all_nodes)))
    return edge_data_by_type


def load_testing_data(f_name):
    print('We are loading data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    return true_edge_data_by_type, false_edge_data_by_type


def load_node_type(f_name):
    print('We are loading node type from:', f_name)
    node_type = {}
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type


def load_feature_data(f_name):
    '''
    :param f_name:
    :return: node特征字典：key为node序号，value为dim维特征
    '''
    feature_dic = {}
    with open(f_name, 'r') as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            items = line.strip().split()
            feature_dic[items[0]] = items[1:]
    return feature_dic


def generate_walks(network_data, num_walks, walk_length, schema, file_name, num_workers):
    if schema is not None:
        node_type = load_node_type(file_name + '/node_type.txt')
    else:
        node_type = None

    all_walks = []
    for layer_id, layer_name in enumerate(network_data):
        tmp_data = network_data[layer_name]  # 同边类型内游走, layer_id就是边类型
        # start to do the random walk on a layer

        layer_walker = RWGraph(get_G_from_edges(tmp_data), node_type, num_workers)
        print('Generating random walks for layer', layer_id)
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)

        all_walks.append(layer_walks)

    print('Finish generating the walks')

    return all_walks


def generate_pairs(all_walks, vocab, window_size, num_workers):
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        print('Generating training pairs for layer', layer_id)
        for walk in tqdm(walks):
            for i in range(len(walk)):  # 游走序列
                for j in range(1, skip_window + 1):  # 1...skip_window
                    if i - j >= 0:  # 序列前面skip_window个都与i产生node对
                        pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index, layer_id))
                    if i + j < len(walk):  # 序列后面skip_window个都与i产生node对
                        pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index, layer_id))
    return pairs


def generate_vocab(all_walks):
    '''
    :param all_walks: 同边类型游走序列
    :return: vocab: node-> node实例(count, index), 序号由高频node到低频node0开始递增
             index2word: nodelist，顺序与vocab的index相同
    '''
    index2word = []
    raw_vocab = defaultdict(int)

    for layer_id, walks in enumerate(all_walks):
        print('Counting vocab for layer', layer_id)
        for walk in tqdm(walks):
            for word in walk:
                raw_vocab[word] += 1
                # 统计每个node出现次数

    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))  # Vocab实例
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)  # 由词频高->低
    for i, word in enumerate(index2word):
        vocab[word].index = i  # 重新赋值序号，由高频->低频

    return vocab, index2word


def load_walks(walk_file):
    print('Loading walks')
    all_walks = []
    with open(walk_file, 'r') as f:
        for line in f:
            content = line.strip().split()
            layer_id = int(content[0])
            if layer_id >= len(all_walks):
                all_walks.append([])
            all_walks[layer_id].append(content[1:])
    return all_walks


def save_walks(walk_file, all_walks):
    with open(walk_file, 'w') as f:
        for layer_id, walks in enumerate(all_walks):
            print('Saving walks for layer', layer_id)
            for walk in tqdm(walks):
                f.write(' '.join([str(layer_id)] + [str(x) for x in walk]) + '\n')


def generate(network_data, num_walks, walk_length, schema, file_name, window_size, num_workers, walk_file):
    # network_data, args.num_walks, args.walk_length, \
    # args.schema, file_name, args.window_size, args.num_workers, args.walk_file
    '''
    :param network_data: 训练集，key为边对类型，value为node对的list
    :param num_walks: 游走序列的个数
    :param walk_length: 游走长度
    :param schema: 游走规则 "nodetype1-nodetype2-nodetype1", split with "-"
    :param file_name: 数据集名称
    :param window_size:
    :param num_workers:多线程数
    :param walk_file:
    :return:vocab:字典vocab[node]=Node(count,index)实例
            index2word:  全node列表，顺序与vocab[node].index一致
            train_pairs: node对(node1, node2, layer_id)
    '''
    if walk_file is not None:
        all_walks = load_walks(walk_file)
    else:
        all_walks = generate_walks(network_data, num_walks, walk_length, schema, file_name, num_workers)
        save_walks(file_name + '/walks.txt', all_walks)
    vocab, index2word = generate_vocab(all_walks)
    train_pairs = generate_pairs(all_walks, vocab, window_size, num_workers)

    return vocab, index2word, train_pairs


def generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples):
    '''
    :param network_data: 边类型：相连node对的list
    :param vocab: node映射字典，vocab[node] = Node(count, index)
    :param num_nodes:  nodes个数
    :param edge_types:  边类型list
    :param neighbor_samples: 记录邻居数量，无边仅自己、不够有放回随机采样、超了随机采样
    :return: neighbors'shape = (num_node, edge_type_count, neighbor_samples)
             记录每个node所相连的指定数量的邻居nodelist
    '''
    edge_type_count = len(edge_types)
    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
    for r in range(edge_type_count):
        print('Generating neighbors for layer', r)
        g = network_data[edge_types[r]]
        for (x, y) in tqdm(g):
            ix = vocab[x].index
            iy = vocab[y].index
            neighbors[ix][r].append(iy)
            neighbors[iy][r].append(ix)
        for i in range(num_nodes):
            if len(neighbors[i][r]) == 0:
                neighbors[i][r] = [i] * neighbor_samples  # 全是自己
            elif len(neighbors[i][r]) < neighbor_samples:
                neighbors[i][r].extend(
                    list(np.random.choice(neighbors[i][r], size=neighbor_samples - len(neighbors[i][r]))))
            elif len(neighbors[i][r]) > neighbor_samples:
                neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))
    return neighbors


def get_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except Exception as e:
        pass


def evaluate(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    true_num = 0
    for edge in true_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    for edge in false_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)



def get_batches(pairs, neighbors, batch_size):
    '''
    :param pairs: -- train_pairs: node对(node1, node2, layer_id)
    :param neighbors: neighbors'shape = (num_node, edge_type_count, neighbor_samples)
                      记录每个node所相连的指定数量的邻居nodelist
    :param batch_size:
    :return:torch.tensor(x), source
            torch.tensor(y), end
            torch.tensor(t), edge_type
            torch.tensor(neigh)  edge_type_count, neighbor_samples
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
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_type_count = edge_type_count
        self.dim_a = dim_a
        self.features = None

        if features is not None:
            self.features = features
            feature_dim = self.features.shape[-1]
            self.embed_trans = Parameter(torch.FloatTensor(feature_dim, embedding_size))
            self.u_embed_trans = Parameter(torch.FloatTensor(edge_type_count, feature_dim, embedding_u_size))
        else:
            self.node_embeddings = Parameter(torch.FloatTensor(num_nodes, embedding_size))
            self.node_type_embeddings = Parameter(
                torch.FloatTensor(num_nodes, edge_type_count, embedding_u_size)
            )
        self.trans_weights = Parameter(
            torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size)
        )
        self.trans_weights_s1 = Parameter(
            torch.FloatTensor(edge_type_count, embedding_u_size, dim_a)
        )
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
            node_embed = self.node_embeddings[train_inputs] #batch_size, embedding_size
            node_embed_neighbors = self.node_type_embeddings[node_neigh]
            #batch_size, edge_type_count, neighbour_num, edge_type_count, embedding_u_size

        else:
            node_embed = torch.mm(self.features[train_inputs], self.embed_trans)
            #einsum 爱因斯坦求和表达https://blog.csdn.net/weixin_45101959/article/details/124483226
            # edge_type_count, feature_dim, embedding_u_size = akm
            # num_nodes, edge_type_count, neighbour_num, feature_dim = bijk
            node_embed_neighbors = torch.einsum('bijk,akm->bijam', self.features[node_neigh], self.u_embed_trans)
        # node_embed_tmp=(batch_size, edge_type_count, neighbour_num, embedding_u_size)
        node_embed_tmp = torch.diagonal(node_embed_neighbors, dim1=1, dim2=3).permute(0, 3, 1, 2)
        # node_embed_tmp=(batch_size, edge_type_count, embedding_u_size)
        node_type_embed = torch.sum(node_embed_tmp, dim=2)
        #trans_weights=(edge_type_count, embedding_u_size, embedding_size)
        #train_types = (batch_size)
        trans_w = self.trans_weights[train_types]
        #trans_w = (batch_size, embedding_u_size, embedding_size)
        #trans_weights_s1 = (edge_type_count, embedding_u_size, dim_a)
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
        ).unsqueeze(1) #batch_size, 1, edge_type_count
        #batch_size, edge_type_count, embedding_u_size ->  batch_size, 1, embedding_u_size
        node_type_embed = torch.matmul(attention, node_type_embed)
        #batch_size, embedding_size
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
        for i in range(num_nodes):
            train_inputs = torch.tensor([i for _ in range(edge_type_count)]).to(device)
            train_types = torch.tensor(list(range(edge_type_count))).to(device)
            node_neigh = torch.tensor(
                [neighbors[i] for _ in range(edge_type_count)]
            ).to(device)
            node_emb = model(train_inputs, train_types, node_neigh)
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
                    valid_true_data_by_edge[edge_types[i]],
                    valid_false_data_by_edge[edge_types[i]],
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
