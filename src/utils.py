import argparse
import multiprocessing
from collections import defaultdict
from operator import index

import numpy as np
from six import iteritems
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)
from tqdm import tqdm

from walk import RWGraph


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
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)) # x*y/|x||y|
    except Exception as e:
        pass


def evaluate(model, true_edges, false_edges):
    '''
    :param model: dict={node: embedding}
    :param true_edges: list=[(source_node, end_node)]
    :param false_edges: list=[(source_node, end_node)]
    :return:
    '''
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
    sorted_pred.sort() #理论上负样排前面
    threshold = sorted_pred[-true_num] # 按正样占比截取阈值

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)
