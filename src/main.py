import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
from numpy import random
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


# from utils import *
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
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
            neigh.append(neighbors[pairs[index][0]])
        yield (np.array(x).astype(np.int32), np.array(y).reshape(-1, 1).astype(np.int32), np.array(t).astype(np.int32), np.array(neigh).astype(np.int32)) 

def train_model(network_data, feature_dic, log_name):
    vocab, index2word, train_pairs = generate(network_data, args.num_walks, args.walk_length, args.schema, file_name, args.window_size, args.num_workers, args.walk_file)

    edge_types = list(network_data.keys())

    num_nodes = len(index2word)
    edge_type_count = len(edge_types)
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions # Dimension of the embedding vector.
    embedding_u_size = args.edge_dim
    u_num = edge_type_count
    num_sampled = args.negative_samples # Number of negative examples to sample.
    dim_a = args.att_dim
    att_head = 1
    neighbor_samples = args.neighbor_samples 

    neighbors = generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples)

    graph = tf.Graph()

    if feature_dic is not None:
        feature_dim = len(list(feature_dic.values())[0])
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():
            if key in vocab:
                features[vocab[key].index, :] = np.array(value)
        
    with graph.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if feature_dic is not None:
            node_features = tf.Variable(features, name='node_features', trainable=False)
            feature_weights = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0))

            embed_trans = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            u_embed_trans = tf.Variable(tf.truncated_normal([edge_type_count, feature_dim, embedding_u_size], stddev=1.0 / math.sqrt(embedding_size)))
        else:
            node_embeddings = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))
            node_type_embeddings = tf.Variable(tf.random_uniform([num_nodes, u_num, embedding_u_size], -1.0, 1.0))

        trans_weights = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, embedding_size // att_head], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s1 = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, dim_a], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s2 = tf.Variable(tf.truncated_normal([edge_type_count, dim_a, att_head], stddev=1.0 / math.sqrt(embedding_size)))
        nce_weights = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([num_nodes]))

        # Input data
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        train_types = tf.placeholder(tf.int32, shape=[None])
        node_neigh = tf.placeholder(tf.int32, shape=[None, edge_type_count, neighbor_samples])
        
        # Look up embeddings for nodes
        if feature_dic is not None:
            node_embed = tf.nn.embedding_lookup(node_features, train_inputs)
            node_embed = tf.matmul(node_embed, embed_trans)
        else:
            node_embed = tf.nn.embedding_lookup(node_embeddings, train_inputs)
        
        if feature_dic is not None:
            node_embed_neighbors = tf.nn.embedding_lookup(node_features, node_neigh)
            node_embed_tmp = tf.concat([tf.matmul(tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, 0], [-1, 1, -1, -1]), [-1, feature_dim]), tf.reshape(tf.slice(u_embed_trans, [i, 0, 0], [1, -1, -1]), [feature_dim, embedding_u_size])) for i in range(edge_type_count)], axis=0)
            node_type_embed = tf.transpose(tf.reduce_mean(tf.reshape(node_embed_tmp, [edge_type_count, -1, neighbor_samples, embedding_u_size]), axis=2), perm=[1,0,2])
        else:
            node_embed_neighbors = tf.nn.embedding_lookup(node_type_embeddings, node_neigh)
            node_embed_tmp = tf.concat([tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, i, 0], [-1, 1, -1, 1, -1]), [1, -1, neighbor_samples, embedding_u_size]) for i in range(edge_type_count)], axis=0)
            node_type_embed = tf.transpose(tf.reduce_mean(node_embed_tmp, axis=2), perm=[1,0,2])

        trans_w = tf.nn.embedding_lookup(trans_weights, train_types)
        trans_w_s1 = tf.nn.embedding_lookup(trans_weights_s1, train_types)
        trans_w_s2 = tf.nn.embedding_lookup(trans_weights_s2, train_types)
        
        attention = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(tf.matmul(node_type_embed, trans_w_s1)), trans_w_s2), [-1, u_num])), [-1, att_head, u_num])
        node_type_embed = tf.matmul(attention, node_type_embed)
        node_embed = node_embed + tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size])

        if feature_dic is not None:
            node_feat = tf.nn.embedding_lookup(node_features, train_inputs)
            node_embed = node_embed + tf.matmul(node_feat, feature_weights)

        last_node_embed = tf.nn.l2_normalize(node_embed, axis=1)

        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=last_node_embed,
                num_sampled=num_sampled,
                num_classes=num_nodes))
        plot_loss = tf.summary.scalar("loss", loss)

        # Optimizer.
        optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        # Add ops to save and restore all the variables.
        # saver = tf.train.Saver(max_to_keep=20)

        merged = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        # Initializing the variables
        init = tf.global_variables_initializer()

    # Launch the graph
    print("Optimizing")
    
    with tf.Session(graph=graph) as sess:
        writer = tf.summary.FileWriter("./runs/" + log_name, sess.graph) # tensorboard --logdir=./runs
        sess.run(init)

        print('Training')
        g_iter = 0
        best_score = 0
        test_score = (0.0, 0.0, 0.0)
        patience = 0
        for epoch in range(epochs):
            random.shuffle(train_pairs)
            batches = get_batches(train_pairs, neighbors, batch_size)

            data_iter = tqdm(batches,
                            desc="epoch %d" % (epoch),
                            total=(len(train_pairs) + (batch_size - 1)) // batch_size,
                            bar_format="{l_bar}{r_bar}")
            avg_loss = 0.0

            for i, data in enumerate(data_iter):
                feed_dict = {train_inputs: data[0], train_labels: data[1], train_types: data[2], node_neigh: data[3]}
                _, loss_value, summary_str = sess.run([optimizer, loss, merged], feed_dict)
                writer.add_summary(summary_str, g_iter)

                g_iter += 1

                avg_loss += loss_value

                if i % 5000 == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss_value
                    }
                    data_iter.write(str(post_fix))
            
            final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
            for i in range(edge_type_count):
                for j in range(num_nodes):
                    final_model[edge_types[i]][index2word[j]] = np.array(sess.run(last_node_embed, {train_inputs: [j], train_types: [i], node_neigh: [neighbors[j]]})[0])
            valid_aucs, valid_f1s, valid_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if args.eval_type == 'all' or edge_types[i] in args.eval_type.split(','):
                    tmp_auc, tmp_f1, tmp_pr = evaluate(final_model[edge_types[i]], valid_true_data_by_edge[edge_types[i]], valid_false_data_by_edge[edge_types[i]])
                    valid_aucs.append(tmp_auc)
                    valid_f1s.append(tmp_f1)
                    valid_prs.append(tmp_pr)

                    tmp_auc, tmp_f1, tmp_pr = evaluate(final_model[edge_types[i]], testing_true_data_by_edge[edge_types[i]], testing_false_data_by_edge[edge_types[i]])
                    test_aucs.append(tmp_auc)
                    test_f1s.append(tmp_f1)
                    test_prs.append(tmp_pr)
            print('valid auc:', np.mean(valid_aucs))
            print('valid pr:', np.mean(valid_prs))
            print('valid f1:', np.mean(valid_f1s))

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
                    print('Early Stopping')
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

    log_name = file_name.split('/')[-1] + f'_evaltype_{args.eval_type}_b_{args.batch_size}_e_{args.epoch}'

    training_data_by_type = load_training_data(file_name + '/train.txt')
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/valid.txt')
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test.txt')

    average_auc, average_f1, average_pr = train_model(training_data_by_type, feature_dic, log_name + '_' + time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time())))

    print('Overall ROC-AUC:', average_auc)
    print('Overall PR-AUC', average_pr)
    print('Overall F1:', average_f1)
