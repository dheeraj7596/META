import sys
import itertools
import pickle
import json
import time
import random
from keras_han.model import HAN
from util.utils import *
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from fast_pagerank import pagerank
import numpy as np
import tensorflow as tf
import os


def train_word2vec(df, tokenizer, word_index, index_word):
    def get_target(words, idx, window_size=5):
        ''' Get a list of words in a window around an index. '''

        R = np.random.randint(1, window_size + 1)
        start = idx - R if (idx - R) > 0 else 0
        stop = idx + R
        target_words = set(words[start:idx] + words[idx + 1:stop + 1])
        return list(target_words)

    def get_idx_pairs(df, tokenizer):
        x = []
        y = []
        for i, row in df.iterrows():
            tokenized_text_words = tokenizer.texts_to_sequences([row["text"]])[0]
            for i, word in enumerate(tokenized_text_words):
                x.append(word)
                target_words = get_target(tokenized_text_words, i)
                y.append(target_words)
        return x, y

    def get_batches(x, y, batch_size):
        ''' Create a generator of word batches as a tuple (inputs, targets) '''
        n_batches = len(x) // batch_size

        # only full batches
        words = x[:n_batches * batch_size]

        for idx in range(0, len(words), batch_size):
            curr_words, context_words = [], []
            batch_x = words[idx:idx + batch_size]
            batch_y = y[idx:idx + batch_size]

            for ii in range(len(batch_x)):
                context_words.extend(batch_y[ii])
                curr_words.extend([batch_x[ii]] * len(batch_y[ii]))
            yield curr_words, context_words

    vocabulary = list(word_index.keys())
    int_to_vocab = index_word

    print("Size of vocabulary: ", len(vocabulary), flush=True)

    current_words, context_words = get_idx_pairs(df, tokenizer)

    # Graph
    train_graph = tf.Graph()
    with train_graph.as_default():
        inputs = tf.placeholder(tf.int32, [None], name='inputs')
        #     labels = tf.placeholder(tf.int32, [None, None], name='labels')
        labels = tf.placeholder(tf.int32, [None, None], name='labels')

    n_vocab = len(int_to_vocab)
    n_embedding = 100
    with train_graph.as_default():
        embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs)  # use tf.nn.embedding_lookup to get the hidden layer output

    # Number of negative labels to sample
    n_sampled = 100
    with train_graph.as_default():
        softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding)))  # create softmax weight matrix here
        softmax_b = tf.Variable(tf.zeros(n_vocab), name="softmax_bias")  # create softmax biases here

        # Calculate the loss using negative sampling
        loss = tf.nn.sampled_softmax_loss(
            weights=softmax_w,
            biases=softmax_b,
            labels=labels,
            inputs=embed,
            num_sampled=n_sampled,
            num_classes=n_vocab)

        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    with train_graph.as_default():
        ## From Thushan Ganegedara's implementation
        valid_size = 16  # Random set of words to evaluate similarity on.
        valid_window = 100
        # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent
        valid_examples = np.array(random.sample(range(1, valid_window), valid_size // 2))
        valid_examples = np.append(valid_examples,
                                   random.sample(range(1000, 1000 + valid_window), valid_size // 2))

        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
        normalized_embedding = embedding / norm
        valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
        similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

    epochs = 100
    batch_size = 1000

    with tf.Session(graph=train_graph) as sess:
        iteration = 1
        loss = 0
        sess.run(tf.global_variables_initializer())

        for e in range(1, epochs + 1):
            batches = get_batches(current_words, context_words, batch_size)
            start = time.time()
            for x, y in batches:

                feed = {inputs: x,
                        labels: np.array(y)[:, None]}
                train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

                loss += train_loss

                if iteration % 100 == 0:
                    end = time.time()
                    print("Epoch {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Avg. Training loss: {:.4f}".format(loss / 100),
                          "{:.4f} sec/batch".format((end - start) / 100), flush=True)
                    loss = 0
                    start = time.time()

                if iteration % 1000 == 0:
                    ## From Thushan Ganegedara's implementation
                    # note that this is expensive (~20% slowdown if computed every 500 steps)
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = int_to_vocab[valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                            close_word = int_to_vocab[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        print(log, flush=True)

                iteration += 1
        embed_mat = sess.run(normalized_embedding)

    return embed_mat


def train_classifier(df, tokenizer, embedding_matrix, labels, motpat_label_motifs_dict, label_to_index,
                     index_to_label, index_word, dataset_path, config):
    def generate_pseudo_labels(df, labels, motpat_label_motifs_dict, tokenizer, index_word, config):
        y = []
        X = []

        for index, row in df.iterrows():
            count_dict = {}
            flag = 0
            for mot_pat in motpat_label_motifs_dict:
                label_motifs_dict = motpat_label_motifs_dict[mot_pat]
                if len(label_motifs_dict) < 0:
                    continue
                if mot_pat == "phrase":
                    tokens = tokenizer.texts_to_sequences([row["text"]])[0]
                    words = []
                    for tok in tokens:
                        words.append(index_word[tok])
                    for l in labels:
                        seed_words = set(label_motifs_dict[l].keys())
                        int_words = list(set(words).intersection(seed_words))
                        for word in int_words:
                            flag = 1
                            try:
                                count_dict[l] += label_motifs_dict[l][word]
                            except:
                                count_dict[l] = label_motifs_dict[l][word]
                else:
                    size = len(mot_pat)
                    if size == 1:
                        first = mot_pat[0]
                        entities = get_entity_from_col(row[first], first, config)
                    elif size == 2:
                        first = mot_pat[0]
                        second = mot_pat[1]
                        first_ents = get_entity_from_col(row[first], first, config)
                        second_ents = get_entity_from_col(row[second], second, config)
                        entities = set(itertools.product(first_ents, second_ents))
                    else:
                        raise Exception(
                            "Motif patterns of size more than 2 not yet handled but can be easily extended.")
                    for l in labels:
                        seed_entities = set(label_motifs_dict[l].keys())
                        int_ents = list(entities.intersection(seed_entities))
                        for ent in int_ents:
                            flag = 1
                            try:
                                count_dict[l] += label_motifs_dict[l][ent]
                            except:
                                count_dict[l] = label_motifs_dict[l][ent]

            if flag:
                lbl = max(count_dict, key=count_dict.get)
                if not lbl:
                    continue
                y.append(lbl)
                X.append(row["text"])
        return X, y

    basepath = dataset_path
    model_name = "meta"
    dump_dir = basepath + "models/" + model_name + "/"
    tmp_dir = basepath + "checkpoints/" + model_name + "/"
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    max_sentence_length = 100
    max_sentences = 15
    max_words = 20000

    print("Generating pseudo-labels", flush=True)
    X, y = generate_pseudo_labels(df, labels, motpat_label_motifs_dict, tokenizer, index_word, config)
    y_vec = make_one_hot(y, label_to_index)

    print("Splitting into train, dev...", flush=True)
    X_train, y_train, X_val, y_val = create_train_dev(X, labels=y_vec, tokenizer=tokenizer,
                                                      max_sentences=max_sentences,
                                                      max_sentence_length=max_sentence_length,
                                                      max_words=max_words)

    print("Initializing model...", flush=True)
    model = HAN(max_words=max_sentence_length, max_sentences=max_sentences, output_size=len(y_train[0]),
                embedding_matrix=embedding_matrix)
    print("Compiling model...", flush=True)
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    print("model fitting - Hierachical attention network...", flush=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint(filepath=tmp_dir + 'model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', mode='max',
                         verbose=1, save_weights_only=True, save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=100, batch_size=256, callbacks=[es, mc])
    print("****************** CLASSIFICATION REPORT FOR All DOCUMENTS ********************", flush=True)
    X_all = prep_data(texts=df["text"], max_sentences=max_sentences, max_sentence_length=max_sentence_length,
                      tokenizer=tokenizer)
    y_true_all = df["label"]
    pred = model.predict(X_all)
    pred_labels = get_from_one_hot(pred, index_to_label)
    print(classification_report(y_true_all, pred_labels), flush=True)
    print("Dumping the model...", flush=True)
    model.save_weights(dump_dir + "model_weights_" + model_name + ".h5")
    model.save(dump_dir + "model_" + model_name + ".h5")
    return pred_labels, pred


def expand_motifs(df, probs, labels, motpat_label_motifs_dict, graph_dict, entity_node_id_dict, node_id_entity_dict,
                  entity_docid_dict, label_to_index):
    def rank(probs, df, G, entity_node_id, node_id_entity, label_to_index):
        def get_scaling_factor(key, label_entity_dict):
            total_sum = 0
            for l in label_entity_dict:
                total_sum += label_entity_dict[l][key]
            return total_sum

        def scale(label_entity_dict):
            scaling_factor = {}
            for l in label_entity_dict:
                for key in label_entity_dict[l]:
                    try:
                        factor = scaling_factor[key]
                    except:
                        factor = get_scaling_factor(key, label_entity_dict)
                        scaling_factor[key] = factor
                    label_entity_dict[l][key] = label_entity_dict[l][key] / factor

            for l in label_entity_dict:
                label_entity_dict[l] = {k: v for k, v in
                                        sorted(label_entity_dict[l].items(), key=lambda item: -item[1])}
            return label_entity_dict

        label_entity_dict = {}
        start = len(df)
        count = len(df) + len(entity_node_id)
        for l in label_to_index:
            print("Pagerank running for: ", l, flush=True)
            personalized = np.zeros((count,))
            personalized[:len(df)] = probs[:, label_to_index[l]]
            pr = pagerank(G, p=0.85, personalize=personalized)
            temp_list = list(pr)[start:]
            args = np.argsort(temp_list)[::-1]
            top_ents = {}
            for i in args:
                top_ents[node_id_entity[start + i]] = temp_list[i]
            label_entity_dict[l] = top_ents
        label_entity_dict = scale(label_entity_dict)
        return label_entity_dict

    def unified_filtering(motpat_label_motifs_dict, entity_docid_dict, df, labels):
        filtered_motpat_label_motifs_dict = {}
        thresh = 1 / len(labels)
        for motpat in motpat_label_motifs_dict:
            filtered_dict = {}
            for l in motpat_label_motifs_dict[motpat]:
                filtered_dict[l] = {}
            filtered_motpat_label_motifs_dict[motpat] = filtered_dict

        sorted_tups_dict = {}
        for l in labels:
            all_tups = []
            for motpat in motpat_label_motifs_dict:
                label_motifs_dict = motpat_label_motifs_dict[motpat]
                all_tups += list(label_motifs_dict[l].items())
            sorted_tups_dict[l] = list(filter(lambda a: a[1] > thresh, sorted(all_tups, key=lambda tup: -tup[1])))

        visited_motifs_dict = {}
        for motpat in entity_docid_dict:
            visited_motifs_dict[motpat] = {}

        flagged = {}
        doc_id_set = set()
        index = 0
        while len(doc_id_set) < len(df):
            if len(flagged) == len(entity_docid_dict):
                break
            flag = 0
            for l in labels:
                if index < len(sorted_tups_dict[l]):
                    flag = 1
                    tup = sorted_tups_dict[l][index]
                    for motpat in entity_docid_dict:
                        if motpat in flagged:
                            continue
                        try:
                            temp = visited_motifs_dict[motpat][tup[0]]
                            flagged[motpat] = 1
                            continue
                        except:
                            pass
                        try:
                            entity_docid = entity_docid_dict[motpat]
                            temp = entity_docid[tup[0]]
                            filtered_motpat_label_motifs_dict[motpat][l][tup[0]] = tup[1]
                            doc_id_set.update(entity_docid[tup[0]])
                            visited_motifs_dict[motpat][tup[0]] = 1
                        except:
                            continue
            if flag == 0:
                break
            index += 1
        return filtered_motpat_label_motifs_dict

    for motpat in motpat_label_motifs_dict:
        G = graph_dict[motpat]
        entity_node_id = entity_node_id_dict[motpat]
        node_id_entity = node_id_entity_dict[motpat]
        motpat_label_motifs_dict[motpat] = rank(probs, df, G, entity_node_id, node_id_entity, label_to_index)

    expanded_motpat_label_motifs_dict = unified_filtering(motpat_label_motifs_dict, entity_docid_dict, df, labels)
    return expanded_motpat_label_motifs_dict


def main(data_path, tmp_path, print_flag=True):
    config = json.load(open(data_path + "metadata_config.json", "r"))
    df = pickle.load(open(tmp_path + "df_phrase_removed_stopwords.pkl", "rb"))
    graph_dict = pickle.load(open(tmp_path + "graph_dict.pkl", "rb"))
    entity_node_id_dict = pickle.load(open(tmp_path + "entity_node_id_dict.pkl", "rb"))
    node_id_entity_dict = pickle.load(open(tmp_path + "node_id_entity_dict.pkl", "rb"))
    entity_docid_dict = pickle.load(open(tmp_path + "entity_docid_dict.pkl", "rb"))
    label_term_dict = json.load(open(tmp_path + "seedwords_fnust.json", "r"))
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    id_phrase_map = pickle.load(open(tmp_path + "id_phrase_map.pkl", "rb"))

    word_to_index, index_to_word = create_index(tokenizer)
    labels, label_to_index, index_to_label = get_distinct_labels(df)

    try:
        embedding_matrix = pickle.load(open(data_path + "embedding_matrix.pkl", "rb"))
        print("Embedding matrix available.. Loading embedding matrix..", flush=True)
    except:
        print("Training Word2Vec to get embedding matrix..", flush=True)
        embedding_matrix = train_word2vec(df, tokenizer, word_to_index, index_to_word)
        pickle.dump(embedding_matrix, open(data_path + "embedding_matrix.pkl", "wb"))

    motpat_label_motifs_dict = {}
    for mot_pat in entity_node_id_dict:
        if mot_pat == "phrase":
            motpat_label_motifs_dict[mot_pat] = label_term_dict
        else:
            motpat_label_motifs_dict[mot_pat] = {}

    for i in range(9):
        print("ITERATION: ", i, flush=True)
        pred_labels, pred_probs = train_classifier(df, tokenizer, embedding_matrix, labels, motpat_label_motifs_dict,
                                                   label_to_index, index_to_label, index_to_word, data_path, config)
        motpat_label_motifs_dict = expand_motifs(df, pred_probs, labels, motpat_label_motifs_dict, graph_dict,
                                                 entity_node_id_dict, node_id_entity_dict, entity_docid_dict,
                                                 label_to_index)
        if print_flag:
            for mot_pat in motpat_label_motifs_dict:
                print("Printing entities of motif pattern:", mot_pat, flush=True)
                label_motifs_dict = motpat_label_motifs_dict[mot_pat]
                if mot_pat == "phrase":
                    print_label_phrase_dict(label_motifs_dict, id_phrase_map)
                else:
                    print_label_motifs_dict(label_motifs_dict)
        print("#" * 80, flush=True)


if __name__ == "__main__":
    data_path = sys.argv[1]
    tmp_path = sys.argv[2]
    use_gpu = int(sys.argv[3])
    gpu_id = int(sys.argv[4])

    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        from keras.backend.tensorflow_backend import set_session

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        sess = tf.compat.v1.Session(config=config)
        set_session(sess)

    main(
        data_path=data_path,
        tmp_path=tmp_path
    )
