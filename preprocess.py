import sys
from util.utils import get_entity_from_col, detect_phrase, detect_phrase_and_nonphrase_seeds
from util.parse_autophrase_output import generate_name
from scipy import sparse
import itertools
import pickle
import numpy as np
from nltk.corpus import stopwords
import json
from keras.preprocessing.text import Tokenizer


def fit_tokenizer(df):
    corpus = []
    for i, row in df.iterrows():
        corpus.append(row["text"])
    tokenizer = Tokenizer(num_words=1000000)
    tokenizer.fit_on_texts(corpus)
    return tokenizer


def modify_phrases(label_term_dict, phrase_id_map):
    non_phrase_seeds = []
    for l in label_term_dict:
        temp_list = []
        for term in label_term_dict[l]:
            try:
                temp_list.append(generate_name(phrase_id_map[term]))
            except:
                temp_list.append(term)
                non_phrase_seeds.append(term)
        label_term_dict[l] = temp_list
    return label_term_dict, non_phrase_seeds


def make_phrases_map(df, tokenizer, index_word, id_phrase_map, non_phrase_seeds):
    count = len(df)
    texts = list(df.text)
    fnust_id = {}
    id_fnust = {}

    for i, sent in enumerate(texts):
        phrases = detect_phrase(sent, tokenizer, index_word, id_phrase_map, i)
        for ph in phrases:
            try:
                temp = fnust_id[ph]
            except:
                fnust_id[ph] = count
                id_fnust[count] = ph
                count += 1

    for seed in non_phrase_seeds:
        try:
            temp = fnust_id[seed]
        except:
            fnust_id[seed] = count
            id_fnust[count] = seed
            count += 1

    return fnust_id, id_fnust, count


def create_phrase_doc_id_map(df, tokenizer, index_word, id_phrase_map, non_phrase_seeds):
    phrase_docid = {}

    for i, row in df.iterrows():
        text = row["text"]
        phrases = detect_phrase_and_nonphrase_seeds(text, tokenizer, index_word, id_phrase_map, non_phrase_seeds, i)
        for ph in phrases:
            try:
                phrase_docid[ph].add(i)
            except:
                phrase_docid[ph] = {i}
    return phrase_docid


def remove_stop_words(df):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    texts = list(df["text"])

    clean_texts = []
    for abs in texts:
        word_list = abs.strip().split()
        filtered_words = [word for word in word_list if word not in stop_words]
        temp = " ".join(filtered_words)
        clean_texts.append(temp)

    df["text"] = clean_texts
    return df


def get_motif_patterns(df, motif_lines):
    cols = set(df.columns)
    motif_patterns = []
    meta_cols = set([])
    for line in motif_lines:
        motif = line.strip().split(",")
        if len(set(motif).intersection(cols)) != len(motif):
            raise Exception("Unknown column in motif found ", motif)
        motif_patterns.append(tuple(motif))
        meta_cols.update(motif)
    return meta_cols, motif_patterns


def verify_config(df, config):
    cols = set(df.columns)
    if len(set(config.keys()).intersection(cols)) != len(set(config.keys())):
        raise Exception("Unknown column in config found")


def create_dicts(df, motif_patterns, config):
    entity_node_id_dict = {}
    node_id_entity_dict = {}
    node_count_dict = {}
    entity_docid_dict = {}

    for mot_pat in motif_patterns:
        length = len(mot_pat)
        count = len(df)
        entity_id = {}
        id_entity = {}
        entity_docid = {}

        entity_set = set()
        if length == 1:
            mot_pat = mot_pat[0]
            for i, row in df.iterrows():
                ent = get_entity_from_col(row[mot_pat], mot_pat, config)
                entity_set.update(ent)
                for e in ent:
                    try:
                        entity_docid[e].add(i)
                    except:
                        entity_docid[e] = {i}
        elif length == 2:
            first = mot_pat[0]
            second = mot_pat[1]
            for i, row in df.iterrows():
                first_ents = get_entity_from_col(row[first], first, config)
                second_ents = get_entity_from_col(row[second], second, config)
                temp_ents = set(itertools.product(first_ents, second_ents))
                entity_set.update(temp_ents)
                for temp_ent in temp_ents:
                    try:
                        entity_docid[temp_ent].add(i)
                    except:
                        entity_docid[temp_ent] = {i}
        else:
            raise Exception(
                "Currently only motif patterns of size upto 2 are supported. The code can be easily extended to multiple ones.")

        for i, ent in enumerate(entity_set):
            entity_id[ent] = count
            id_entity[count] = ent
            count += 1

        entity_node_id_dict[mot_pat] = entity_id
        node_id_entity_dict[mot_pat] = id_entity
        node_count_dict[mot_pat] = count
        entity_docid_dict[mot_pat] = entity_docid

    return entity_node_id_dict, node_id_entity_dict, node_count_dict, entity_docid_dict


def create_graphs(df, motif_patterns, entity_id_dict, node_count_dict, config):
    graph_dict = {}
    for mot_pat in motif_patterns:
        entity_id = entity_id_dict[mot_pat]
        node_count = node_count_dict[mot_pat]
        edges = []
        weights = []
        length = len(mot_pat)
        if length == 1:
            mot_pat = mot_pat[0]
            for i, row in df.iterrows():
                ent = get_entity_from_col(row[mot_pat], mot_pat, config)
                for e in ent:
                    edges.append([i, entity_id[e]])
                    weights.append(1)
        elif length == 2:
            first = mot_pat[0]
            second = mot_pat[1]
            for i, row in df.iterrows():
                first_ents = get_entity_from_col(row[first], first, config)
                second_ents = get_entity_from_col(row[second], second, config)
                temp_ents = set(itertools.product(first_ents, second_ents))
                for e in temp_ents:
                    edges.append([i, entity_id[e]])
                    weights.append(1)
        else:
            raise Exception(
                "Currently motif patterns of size <= 2 are supported. The code can be easily extended to multiple ones.")

        edges = np.array(edges)
        G = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(node_count, node_count))
        graph_dict[mot_pat] = G
    return graph_dict


def create_phrase_graph(df, tokenizer, index_word, id_phrase_map, non_phrase_seeds, fnust_id, fnust_graph_node_count):
    edges = []
    weights = []
    for i, row in df.iterrows():
        text = row["text"]
        phrases = detect_phrase_and_nonphrase_seeds(text, tokenizer, index_word, id_phrase_map, non_phrase_seeds, i)
        for ph in phrases:
            edges.append([i, fnust_id[ph]])
            weights.append(1)
    edges = np.array(edges)
    G_phrase = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                                 shape=(fnust_graph_node_count, fnust_graph_node_count))
    return G_phrase


if __name__ == "__main__":
    data_path = sys.argv[1]
    tmp_path = sys.argv[2]

    df = pickle.load(open(tmp_path + "df_phrase.pkl", "wb"))

    f = open(data_path + "motif_patterns.txt", "r")
    motif_lines = f.readlines()
    f.close()

    label_term_dict = json.load(open(data_path + "seedwords.json", "r"))
    phrase_id_map = pickle.load(open(tmp_path + "phrase_id_map.pkl", "rb"))
    id_phrase_map = pickle.load(open(tmp_path + "id_phrase_map.pkl", "rb"))
    label_term_dict, non_phrase_seeds = modify_phrases(label_term_dict, phrase_id_map)

    df = remove_stop_words(df)
    tokenizer = fit_tokenizer(df)
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w

    config = json.load(open(data_path + "metadata_config.json", "r"))
    meta_cols, motif_patterns = get_motif_patterns(df, motif_lines)
    verify_config(df, config)

    print("Creating Dictionaries..")
    entity_node_id_dict, node_id_entity_dict, node_count_dict, entity_docid_dict = create_dicts(df, motif_patterns,
                                                                                                config)
    fnust_id, id_fnust, fnust_graph_node_count = make_phrases_map(df, tokenizer, index_word, id_phrase_map,
                                                                  non_phrase_seeds)
    phrase_doc_id_map = create_phrase_doc_id_map(df, tokenizer, index_word, id_phrase_map, non_phrase_seeds)

    entity_node_id_dict["phrase"] = fnust_id
    node_id_entity_dict["phrase"] = id_fnust
    node_count_dict["phrase"] = fnust_graph_node_count
    entity_docid_dict["phrase"] = phrase_doc_id_map

    print("Creating Graphs..")
    graph_dict = create_graphs(df, motif_patterns, entity_node_id_dict, node_id_entity_dict, node_count_dict)

    phrase_graph = create_phrase_graph(df, tokenizer, index_word, id_phrase_map, non_phrase_seeds, fnust_id,
                                       fnust_graph_node_count)
    graph_dict["phrase"] = phrase_graph

    json.dump(label_term_dict, open(tmp_path + "seedwords_fnust.json", "w"))
    pickle.dump(tokenizer, open(data_path + "tokenizer.pkl", "wb"))
    pickle.dump(df, open(tmp_path + "df_phrase_removed_stopwords.pkl", "wb"))
    pickle.dump(graph_dict, open(tmp_path + "graph_dict.pkl", "wb"))
    pickle.dump(entity_node_id_dict, open(tmp_path + "entity_node_id_dict.pkl", "wb"))
    pickle.dump(node_id_entity_dict, open(tmp_path + "node_id_entity_dict.pkl", "wb"))
    pickle.dump(entity_docid_dict, open(tmp_path + "entity_docid_dict.pkl", "wb"))
