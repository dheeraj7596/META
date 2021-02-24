import numpy as np
from nltk import tokenize
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def get_entity_from_col(ent, mot_pat, config) -> set:
    separator = config[mot_pat]
    if separator is None:
        try:
            return set(ent)
        except:
            return {ent}
    else:
        return set(ent.split(separator))


def decrypt(val):
    try:
        if val[:5] == "fnust":
            return int(val[5:])
    except:
        return None


def detect_phrase(sentence, tokenizer, index_word, id_phrase_map, idx):
    tokens = tokenizer.texts_to_sequences([sentence])
    temp = []
    for tok in tokens[0]:
        try:
            id = decrypt(index_word[tok])
            if id == None or id not in id_phrase_map:
                if index_word[tok].startswith("fnust"):
                    num_str = index_word[tok][5:]
                    flag = 0
                    for index, char in enumerate(num_str):
                        if index >= 5:
                            break
                        try:
                            temp_int = int(char)
                            flag = 1
                        except:
                            break
                    if flag == 1:
                        if int(num_str[:index]) in id_phrase_map:
                            temp.append(index_word[tok])
                    else:
                        print(idx, index_word[tok])
            else:
                temp.append(index_word[tok])
        except Exception as e:
            pass
    return temp


def detect_non_phrase_seed_word(sentence, tokenizer, index_word, non_phrase_seeds):
    tokens = tokenizer.texts_to_sequences([sentence])
    temp = set()
    for tok in tokens[0]:
        temp.add(index_word[tok])
    return temp.intersection(non_phrase_seeds)


def detect_phrase_and_nonphrase_seeds(sentence, tokenizer, index_word, id_phrase_map, non_phrase_seeds, idx):
    phrase_set = set(detect_phrase(sentence, tokenizer, index_word, id_phrase_map, idx))
    non_phrase_seed_set = set(detect_non_phrase_seed_word(sentence, tokenizer, index_word, non_phrase_seeds))
    phrase_set.update(non_phrase_seed_set)
    return phrase_set


def create_index(tokenizer):
    index_to_word = {}
    word_to_index = tokenizer.word_index
    for word in word_to_index:
        index_to_word[word_to_index[word]] = word
    return word_to_index, index_to_word


def get_distinct_labels(df):
    label_to_index = {}
    index_to_label = {}
    labels = set(df["label"])

    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    return labels, label_to_index, index_to_label


def make_one_hot(y, label_to_index):
    labels = list(label_to_index.keys())
    n_classes = len(labels)
    y_new = []
    for label in y:
        current = np.zeros(n_classes)
        i = label_to_index[label]
        current[i] = 1.0
        y_new.append(current)
    y_new = np.asarray(y_new)
    return y_new


def prep_data(max_sentence_length, max_sentences, texts, tokenizer):
    data = np.zeros((len(texts), max_sentences, max_sentence_length), dtype='int32')
    documents = []
    for text in texts:
        sents = tokenize.sent_tokenize(text)
        documents.append(sents)
    for i, sentences in enumerate(documents):
        tokenized_sentences = tokenizer.texts_to_sequences(
            sentences
        )
        tokenized_sentences = pad_sequences(
            tokenized_sentences, maxlen=max_sentence_length
        )

        pad_size = max_sentences - tokenized_sentences.shape[0]

        if pad_size < 0:
            tokenized_sentences = tokenized_sentences[0:max_sentences]
        else:
            tokenized_sentences = np.pad(
                tokenized_sentences, ((0, pad_size), (0, 0)),
                mode='constant', constant_values=0
            )

        data[i] = tokenized_sentences[None, ...]
    return data


def create_train_dev(texts, labels, tokenizer, max_sentences=15, max_sentence_length=100, max_words=20000):
    data = prep_data(max_sentence_length, max_sentences, texts, tokenizer)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    return X_train, y_train, X_test, y_test


def get_from_one_hot(pred, index_to_label):
    pred_labels = np.argmax(pred, axis=-1)
    ans = []
    for l in pred_labels:
        ans.append(index_to_label[l])
    return ans


def print_label_phrase_dict(label_phrase_dict, id_phrase_map):
    for label in label_phrase_dict:
        print(label)
        print("*" * 80)
        print("Number of phrases: ", len(label_phrase_dict[label]))
        for key in label_phrase_dict[label]:
            id = decrypt(key)
            if id is None:
                print(key, label_phrase_dict[label][key])
            else:
                print(id_phrase_map[id], label_phrase_dict[label][key])


def print_label_motifs_dict(label_entity_dict):
    for label in label_entity_dict:
        print(label)
        print("*" * 80)
        print("Number of entities: ", len(label_entity_dict[label]))
        for key in label_entity_dict[label]:
            print(key, label_entity_dict[label][key])
