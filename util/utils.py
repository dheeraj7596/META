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
