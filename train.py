import sys
import pickle
import json

if __name__ == "__main__":
    data_path = sys.argv[1]
    tmp_path = sys.argv[2]
    use_gpu = int(sys.argv[3])
    gpu_id = int(sys.argv[4])

    model_name = "meta"
    df = pickle.load(open(tmp_path + "df_phrase_removed_stopwords.pkl", "rb"))
    graph_dict = pickle.load(open(tmp_path + "graph_dict.pkl", "rb"))
    entity_dict = pickle.load(open(tmp_path + "entity_id_dict.pkl", "rb"))
    id_entity_dict = pickle.load(open(tmp_path + "id_entity_dict.pkl", "rb"))
    entity_docid_dict = pickle.load(open(tmp_path + "entity_docid_dict.pkl", "rb"))
    label_term_dict = json.load(open(tmp_path + "seedwords_fnust.json", "r"))

