from bs4 import BeautifulSoup
import bleach
import pickle
import sys


def generate_name(id):
    return "fnust" + str(id)


def decrypt(val):
    try:
        if val[:5] == "fnust":
            return int(val[5:])
    except:
        return None


if __name__ == "__main__":
    tmp_path = sys.argv[1]
    data_path = sys.argv[2]
    out_path = tmp_path + "segmentation.txt"
    df = pickle.load(open(data_path + "df.pkl", "rb"))
    f = open(out_path, "r")
    lines = f.readlines()
    f.close()

    phrase_id_map = {}
    counter = 0
    data = []

    for line in lines:
        line = line.lower()
        soup = BeautifulSoup(line)
        for p in soup.findAll("phrase"):
            phrase = p.string
            if phrase is None:
                continue
            try:
                temp = phrase_id_map[phrase]
            except:
                phrase_id_map[phrase] = counter
                counter += 1
            name = generate_name(phrase_id_map[phrase])
            p.string.replace_with(" " + name + " ")
        temp_str = bleach.clean(str(soup), tags=[], strip=True)
        data.append(temp_str)

    df["text"] = data
    pickle.dump(df, open(tmp_path + "df_phrase.pkl", "wb"))

    id_phrase_map = {}
    for ph in phrase_id_map:
        id_phrase_map[phrase_id_map[ph]] = ph

    print("Number of phrases: ", len(phrase_id_map))
    pickle.dump(phrase_id_map, open(tmp_path + "phrase_id_map.pkl", "wb"))
    pickle.dump(id_phrase_map, open(tmp_path + "id_phrase_map.pkl", "wb"))
