import pickle
import sys

if __name__ == "__main__":
    path = sys.argv[1]
    df = pickle.load(open(path + "df.pkl", "rb"))

    texts = list(df["text"])

    f = open("AutoPhrase/data/EN/text.txt", "w+")
    for i, r in enumerate(texts):
        f.write(r)
        f.write("\n")

    f.close()
