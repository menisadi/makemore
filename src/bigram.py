import csv
from tqdm import tqdm


def load_data(file_name: str) -> list[str]:
    with open(file_name, newline="") as f:
        reader = csv.reader(f)
        first_column = [row[0] for row in reader]

    return first_column[1:]


def setup_vocab(corpus: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    names = [n.lower() for n in corpus]

    letters = sorted(list(set("".join(names))))
    stoi = {s: i + 1 for i, s in enumerate(letters)}
    stoi["."] = 0
    names = ["." + w + "." for w in names]

    itos = {i: s for s, i in stoi.items()}

    return names, stoi, itos


def create_train(vocab: list[str], stoi: dict[str, int]) -> list[tuple[int, int]]:
    bigrams: list[tuple[int, int]] = []
    for w in tqdm(vocab):
        for i in range(len(w) - 1):
            ind1 = stoi[w[i]]
            ind2 = stoi[w[i + 1]]
            bigrams.append((ind1, ind2))

    return bigrams


def loss(ytrue, ypred):
    return
