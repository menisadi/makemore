import csv
from tqdm import tqdm
import torch


def load_data(file_name: str) -> list[str]:
    with open(file_name, newline="") as f:
        reader = csv.reader(f)
        first_column = [row[0] for row in reader]

    return first_column[1:]


def setup_data(corpus: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    names = [n.lower() for n in corpus]

    letters = sorted(list(set("".join(names))))
    stoi = {s: i + 1 for i, s in enumerate(letters)}
    stoi["."] = 0
    names = ["." + w + "." for w in names]

    itos = {i: s for s, i in stoi.items()}

    return names, stoi, itos


def create_train(vocab: list[str], stoi: dict[str, int]) -> tuple[list[int], list[int]]:
    xs: list[int] = []
    ys: list[int] = []
    for w in tqdm(vocab):
        for i in range(len(w) - 1):
            ind1 = stoi[w[i]]
            ind2 = stoi[w[i + 1]]
            xs.append(ind1)
            ys.append(ind2)

    return xs, ys


def loss(ytrue: torch.Tensor, ypred: torch.Tensor):
    return -ypred[torch.arange(len(ytrue)), ytrue].log().mean()


def train(
    xs: list[int],
    ys: list[int],
    stoi: dict[str, int],
    learning_rate: float = 50,
    iterations: int = 1000,
):
    n = len(stoi)
    weights = torch.zeros(n, n, requires_grad=True)
    ts = torch.tensor(xs)
    ytrue = torch.tensor(ys)
    ohs = torch.nn.functional.one_hot(ts, n).float()

    current_loss = None
    for _ in tqdm(range(iterations)):
        ylogits = ohs @ weights
        ycounts = ylogits.exp()
        ypred = ycounts / ycounts.sum(dim=1, keepdim=True)
        current_loss = loss(ytrue, ypred)

        current_loss.backward()
        weights.data -= weights.grad * learning_rate
        weights.grad = None

    return weights, current_loss


if __name__ == "__main__":
    raw_data = load_data("female_names.csv")
    names, stoi, itos = setup_data(raw_data)
    xs, ys = create_train(names, stoi)

    weights, final_loss = train(xs, ys, stoi)
    if final_loss is not None:
        print(final_loss.item())
