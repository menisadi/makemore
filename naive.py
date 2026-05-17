import torch
import csv
from tqdm import tqdm

with open("names.csv", newline="") as f:
    reader = csv.reader(f)
    first_column = [row[0] for row in reader]

names = first_column[1:]

letters = sorted(list(set("".join(names))))
stoi = {s: i + 1 for i, s in enumerate(letters)}
stoi["."] = 0
names = ["." + w + "." for w in names]

itos = {i: s for s, i in stoi.items()}

N = torch.zeros((31, 31))

for w in tqdm(names):
    for i in range(len(w) - 1):
        ind1 = stoi[w[i]]
        ind2 = stoi[w[i + 1]]
        N[ind1, ind2] += 1
P = N / N.sum(1, keepdim=True)

gen: list[str] = []
c = "."
while True:
    gen.append(c)
    ind = stoi[c]
    res: int = int(torch.multinomial(P[ind], 1, replacement=True).item())
    c = itos[res]
    if c == ".":
        break

final_name = "".join(gen[1:])

print(final_name)
