import csv


def load_data(file_name: str):
    with open(file_name, newline="") as f:
        reader = csv.reader(f)
        first_column = [row[0] for row in reader]

    return first_column[1:]


def bigram_counts():
    pass


def loss():
    pass
