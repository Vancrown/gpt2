from __future__ import annotations

import re


def get_vocab():
    with open("./data/the-verdict.txt", "r+") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [x.strip() for x in preprocessed if x.strip()]
    all_words = sorted(set(preprocessed))
    vocab = {w: i for i, w in enumerate(all_words)}
    return vocab


if __name__ == "__main__":
    print(get_vocab())
    print("done")
