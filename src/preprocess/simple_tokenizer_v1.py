from __future__ import annotations

import re

from src.preprocess.process_txt_to_vocab import get_vocab


class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str, int]):
        self._stoi = vocab
        self._itos = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.?_!"{}\']|--|\s)', text)
        preprocessed = [x.strip() for x in preprocessed if x.strip()]
        ids = [self._stoi[x] for x in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self._itos[x] for x in ids])
        text = re.sub(r'\s+([,.?!"{}\'])', r"\1", text)
        return text


def test():
    vocab = get_vocab()
    obj = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know, "
    Mrs. Gisburn said with pardonable pride."""
    ids = obj.encode(text)
    print(ids)
    print(obj.decode(ids))
    return


if __name__ == "__main__":
    test()
    print("done")
