from __future__ import annotations

import re

from src.preprocess.process_txt_to_vocab import get_vocab_v1, get_vocab_v2


class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str, int]):
        self._stoi = vocab
        self._itos = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [x.strip() for x in preprocessed if x.strip()]
        ids = [self._stoi[x] for x in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self._itos[x] for x in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


def test_v1():
    vocab = get_vocab_v1()
    obj = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know, "
    Mrs. Gisburn said with pardonable pride."""
    ids = obj.encode(text)
    print(ids)
    print(obj.decode(ids))
    return


class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str, int]):
        self._stoi = vocab
        self._itos = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [x.strip() for x in preprocessed if x.strip()]
        preprocessed = [x if x in self._stoi else "<|unk|>" for x in preprocessed]
        ids = [self._stoi[x] for x in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self._itos[x] for x in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


def test_v2():
    vocab = get_vocab_v2()
    obj = SimpleTokenizerV2(vocab)
    text = "<|endoftext|> ".join(
        [
            "Hello, do you like tea?",
            "In the sunlit terraces of the palace.",
        ]
    )
    ids = obj.encode(text)
    print(ids)
    print(obj.decode(ids))
    return


if __name__ == "__main__":
    # test_v1()
    test_v2()
    print("done")
