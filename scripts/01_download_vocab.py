from __future__ import annotations

import urllib.request


def download():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
    filename = "./data/the-verdict.txt"
    urllib.request.urlretrieve(url, filename)
    return


if __name__ == "__main__":
    download()
    print("done")
