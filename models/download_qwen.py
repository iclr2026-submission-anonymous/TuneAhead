#!/usr/bin/env python3


import os
from modelscope import snapshot_download


def get_dir_size_gb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            fp = os.path.join(dirpath, filename)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 ** 3)


def main():
    model_id = 'Qwen/Qwen2.5-7B-Instruct-1M'
    cache_dir = '/model/models'

    os.makedirs(cache_dir, exist_ok=True)

    print(f"\U0001F680 Start downloading model to cache directory: {os.path.abspath(cache_dir)}")

    # ModelScope uses cache_dir to specify the local cache root directory
    # The return value is the actual local model directory path
    model_dir = snapshot_download(
        model_id,
        cache_dir=cache_dir
    )

    print("\u2705 Model download completed!")
    print(f"\U0001F4C1 Saved to: {model_dir}")
    print(f"\U0001F4CA Directory size: {get_dir_size_gb(model_dir):.2f} GB")


if __name__ == '__main__':
    main()