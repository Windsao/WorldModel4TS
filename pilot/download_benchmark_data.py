"""Download the two added official VisionTS/TSLib benchmark datasets."""

import argparse

from huggingface_hub import hf_hub_download


FILES = ("ETT-small/ETTm1.csv", "weather/weather.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="pilot/data")
    args = parser.parse_args()
    for filename in FILES:
        path = hf_hub_download(
            repo_id="thuml/Time-Series-Library",
            filename=filename,
            repo_type="dataset",
            local_dir=args.data_dir,
        )
        print(path)


if __name__ == "__main__":
    main()
