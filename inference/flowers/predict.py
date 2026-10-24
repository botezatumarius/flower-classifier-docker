import argparse
from pathlib import Path

from PIL import Image

from flowers.model import FlowerClassifier

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def expand(paths: list[Path]) -> list[Path]:
    files = (file for path in paths for file in (sorted(path.iterdir()) if path.is_dir() else [path]))
    return [file for file in files if file.suffix.lower() in IMAGE_SUFFIXES]


def main():
    parser = argparse.ArgumentParser(description="Classify flower photos from the terminal.")
    parser.add_argument("paths", nargs="+", type=Path, help="image files or directories")
    parser.add_argument("-k", type=int, default=3, help="number of top predictions")
    args = parser.parse_args()

    classifier = FlowerClassifier()
    for path in expand(args.paths):
        with Image.open(path) as image:
            predictions = classifier.predict(image, args.k)
        print(f"{path.name:<24}" + "  ".join(f"{p['label']} {p['score']:.1%}" for p in predictions))


if __name__ == "__main__":
    main()
