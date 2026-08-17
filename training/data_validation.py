from pathlib import Path
from PIL import Image
import hashlib
from collections import Counter, defaultdict
import statistics


TRAINING_DIR = Path(__file__).resolve().parent
DATA_DIR = TRAINING_DIR / "Core_Dataset" / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_data() -> None:
    valid_images = 0
    corrupt_images = []
    duplicates = {}
    seen_hashes = {}

    class_counts = Counter()

    widths = []
    heights = []

    class_widths = defaultdict(list)
    class_heights = defaultdict(list)

    for file_path in DATA_DIR.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            with Image.open(file_path) as img:
                img.verify()

            with Image.open(file_path) as img:
                width, height = img.size
                image_format = img.format

            valid_images += 1

            widths.append(width)
            heights.append(height)

            class_name = file_path.relative_to(DATA_DIR).parts[0]

            class_counts[class_name] += 1
            class_widths[class_name].append(width)
            class_heights[class_name].append(height)

            file_hash = hashlib.md5(
                file_path.read_bytes()
            ).hexdigest()

            if file_hash in seen_hashes:
                duplicates.setdefault(
                    file_hash,
                    [seen_hashes[file_hash]]
                )
                duplicates[file_hash].append(file_path)

            else:
                seen_hashes[file_hash] = file_path

            print(
                f"OK: {file_path} | "
                f"{width}x{height} | {image_format}"
            )

        except Exception as e:
            corrupt_images.append(
                (file_path, str(e))
            )

            print(
                f"BAD: {file_path} | {e}"
            )

    print("\n--- SUMMARY ---")
    print(f"Valid images: {valid_images}")
    print(f"Corrupt images: {len(corrupt_images)}")
    print(f"Corrupt images: {corrupt_images}")
    print(f"Duplicate groups: {len(duplicates)}")

    if widths and heights:
        print("\n--- OVERALL DIMENSIONS ---")
        print(
            f"Average: "
            f"{statistics.mean(widths):.0f} x "
            f"{statistics.mean(heights):.0f}"
        )
        print(
            f"Median: "
            f"{statistics.median(widths):.0f} x "
            f"{statistics.median(heights):.0f}"
        )
        print(
            f"Min: "
            f"{min(widths)} x {min(heights)}"
        )
        print(
            f"Max: "
            f"{max(widths)} x {max(heights)}"
        )

    print("\n--- CLASS COUNTS ---")

    for class_name, count in sorted(
        class_counts.items()
    ):
        print(
            f"{class_name}: {count}"
        )

    print("\n--- DIMENSIONS BY CLASS ---")

    for class_name in sorted(class_counts):

        class_width_values = (
            class_widths[class_name]
        )

        class_height_values = (
            class_heights[class_name]
        )

        print(f"\n{class_name}:")

        print(
            f"  Average: "
            f"{statistics.mean(class_width_values):.0f} x "
            f"{statistics.mean(class_height_values):.0f}"
        )

        print(
            f"  Median:  "
            f"{statistics.median(class_width_values):.0f} x "
            f"{statistics.median(class_height_values):.0f}"
        )

        print(
            f"  Min:     "
            f"{min(class_width_values)} x "
            f"{min(class_height_values)}"
        )

        print(
            f"  Max:     "
            f"{max(class_width_values)} x "
            f"{max(class_height_values)}"
        )


if __name__ == "__main__":
    validate_data()