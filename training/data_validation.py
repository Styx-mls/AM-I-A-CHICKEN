from pathlib import Path 
from PIL import Image
import hashlib 
from collections import Counter, defaultdict
import statistics

DATA_DIR = Path("Core_Dataset/raw")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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

            file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()

            if file_hash in seen_hashes:
                duplicates.setdefault(file_hash, [seen_hashes[file_hash]])
                duplicates[file_hash].append(file_path)
            else:
                seen_hashes[file_hash] = file_path
            
            print(
                f"OK: {file_path} | "
                f"{width}x{height} | {image_format}"
            )
    except Exception as e:
        corrupt_images.append((file_path, str(e)))
        print(f"BAD: {file_path} | {e}")


print("\n--- SUMMARY ---")
print(f"Valid images: {valid_images}")
print(f"Corrupt images: {len(corrupt_images)}")
print(f"Corrupt images: {corrupt_images}")
print(f"Duplicate groups: {len(duplicates)}")

print("\n--- CLASS COUNTS ---")
for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")

print("\n--- DIMENSIONS BY CLASS ---")

for class_name in sorted(class_counts):

    widths = class_widths[class_name]
    heights = class_heights[class_name]

    print(f"\n{class_name}:")
    print(f"  Average: {statistics.mean(widths):.0f} x "
          f"{statistics.mean(heights):.0f}")

    print(f"  Median:  {statistics.median(widths):.0f} x "
          f"{statistics.median(heights):.0f}")

    print(f"  Min:     {min(widths)} x {min(heights)}")
    print(f"  Max:     {max(widths)} x {max(heights)}")