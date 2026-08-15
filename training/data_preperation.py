from pathlib import Path
import shutil
import random

RAW_DIR = Path("Core_Dataset/raw")
PROCESSED_DIR = Path("Core_Dataset/processed_v1")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SEED = 18
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

OTHER_PER_CLASS = 550

random.seed(SEED)

def map_label(source_class):

    source_class = source_class.lower()

    if source_class == "chicken":
        return "chicken"
    
    if source_class == "humans":
        return "human"
    
    return "other"

def get_images(class_dir):

    return [
            path for path in class_dir.rglob("*")
            if path.is_file() and 
            path.suffix.lower() in IMAGE_EXTENSIONS
            ]

def create_output_directories():
    
    for split in ["train", "val", "test"]:

        for label in ["chicken", "human", "other"]:
            directory = PROCESSED_DIR / split / label
            directory.mkdir(parents = True, exist_ok = True)


def copy_samples(samples, split):

    for image_path, label in samples:

        destination_dir = PROCESSED_DIR / split / label
        
        source_class = image_path.parent.name
        destination_name = (f"{source_class}_{image_path.name}")
        destination_path = destination_dir/ destination_name

        shutil.copy2(image_path, destination_path)


create_output_directories()

samples_by_label = { "chicken" : [],
                     "human" : [],
                     "other": []
                    }

for class_dir in RAW_DIR.iterdir():

    if not class_dir.is_dir():
        continue
    
    source_class = class_dir.name
    label = map_label(source_class)

    images = get_images(class_dir)

    random.shuffle(images)

    if label == "other":
        images = images[:OTHER_PER_CLASS]
    
    for image_path in images:
        samples_by_label[label].append((image_path, label))
    

train_samples = []
val_samples = []
test_samples = []

for label, samples in samples_by_label.items():

    random.shuffle(samples)

    n = len(samples)

    train_end = int(n*TRAIN_RATIO)
    val_end = train_end + int(n*VAL_RATIO)

    train_samples.extend(samples[:train_end])
    val_samples.extend(samples[train_end:val_end])
    test_samples.extend(samples[val_end:])


random.shuffle(train_samples)
random.shuffle(val_samples)
random.shuffle(test_samples)

copy_samples(train_samples, "train")
copy_samples(val_samples, "val")
copy_samples(test_samples, "test")


for name, samples in [
    ("Train", train_samples),
    ("Validation", val_samples),
    ("Test", test_samples)
]:

    print(f"\n{name}: {len(samples)}")

    for label in ["chicken", "human", "other"]:

        count = sum(
            sample_label == label
            for _, sample_label in samples
        )

        print(f"  {label}: {count}")