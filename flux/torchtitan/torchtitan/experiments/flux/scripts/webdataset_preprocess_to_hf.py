from datasets import load_dataset
from torchvision import transforms
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--output_size", type=int, default=256)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()


img_transform = transforms.Compose(
        [
            transforms.Resize(args.output_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.output_size),
        ]
    )

dataset = load_dataset("webdataset", data_dir=args.path, split="train", num_proc=8, cache_dir=os.path.join(args.path, "..", "hf_cache"))

def transform_fn(examples):
    examples["jpg"] = [img_transform(image.convert("RGB")) for image in examples["jpg"]]
    # in the future also tokenize and even encode text
    return examples

dataset = dataset.map(transform_fn, num_proc=32, batched=True)

print(dataset[0])
dataset.save_to_disk(args.output_path, max_shard_size="1GB")

