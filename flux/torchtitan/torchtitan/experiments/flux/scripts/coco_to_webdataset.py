import math
import os
import pandas as pd
import tarfile
import io
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tsv_file", type=str, required=True)
parser.add_argument("--image_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--samples_per_shard", type=int, default=1000)
args = parser.parse_args()


# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load the TSV file
df = pd.read_csv(args.tsv_file, sep='\t')[["image_id", "caption"]]

# Group data by shard
num_shards = math.ceil(len(df) / args.samples_per_shard)

for shard_idx in tqdm(range(num_shards), desc="Creating shards"):
    # Create a new tar file for this shard
    shard_name = f"{args.output_dir}/shard_{shard_idx:05d}.tar"
    
    start_idx = shard_idx * args.samples_per_shard
    end_idx = min((shard_idx + 1) * args.samples_per_shard, len(df))
    
    with tarfile.open(shard_name, "w") as tar:
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]
            image_id = row['image_id']
            caption = row['caption']
            
            # Define the base filename using image_id
            base_name = f"{image_id:012d}"  # Format as 12-digit number
            
            # Path to the image file
            img_path = os.path.join(args.image_dir, f"COCO_val2014_{base_name}.jpg")
            
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found. Skipping.")
                continue
            
            # Add image to tar
            img_info = tarfile.TarInfo(f"{base_name}.jpg")
            img_data = open(img_path, "rb").read()
            img_info.size = len(img_data)
            tar.addfile(img_info, io.BytesIO(img_data))
            
            # Create and add txt file with the caption
            txt_info = tarfile.TarInfo(f"{base_name}.txt")
            txt_data = caption
            txt_info.size = len(txt_data.encode('utf-8'))
            tar.addfile(txt_info, io.BytesIO(txt_data.encode('utf-8')))

print(f"WebDataset created with {num_shards} shards in {args.output_dir}")
