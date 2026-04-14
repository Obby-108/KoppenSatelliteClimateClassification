import os
import shutil
from tqdm.auto import tqdm

def cache_shards_locally(file_list, local_dir='../data'):
    os.makedirs(local_dir, exist_ok=True)
    cached = []

    for src_path in tqdm(file_list, desc="Caching shards locally"):
        filename = os.path.basename(str(src_path))
        dst_path = os.path.join(local_dir, filename)

        # Skip if already cached from a previous run
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

        cached.append(dst_path)

    return cached
