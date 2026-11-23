from beir import util
import os
from pathlib import Path
import zipfile

# Download and save dataset
datasets = ['nq', 
            'msmarco', 
            'hotpotqa',
            'nfcorpus',
            'trec-covid']
for dataset in datasets:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, dataset)
    zip_path = Path(out_dir) / f"{dataset}.zip"
    if not os.path.exists(data_path):
        for attempt in range(1, 4):
            # Remove partial/corrupted archives left by interrupted downloads.
            if zip_path.exists() and not zipfile.is_zipfile(zip_path):
                zip_path.unlink(missing_ok=True)
            try:
                data_path = util.download_and_unzip(url, out_dir)
                break
            except zipfile.BadZipFile:
                zip_path.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                print(f"Corrupted archive for {dataset}; retrying ({attempt}/3)...")

for zip_file in Path(out_dir).glob("*.zip"):
    zip_file.unlink(missing_ok=True)