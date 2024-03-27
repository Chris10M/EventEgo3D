import os
import argparse
import tarfile
import urllib.request
from tqdm import tqdm


ROOT_URL = 'https://eventego3d.mpi-inf.mpg.de'
file = 'EE3D_BG.tar.gz'


def download_and_extract(url, save_location):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    
    # Download the file
    file_path = os.path.join(save_location, url.split('/')[-1])
    with tqdm(unit='B', unit_scale=True, desc='Downloading '+ url.split('/')[-1]) as tqdm_instance:
        urllib.request.urlretrieve(url, filename=file_path, reporthook=lambda block_num, block_size, total_size: tqdm_instance.update(block_size))
    
    # Extract the tar.gz file with progress bar
    with tarfile.open(file_path, 'r:gz') as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc='Extracting', unit='file') as tqdm_instance:
            for member in members:
                tar.extract(member, path=save_location)
                tqdm_instance.update(1)
    
    # Remove the tar.gz file after extraction
    os.remove(file_path)

    print(f"Files downloaded and extracted successfully to {save_location}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract a EE3D-BG.')
    parser.add_argument('--location', type=str, required=True, help='Location to save the downloaded and extracted files')

    args = parser.parse_args()

    download_and_extract(url=f'{ROOT_URL}/{file}', save_location=args.location)
