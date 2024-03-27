import os
import argparse
import tarfile
import urllib.request
from tqdm import tqdm

DATASET = 'EE3D-S-Test'	
ROOT_URL = f'https://eventego3d.mpi-inf.mpg.de/{DATASET}'

files = [
    'test.txt',
    'train.txt',
    'val.txt',
    'pose_ung_144_20.tar.gz',
    'pose_03_04.tar.gz',
    'pose_05_11.tar.gz',
    'pose_05_16.tar.gz',
    'pose_05_17.tar.gz',
    'pose_13_17.tar.gz',
    'pose_15_13.tar.gz',
    'pose_18_03.tar.gz',
    'pose_18_05.tar.gz',
    'pose_32_18.tar.gz',
    'pose_35_02.tar.gz',
    'pose_35_33.tar.gz',
    'pose_36_12.tar.gz',
    'pose_54_06.tar.gz',
    'pose_55_15.tar.gz',
    'pose_63_25.tar.gz',
    'pose_79_20.tar.gz',
    'pose_79_47.tar.gz',
    'pose_79_72.tar.gz',
    'pose_80_28.tar.gz',
    'pose_80_40.tar.gz',
    'pose_91_02.tar.gz',
    'pose_105_14.tar.gz',
    'pose_105_40.tar.gz',
    'pose_105_52.tar.gz',
    'pose_106_14.tar.gz',
    'pose_108_20.tar.gz',
    'pose_111_01.tar.gz',
    'pose_123_07.tar.gz',
    'pose_128_01.tar.gz',
    'pose_131_08.tar.gz',
    'pose_132_30.tar.gz',
    'pose_132_45.tar.gz',
    'pose_132_56.tar.gz',
    'pose_135_02.tar.gz',
    'pose_137_18.tar.gz',
    'pose_137_35.tar.gz',
    'pose_137_37.tar.gz',
    'pose_138_25.tar.gz',
    'pose_143_19.tar.gz',
    'pose_143_41.tar.gz',
    'pose_144_04.tar.gz',
    'pose_144_19.tar.gz',
    'pose_ung_49_15.tar.gz',
    'pose_ung_60_09.tar.gz',
    'pose_ung_74_04.tar.gz',
    'pose_ung_74_06.tar.gz',
    'pose_ung_74_07.tar.gz',
    'pose_ung_76_10.tar.gz',
    'pose_ung_77_22.tar.gz',
    'pose_ung_77_24.tar.gz',
    'pose_ung_77_27.tar.gz',
    'pose_ung_77_34.tar.gz',
    'pose_ung_78_31.tar.gz',
    'pose_ung_82_06.tar.gz',
    'pose_ung_84_15.tar.gz',
    'pose_ung_90_14.tar.gz',
    'pose_ung_94_15.tar.gz',
    'pose_ung_102_14.tar.gz',
    'pose_ung_102_29.tar.gz',
    'pose_ung_102_32.tar.gz',
    'pose_ung_104_06.tar.gz',
    'pose_ung_104_07.tar.gz',
    'pose_ung_104_16.tar.gz',
    'pose_ung_104_48.tar.gz',
    'pose_ung_105_06.tar.gz',
    'pose_ung_105_24.tar.gz',
    'pose_ung_105_50.tar.gz',
    'pose_ung_105_56.tar.gz',
    'pose_ung_111_08.tar.gz',
    'pose_ung_111_21.tar.gz',
    'pose_ung_111_23.tar.gz',
    'pose_ung_113_07.tar.gz',
    'pose_ung_114_04.tar.gz',
    'pose_ung_115_01.tar.gz',
    'pose_ung_120_09.tar.gz',
    'pose_ung_122_03.tar.gz',
    'pose_ung_122_18.tar.gz',
    'pose_ung_126_13.tar.gz',
    'pose_ung_127_11.tar.gz',
    'pose_ung_132_10.tar.gz',
    'pose_ung_132_41.tar.gz',
    'pose_ung_139_15.tar.gz',
    'pose_ung_139_33.tar.gz',
    'pose_ung_144_01.tar.gz',
    'pose_ung_144_14.tar.gz',
    'pose_ung_144_18.tar.gz',
]


def extract_tar(file_path):
    # Extract the tar.gz file with progress bar
    with tarfile.open(file_path, 'r:gz') as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc='Extracting', unit='file') as tqdm_instance:
            for member in members:
                tar.extract(member, path=save_location)
                tqdm_instance.update(1)
    
    # Remove the tar.gz file after extraction
    os.remove(file_path)

def download_and_extract(url, save_location):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    
    # Download the file
    file_path = os.path.join(save_location, url.split('/')[-1])
    with tqdm(unit='B', unit_scale=True, desc='Downloading '+ url.split('/')[-1]) as tqdm_instance:
        urllib.request.urlretrieve(url, filename=file_path, reporthook=lambda block_num, block_size, total_size: tqdm_instance.update(block_size))
    
    if file_path.endswith('.tar.gz'):
        extract_tar(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract a EE3D-BG.')
    parser.add_argument('--location', type=str, required=True, help='Location to save the downloaded and extracted files')

    args = parser.parse_args()

    save_location = os.path.join(args.location , DATASET)
    os.makedirs(save_location, exist_ok=True)

    print('Downloading sequences..')
    for idx, file in enumerate(files, 1):  
        print('[{}/{}]: [{}]'.format(idx, len(files), file))
        download_and_extract(url=f'{ROOT_URL}/{file}', save_location=save_location)

    print(f"Files downloaded and extracted successfully to {save_location}")
