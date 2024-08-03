# Datasets

#### EE3D-S

The EE3D-S can be downloaded from by executing the following script,

```bash
python download_ee3d_s.py --location EE3D-S_SAVE_PATH
```

After the dataset is downloaded, you can set ````config.DATASET.SYN_ROOT```` in ```settings.py``` to ```EE3D-S_SAVE_PATH```.


#### EE3D-S Test

The EE3D-S-Test can be downloaded from by executing the following script,

```bash
python download_ee3d_s_test.py --location EE3D-S-Test_SAVE_PATH
```

After the dataset is downloaded, you can set ````config.DATASET.SYN_TEST_ROOT```` in ```settings.py``` to ```EE3D-S-Test_SAVE_PATH```.

#### EE3D-R

The EE3D-R can be downloaded from by executing the following script,

```bash
python download_ee3d_r.py --location EE3D-R_SAVE_PATH
```

After the dataset is downloaded, you can set ````config.DATASET.REAL_ROOT```` in ```settings.py``` to ```EE3D-R_SAVE_PATH```.


#### EE3D [BG-AUG] 

The data for augmenting the EE3D-R and EE3D-S datasets with background events can be obtained by executing the following script,

```bash
python download_ee3d_bg.py --location EE3D-BG_SAVE_PATH
```

After the data is downloaded, you can set ````config.DATASET.BACKGROUND_DATASET_ROOT```` in ```settings.py``` to to ```EE3D-BG_SAVE_PATH```.

