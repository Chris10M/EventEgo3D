"""
EventEgo3D: 3D Human Motion Capture from Egocentric Event Streams
https://4dqv.mpi-inf.mpg.de/EventEgo3D/

Dataset download script
"""

import os
import argparse
import tarfile
import urllib.request
from tqdm import tqdm

DATASET = 'EE3D-S'	
ROOT_URL = f'https://eventego3d.mpi-inf.mpg.de/{DATASET}'

files = [
    'test.txt',
    'train.txt',
    'val.txt',
    'pose_18_01.tar.gz', 'pose_18_06.tar.gz', 'pose_18_07.tar.gz', 'pose_18_10.tar.gz', 'pose_18_11.tar.gz', 'pose_18_12.tar.gz', 'pose_18_13.tar.gz', 
    'pose_18_15.tar.gz', 'pose_19_01.tar.gz', 'pose_19_07.tar.gz', 'pose_19_10.tar.gz', 'pose_19_12.tar.gz', 'pose_19_13.tar.gz', 'pose_19_15.tar.gz', 
    'pose_24_01.tar.gz', 'pose_26_02.tar.gz', 'pose_26_04.tar.gz', 'pose_26_06.tar.gz', 'pose_26_09.tar.gz', 'pose_26_11.tar.gz', 'pose_27_01.tar.gz', 
    'pose_27_03.tar.gz', 'pose_27_06.tar.gz', 'pose_27_11.tar.gz', 'pose_28_02.tar.gz', 'pose_28_03.tar.gz', 'pose_28_07.tar.gz', 'pose_28_13.tar.gz', 
    'pose_28_17.tar.gz', 'pose_32_01.tar.gz', 'pose_32_03.tar.gz', 'pose_32_05.tar.gz', 'pose_32_07.tar.gz', 'pose_32_08.tar.gz', 'pose_32_11.tar.gz', 
    'pose_32_15.tar.gz', 'pose_32_17.tar.gz', 'pose_32_19.tar.gz', 'pose_35_15.tar.gz', 'pose_35_19.tar.gz', 'pose_35_21.tar.gz', 'pose_35_22.tar.gz', 
    'pose_35_27.tar.gz', 'pose_35_28.tar.gz', 'pose_35_30.tar.gz', 'pose_35_31.tar.gz', 'pose_35_34.tar.gz', 'pose_36_01.tar.gz', 'pose_36_02.tar.gz', 
    'pose_36_03.tar.gz', 'pose_36_04.tar.gz', 'pose_36_07.tar.gz', 'pose_36_13.tar.gz', 'pose_36_17.tar.gz', 'pose_36_18.tar.gz', 'pose_36_22.tar.gz', 
    'pose_36_30.tar.gz', 'pose_36_32.tar.gz', 'pose_36_34.tar.gz', 'pose_36_35.tar.gz', 'pose_38_03.tar.gz', 'pose_39_03.tar.gz', 'pose_39_08.tar.gz', 
    'pose_39_09.tar.gz', 'pose_39_11.tar.gz', 'pose_39_12.tar.gz', 'pose_39_14.tar.gz', 'pose_40_05.tar.gz', 'pose_46_01.tar.gz', 'pose_54_01.tar.gz', 
    'pose_54_03.tar.gz', 'pose_54_12.tar.gz', 'pose_54_21.tar.gz', 'pose_54_24.tar.gz', 'pose_54_27.tar.gz', 'pose_55_01.tar.gz', 'pose_55_05.tar.gz', 
    'pose_55_10.tar.gz', 'pose_55_11.tar.gz', 'pose_55_12.tar.gz', 'pose_55_13.tar.gz', 'pose_55_16.tar.gz', 'pose_55_21.tar.gz', 'pose_55_22.tar.gz', 
    'pose_55_23.tar.gz', 'pose_55_24.tar.gz', 'pose_55_27.tar.gz', 'pose_62_01.tar.gz', 'pose_62_03.tar.gz', 'pose_62_09.tar.gz', 'pose_62_13.tar.gz', 
    'pose_62_14.tar.gz', 'pose_62_16.tar.gz', 'pose_62_18.tar.gz', 'pose_62_21.tar.gz', 'pose_62_22.tar.gz', 'pose_63_01.tar.gz', 'pose_63_02.tar.gz', 
    'pose_63_03.tar.gz', 'pose_63_06.tar.gz', 'pose_63_09.tar.gz', 'pose_63_14.tar.gz', 'pose_63_18.tar.gz', 'pose_63_21.tar.gz', 'pose_63_22.tar.gz',
    'pose_63_27.tar.gz', 'pose_63_30.tar.gz', 'pose_70_02.tar.gz', 'pose_70_03.tar.gz', 'pose_73_03.tar.gz', 'pose_73_09.tar.gz', 'pose_75_02.tar.gz',
    'pose_75_06.tar.gz', 'pose_75_09.tar.gz', 'pose_75_10.tar.gz', 'pose_75_11.tar.gz', 'pose_75_14.tar.gz', 'pose_75_17.tar.gz', 'pose_75_18.tar.gz', 
    'pose_76_01.tar.gz', 'pose_76_02.tar.gz', 'pose_76_06.tar.gz', 'pose_76_08.tar.gz', 'pose_76_10.tar.gz', 'pose_79_06.tar.gz', 'pose_79_08.tar.gz',
    'pose_79_13.tar.gz', 'pose_79_19.tar.gz', 'pose_79_21.tar.gz', 'pose_79_22.tar.gz', 'pose_79_23.tar.gz', 'pose_79_24.tar.gz', 'pose_79_30.tar.gz',
    'pose_79_40.tar.gz', 'pose_79_43.tar.gz', 'pose_79_44.tar.gz', 'pose_79_45.tar.gz', 'pose_79_46.tar.gz', 'pose_79_49.tar.gz', 'pose_79_51.tar.gz',
    'pose_79_57.tar.gz', 'pose_79_59.tar.gz', 'pose_79_61.tar.gz', 'pose_79_63.tar.gz', 'pose_79_65.tar.gz', 'pose_79_66.tar.gz', 'pose_79_69.tar.gz',
    'pose_79_74.tar.gz', 'pose_79_76.tar.gz', 'pose_79_77.tar.gz', 'pose_79_82.tar.gz', 'pose_79_84.tar.gz', 'pose_79_88.tar.gz', 'pose_80_04.tar.gz', 
    'pose_80_07.tar.gz', 'pose_80_11.tar.gz', 'pose_80_13.tar.gz', 'pose_80_21.tar.gz', 'pose_80_22.tar.gz', 'pose_80_23.tar.gz', 'pose_80_29.tar.gz', 
    'pose_80_41.tar.gz', 'pose_80_43.tar.gz', 'pose_80_44.tar.gz', 'pose_80_45.tar.gz', 'pose_80_46.tar.gz', 'pose_80_52.tar.gz', 'pose_80_54.tar.gz', 
    'pose_80_60.tar.gz', 'pose_80_61.tar.gz', 'pose_80_66.tar.gz', 'pose_80_69.tar.gz', 'pose_80_70.tar.gz', 'pose_80_72.tar.gz', 'pose_85_02.tar.gz', 
    'pose_85_04.tar.gz', 'pose_85_09.tar.gz', 'pose_85_14.tar.gz', 'pose_88_06.tar.gz', 'pose_88_07.tar.gz', 'pose_88_09.tar.gz', 'pose_89_01.tar.gz', 
    'pose_89_02.tar.gz', 'pose_89_03.tar.gz', 'pose_89_05.tar.gz', 'pose_89_06.tar.gz', 'pose_91_01.tar.gz', 'pose_91_03.tar.gz', 'pose_91_06.tar.gz', 
    'pose_91_07.tar.gz', 'pose_91_11.tar.gz', 'pose_91_12.tar.gz', 'pose_91_17.tar.gz', 'pose_91_19.tar.gz', 'pose_91_26.tar.gz', 'pose_91_29.tar.gz', 
    'pose_91_35.tar.gz', 'pose_91_36.tar.gz', 'pose_91_40.tar.gz', 'pose_91_47.tar.gz', 'pose_91_49.tar.gz', 'pose_91_50.tar.gz', 'pose_91_51.tar.gz',
    'pose_91_54.tar.gz', 'pose_91_55.tar.gz', 'pose_91_56.tar.gz', 'pose_91_60.tar.gz', 'pose_91_61.tar.gz', 'pose_104_02.tar.gz', 'pose_104_03.tar.gz',
    'pose_104_07.tar.gz', 'pose_104_10.tar.gz', 'pose_104_11.tar.gz', 'pose_104_13.tar.gz', 'pose_104_17.tar.gz', 'pose_104_20.tar.gz', 'pose_104_24.tar.gz',
    'pose_104_29.tar.gz', 'pose_104_33.tar.gz', 'pose_104_36.tar.gz', 'pose_104_37.tar.gz', 'pose_104_39.tar.gz', 'pose_104_45.tar.gz', 'pose_104_54.tar.gz', 
    'pose_104_56.tar.gz', 'pose_104_57.tar.gz', 'pose_105_01.tar.gz', 'pose_105_03.tar.gz', 'pose_105_04.tar.gz', 'pose_105_06.tar.gz', 'pose_105_08.tar.gz', 
    'pose_105_09.tar.gz', 'pose_105_10.tar.gz', 'pose_105_12.tar.gz', 'pose_105_13.tar.gz', 'pose_105_15.tar.gz', 'pose_105_16.tar.gz', 'pose_105_20.tar.gz', 
    'pose_105_22.tar.gz', 'pose_105_24.tar.gz', 'pose_105_25.tar.gz', 'pose_105_29.tar.gz', 'pose_105_30.tar.gz', 'pose_105_31.tar.gz', 'pose_105_32.tar.gz',
    'pose_105_33.tar.gz', 'pose_105_34.tar.gz', 'pose_105_35.tar.gz', 'pose_105_41.tar.gz', 'pose_105_43.tar.gz', 'pose_105_46.tar.gz', 'pose_105_48.tar.gz',
    'pose_105_49.tar.gz', 'pose_105_50.tar.gz', 'pose_105_58.tar.gz', 'pose_106_02.tar.gz', 'pose_106_04.tar.gz', 'pose_106_08.tar.gz', 'pose_106_16.tar.gz', 
    'pose_106_17.tar.gz', 'pose_106_19.tar.gz', 'pose_106_27.tar.gz', 'pose_106_28.tar.gz', 'pose_106_31.tar.gz', 'pose_106_32.tar.gz', 'pose_106_33.tar.gz',
    'pose_107_01.tar.gz', 'pose_107_04.tar.gz', 'pose_107_10.tar.gz', 'pose_107_12.tar.gz', 'pose_108_02.tar.gz', 'pose_108_05.tar.gz', 'pose_108_06.tar.gz', 
    'pose_108_07.tar.gz', 'pose_108_08.tar.gz', 'pose_108_10.tar.gz', 'pose_108_14.tar.gz', 'pose_108_18.tar.gz', 'pose_108_24.tar.gz', 'pose_108_27.tar.gz', 
    'pose_111_17.tar.gz', 'pose_111_18.tar.gz', 'pose_111_24.tar.gz', 'pose_111_25.tar.gz', 'pose_111_28.tar.gz', 'pose_111_31.tar.gz', 'pose_111_34.tar.gz', 
    'pose_123_03.tar.gz', 'pose_123_06.tar.gz', 'pose_123_08.tar.gz', 'pose_123_10.tar.gz', 'pose_123_12.tar.gz', 'pose_126_04.tar.gz', 'pose_126_09.tar.gz',
    'pose_126_11.tar.gz', 'pose_126_12.tar.gz', 'pose_126_13.tar.gz', 'pose_128_11.tar.gz', 'pose_131_02.tar.gz', 'pose_131_05.tar.gz', 'pose_131_06.tar.gz', 
    'pose_131_14.tar.gz', 'pose_132_04.tar.gz', 'pose_132_07.tar.gz', 'pose_132_09.tar.gz', 'pose_132_12.tar.gz', 'pose_132_13.tar.gz', 'pose_132_15.tar.gz', 
    'pose_132_19.tar.gz', 'pose_132_21.tar.gz', 'pose_132_23.tar.gz', 'pose_132_27.tar.gz', 'pose_132_32.tar.gz', 'pose_132_37.tar.gz', 'pose_132_39.tar.gz', 
    'pose_132_42.tar.gz', 'pose_132_44.tar.gz', 'pose_132_50.tar.gz', 'pose_132_51.tar.gz', 'pose_135_03.tar.gz', 'pose_135_04.tar.gz', 'pose_135_05.tar.gz',
    'pose_135_11.tar.gz', 'pose_136_02.tar.gz', 'pose_136_03.tar.gz', 'pose_136_07.tar.gz', 'pose_136_08.tar.gz', 'pose_136_10.tar.gz', 'pose_136_11.tar.gz',
    'pose_136_15.tar.gz', 'pose_136_17.tar.gz', 'pose_136_23.tar.gz', 'pose_136_25.tar.gz', 'pose_136_26.tar.gz', 'pose_137_04.tar.gz', 'pose_137_06.tar.gz',
    'pose_137_07.tar.gz', 'pose_137_12.tar.gz', 'pose_137_13.tar.gz', 'pose_137_14.tar.gz', 'pose_137_15.tar.gz', 'pose_137_21.tar.gz', 'pose_137_22.tar.gz',
    'pose_137_31.tar.gz', 'pose_137_32.tar.gz', 'pose_137_38.tar.gz', 'pose_137_39.tar.gz', 'pose_137_42.tar.gz', 'pose_138_05.tar.gz', 'pose_138_09.tar.gz',
    'pose_138_10.tar.gz', 'pose_138_16.tar.gz', 'pose_138_20.tar.gz', 'pose_138_22.tar.gz', 'pose_138_26.tar.gz', 'pose_138_38.tar.gz', 'pose_138_41.tar.gz', 
    'pose_138_44.tar.gz', 'pose_138_48.tar.gz', 'pose_143_05.tar.gz', 'pose_143_12.tar.gz', 'pose_143_14.tar.gz', 'pose_143_15.tar.gz', 'pose_143_16.tar.gz', 
    'pose_143_17.tar.gz', 'pose_143_18.tar.gz', 'pose_143_23.tar.gz', 'pose_143_31.tar.gz', 'pose_143_35.tar.gz', 'pose_143_37.tar.gz', 'pose_144_03.tar.gz', 
    'pose_144_05.tar.gz', 'pose_144_09.tar.gz', 'pose_144_13.tar.gz', 'pose_144_14.tar.gz', 'pose_144_17.tar.gz', 'pose_144_21.tar.gz', 'pose_144_22.tar.gz', 
    'pose_144_25.tar.gz', 'pose_144_29.tar.gz', 'pose_ung_07_03.tar.gz', 'pose_ung_07_04.tar.gz', 'pose_ung_07_07.tar.gz', 'pose_ung_07_11.tar.gz', 'pose_ung_10_02.tar.gz', 
    'pose_ung_12_03.tar.gz', 'pose_ung_12_04.tar.gz', 'pose_ung_42_01.tar.gz', 'pose_ung_49_03.tar.gz', 'pose_ung_49_05.tar.gz', 'pose_ung_49_06.tar.gz', 'pose_ung_49_10.tar.gz', 
    'pose_ung_49_16.tar.gz', 'pose_ung_49_18.tar.gz', 'pose_ung_49_21.tar.gz', 'pose_ung_49_22.tar.gz', 'pose_ung_56_04.tar.gz', 'pose_ung_60_01.tar.gz', 'pose_ung_60_02.tar.gz', 
    'pose_ung_60_05.tar.gz', 'pose_ung_60_06.tar.gz', 'pose_ung_60_13.tar.gz', 'pose_ung_60_15.tar.gz', 'pose_ung_61_01.tar.gz', 'pose_ung_61_02.tar.gz', 'pose_ung_61_03.tar.gz', 
    'pose_ung_61_05.tar.gz', 'pose_ung_61_09.tar.gz', 'pose_ung_61_10.tar.gz', 'pose_ung_61_11.tar.gz', 'pose_ung_61_13.tar.gz', 'pose_ung_61_14.tar.gz', 'pose_ung_70_09.tar.gz', 
    'pose_ung_70_10.tar.gz', 'pose_ung_73_04.tar.gz', 'pose_ung_73_05.tar.gz', 'pose_ung_74_03.tar.gz', 'pose_ung_74_14.tar.gz', 'pose_ung_74_17.tar.gz', 'pose_ung_74_18.tar.gz', 
    'pose_ung_74_19.tar.gz', 'pose_ung_76_01.tar.gz', 'pose_ung_76_02.tar.gz', 'pose_ung_76_03.tar.gz', 'pose_ung_76_05.tar.gz', 'pose_ung_76_09.tar.gz', 'pose_ung_77_05.tar.gz', 
    'pose_ung_77_06.tar.gz', 'pose_ung_77_07.tar.gz', 'pose_ung_77_13.tar.gz', 'pose_ung_77_16.tar.gz', 'pose_ung_77_20.tar.gz', 'pose_ung_77_21.tar.gz', 'pose_ung_77_25.tar.gz', 
    'pose_ung_77_29.tar.gz', 'pose_ung_77_30.tar.gz', 'pose_ung_78_02.tar.gz', 'pose_ung_78_03.tar.gz', 'pose_ung_78_09.tar.gz', 'pose_ung_78_13.tar.gz', 'pose_ung_78_14.tar.gz', 
    'pose_ung_78_18.tar.gz', 'pose_ung_78_29.tar.gz', 'pose_ung_82_02.tar.gz', 'pose_ung_82_08.tar.gz', 'pose_ung_82_13.tar.gz', 'pose_ung_83_02.tar.gz', 'pose_ung_83_05.tar.gz', 
    'pose_ung_83_07.tar.gz', 'pose_ung_83_12.tar.gz', 'pose_ung_83_14.tar.gz', 'pose_ung_83_15.tar.gz', 'pose_ung_83_16.tar.gz', 'pose_ung_83_20.tar.gz', 'pose_ung_83_21.tar.gz', 
    'pose_ung_83_23.tar.gz', 'pose_ung_83_24.tar.gz', 'pose_ung_83_26.tar.gz', 'pose_ung_83_29.tar.gz', 'pose_ung_83_32.tar.gz', 'pose_ung_83_34.tar.gz', 'pose_ung_83_35.tar.gz',
    'pose_ung_83_38.tar.gz', 'pose_ung_83_39.tar.gz', 'pose_ung_83_40.tar.gz', 'pose_ung_83_41.tar.gz', 'pose_ung_83_42.tar.gz', 'pose_ung_83_43.tar.gz', 'pose_ung_83_44.tar.gz',
    'pose_ung_83_46.tar.gz', 'pose_ung_83_48.tar.gz', 'pose_ung_83_52.tar.gz', 'pose_ung_83_53.tar.gz', 'pose_ung_83_54.tar.gz', 'pose_ung_83_57.tar.gz', 'pose_ung_83_63.tar.gz', 
    'pose_ung_83_65.tar.gz', 'pose_ung_84_01.tar.gz', 'pose_ung_84_03.tar.gz', 'pose_ung_84_04.tar.gz', 'pose_ung_84_12.tar.gz', 'pose_ung_85_02.tar.gz', 'pose_ung_85_04.tar.gz', 
    'pose_ung_85_07.tar.gz', 'pose_ung_85_08.tar.gz', 'pose_ung_85_09.tar.gz', 'pose_ung_85_14.tar.gz', 'pose_ung_86_02.tar.gz', 'pose_ung_86_09.tar.gz', 'pose_ung_86_10.tar.gz', 
    'pose_ung_86_14.tar.gz', 'pose_ung_86_15.tar.gz', 'pose_ung_87_03.tar.gz', 'pose_ung_87_04.tar.gz', 'pose_ung_90_02.tar.gz', 'pose_ung_90_08.tar.gz', 'pose_ung_90_09.tar.gz', 
    'pose_ung_90_17.tar.gz', 'pose_ung_90_20.tar.gz', 'pose_ung_90_22.tar.gz', 'pose_ung_90_24.tar.gz', 'pose_ung_90_29.tar.gz', 'pose_ung_90_30.tar.gz', 'pose_ung_90_33.tar.gz', 
    'pose_ung_91_03.tar.gz', 'pose_ung_91_05.tar.gz', 'pose_ung_91_08.tar.gz', 'pose_ung_91_12.tar.gz', 'pose_ung_91_14.tar.gz', 'pose_ung_91_16.tar.gz', 'pose_ung_91_18.tar.gz', 
    'pose_ung_91_23.tar.gz', 'pose_ung_91_24.tar.gz', 'pose_ung_91_26.tar.gz', 'pose_ung_91_28.tar.gz', 'pose_ung_91_29.tar.gz', 'pose_ung_91_34.tar.gz', 'pose_ung_91_35.tar.gz', 
    'pose_ung_91_39.tar.gz', 'pose_ung_91_40.tar.gz', 'pose_ung_91_42.tar.gz', 'pose_ung_91_44.tar.gz', 'pose_ung_91_45.tar.gz', 'pose_ung_91_46.tar.gz', 'pose_ung_91_50.tar.gz', 
    'pose_ung_91_51.tar.gz', 'pose_ung_91_52.tar.gz', 'pose_ung_91_54.tar.gz', 'pose_ung_91_58.tar.gz', 'pose_ung_91_60.tar.gz', 'pose_ung_91_61.tar.gz', 'pose_ung_93_02.tar.gz', 
    'pose_ung_94_01.tar.gz', 'pose_ung_94_02.tar.gz', 'pose_ung_94_12.tar.gz', 'pose_ung_94_16.tar.gz', 'pose_ung_102_02.tar.gz', 'pose_ung_102_04.tar.gz', 'pose_ung_102_12.tar.gz', 
    'pose_ung_102_17.tar.gz', 'pose_ung_102_19.tar.gz', 'pose_ung_102_21.tar.gz', 'pose_ung_102_23.tar.gz', 'pose_ung_102_24.tar.gz', 'pose_ung_102_27.tar.gz', 'pose_ung_102_28.tar.gz',
    'pose_ung_102_31.tar.gz', 'pose_ung_103_02.tar.gz', 'pose_ung_103_04.tar.gz', 'pose_ung_103_05.tar.gz', 'pose_ung_104_02.tar.gz', 'pose_ung_104_03.tar.gz', 'pose_ung_104_12.tar.gz', 
    'pose_ung_104_17.tar.gz', 'pose_ung_104_19.tar.gz', 'pose_ung_104_20.tar.gz', 'pose_ung_104_35.tar.gz', 'pose_ung_104_39.tar.gz', 'pose_ung_104_44.tar.gz', 'pose_ung_104_52.tar.gz', 
    'pose_ung_104_53.tar.gz', 'pose_ung_105_03.tar.gz', 'pose_ung_105_04.tar.gz', 'pose_ung_105_05.tar.gz', 'pose_ung_105_13.tar.gz', 'pose_ung_105_25.tar.gz', 'pose_ung_105_27.tar.gz', 
    'pose_ung_105_28.tar.gz', 'pose_ung_105_31.tar.gz', 'pose_ung_105_32.tar.gz', 'pose_ung_105_36.tar.gz', 'pose_ung_105_37.tar.gz', 'pose_ung_105_44.tar.gz', 'pose_ung_105_45.tar.gz', 
    'pose_ung_105_46.tar.gz', 'pose_ung_105_47.tar.gz', 'pose_ung_105_49.tar.gz', 'pose_ung_105_57.tar.gz', 'pose_ung_105_58.tar.gz', 'pose_ung_105_62.tar.gz', 'pose_ung_106_02.tar.gz', 
    'pose_ung_106_06.tar.gz', 'pose_ung_106_13.tar.gz', 'pose_ung_106_14.tar.gz', 'pose_ung_106_15.tar.gz', 'pose_ung_106_17.tar.gz', 'pose_ung_106_22.tar.gz', 'pose_ung_106_25.tar.gz', 
    'pose_ung_106_26.tar.gz', 'pose_ung_106_28.tar.gz', 'pose_ung_106_31.tar.gz', 'pose_ung_107_02.tar.gz', 'pose_ung_107_04.tar.gz', 'pose_ung_107_12.tar.gz', 'pose_ung_107_13.tar.gz', 
    'pose_ung_108_05.tar.gz', 'pose_ung_108_08.tar.gz', 'pose_ung_108_10.tar.gz', 'pose_ung_108_13.tar.gz', 'pose_ung_108_15.tar.gz', 'pose_ung_108_17.tar.gz', 'pose_ung_108_19.tar.gz', 
    'pose_ung_108_21.tar.gz', 'pose_ung_111_06.tar.gz', 'pose_ung_111_07.tar.gz', 'pose_ung_111_14.tar.gz', 'pose_ung_111_15.tar.gz', 'pose_ung_111_20.tar.gz', 'pose_ung_111_26.tar.gz', 
    'pose_ung_111_27.tar.gz', 'pose_ung_111_29.tar.gz', 'pose_ung_111_34.tar.gz', 'pose_ung_111_38.tar.gz', 'pose_ung_111_41.tar.gz', 'pose_ung_113_05.tar.gz', 'pose_ung_113_08.tar.gz', 
    'pose_ung_113_09.tar.gz', 'pose_ung_113_10.tar.gz', 'pose_ung_113_13.tar.gz', 'pose_ung_113_14.tar.gz', 'pose_ung_113_17.tar.gz', 'pose_ung_113_19.tar.gz', 'pose_ung_113_25.tar.gz', 
    'pose_ung_113_28.tar.gz', 'pose_ung_113_29.tar.gz', 'pose_ung_114_02.tar.gz', 'pose_ung_114_03.tar.gz', 'pose_ung_114_08.tar.gz', 'pose_ung_114_09.tar.gz', 'pose_ung_114_10.tar.gz', 
    'pose_ung_115_02.tar.gz', 'pose_ung_115_07.tar.gz', 'pose_ung_118_06.tar.gz', 'pose_ung_118_07.tar.gz', 'pose_ung_118_09.tar.gz', 'pose_ung_118_10.tar.gz', 'pose_ung_118_15.tar.gz', 
    'pose_ung_118_28.tar.gz', 'pose_ung_120_07.tar.gz', 'pose_ung_120_12.tar.gz', 'pose_ung_120_14.tar.gz', 'pose_ung_120_19.tar.gz', 'pose_ung_120_21.tar.gz', 'pose_ung_122_04.tar.gz', 
    'pose_ung_122_05.tar.gz', 'pose_ung_122_07.tar.gz', 'pose_ung_122_09.tar.gz', 'pose_ung_122_12.tar.gz', 'pose_ung_122_14.tar.gz', 'pose_ung_122_21.tar.gz', 'pose_ung_122_22.tar.gz', 
    'pose_ung_122_23.tar.gz', 'pose_ung_122_24.tar.gz', 'pose_ung_122_31.tar.gz', 'pose_ung_122_34.tar.gz', 'pose_ung_122_46.tar.gz', 'pose_ung_122_47.tar.gz', 'pose_ung_122_49.tar.gz', 
    'pose_ung_122_50.tar.gz', 'pose_ung_122_55.tar.gz', 'pose_ung_122_59.tar.gz', 'pose_ung_122_61.tar.gz', 'pose_ung_122_64.tar.gz', 'pose_ung_122_66.tar.gz', 'pose_ung_123_08.tar.gz', 
    'pose_ung_123_12.tar.gz', 'pose_ung_124_02.tar.gz', 'pose_ung_124_04.tar.gz', 'pose_ung_124_11.tar.gz', 'pose_ung_124_13.tar.gz', 'pose_ung_125_01.tar.gz', 'pose_ung_125_02.tar.gz',
    'pose_ung_125_03.tar.gz', 'pose_ung_125_04.tar.gz', 'pose_ung_126_01.tar.gz', 'pose_ung_126_02.tar.gz', 'pose_ung_126_06.tar.gz', 'pose_ung_126_07.tar.gz', 'pose_ung_126_11.tar.gz', 
    'pose_ung_127_03.tar.gz', 'pose_ung_127_10.tar.gz', 'pose_ung_127_20.tar.gz', 'pose_ung_127_21.tar.gz', 'pose_ung_127_23.tar.gz', 'pose_ung_127_27.tar.gz', 'pose_ung_127_28.tar.gz',
    'pose_ung_127_29.tar.gz', 'pose_ung_127_31.tar.gz', 'pose_ung_127_32.tar.gz', 'pose_ung_127_35.tar.gz', 'pose_ung_127_36.tar.gz', 'pose_ung_128_02.tar.gz', 'pose_ung_128_03.tar.gz', 
    'pose_ung_128_05.tar.gz', 'pose_ung_128_07.tar.gz', 'pose_ung_131_01.tar.gz', 'pose_ung_131_04.tar.gz', 'pose_ung_131_05.tar.gz', 'pose_ung_131_07.tar.gz', 'pose_ung_131_08.tar.gz', 
    'pose_ung_131_10.tar.gz', 'pose_ung_131_13.tar.gz', 'pose_ung_132_03.tar.gz', 'pose_ung_132_06.tar.gz', 'pose_ung_132_07.tar.gz', 'pose_ung_132_08.tar.gz', 'pose_ung_132_14.tar.gz', 
    'pose_ung_132_16.tar.gz', 'pose_ung_132_24.tar.gz', 'pose_ung_132_28.tar.gz', 'pose_ung_132_44.tar.gz', 'pose_ung_132_46.tar.gz', 'pose_ung_132_47.tar.gz', 'pose_ung_132_48.tar.gz', 
    'pose_ung_132_49.tar.gz', 'pose_ung_132_51.tar.gz', 'pose_ung_132_55.tar.gz', 'pose_ung_132_56.tar.gz', 'pose_ung_133_01.tar.gz', 'pose_ung_133_02.tar.gz', 'pose_ung_133_06.tar.gz', 
    'pose_ung_133_08.tar.gz', 'pose_ung_133_09.tar.gz', 'pose_ung_133_10.tar.gz', 'pose_ung_133_11.tar.gz', 'pose_ung_133_17.tar.gz', 'pose_ung_133_20.tar.gz', 'pose_ung_133_21.tar.gz', 
    'pose_ung_133_23.tar.gz', 'pose_ung_134_01.tar.gz', 'pose_ung_134_08.tar.gz', 'pose_ung_135_02.tar.gz', 'pose_ung_135_06.tar.gz', 'pose_ung_135_08.tar.gz', 'pose_ung_135_09.tar.gz', 
    'pose_ung_135_11.tar.gz', 'pose_ung_136_02.tar.gz', 'pose_ung_136_03.tar.gz', 'pose_ung_136_08.tar.gz', 'pose_ung_136_10.tar.gz', 'pose_ung_136_13.tar.gz', 'pose_ung_136_14.tar.gz', 
    'pose_ung_136_16.tar.gz', 'pose_ung_136_18.tar.gz', 'pose_ung_136_22.tar.gz', 'pose_ung_136_24.tar.gz', 'pose_ung_136_25.tar.gz', 'pose_ung_136_30.tar.gz', 'pose_ung_137_03.tar.gz', 
    'pose_ung_137_04.tar.gz', 'pose_ung_137_08.tar.gz', 'pose_ung_137_10.tar.gz', 'pose_ung_137_13.tar.gz', 'pose_ung_137_15.tar.gz', 'pose_ung_137_19.tar.gz', 'pose_ung_137_21.tar.gz', 
    'pose_ung_137_26.tar.gz', 'pose_ung_137_27.tar.gz', 'pose_ung_137_28.tar.gz', 'pose_ung_137_31.tar.gz', 'pose_ung_137_32.tar.gz', 'pose_ung_137_33.tar.gz', 'pose_ung_137_35.tar.gz',
    'pose_ung_137_37.tar.gz', 'pose_ung_137_42.tar.gz', 'pose_ung_138_01.tar.gz', 'pose_ung_138_04.tar.gz', 'pose_ung_138_08.tar.gz', 'pose_ung_138_09.tar.gz', 'pose_ung_138_12.tar.gz', 
    'pose_ung_138_17.tar.gz', 'pose_ung_138_20.tar.gz', 'pose_ung_138_22.tar.gz', 'pose_ung_138_26.tar.gz', 'pose_ung_138_27.tar.gz', 'pose_ung_138_28.tar.gz', 'pose_ung_138_36.tar.gz', 
    'pose_ung_138_42.tar.gz', 'pose_ung_138_47.tar.gz', 'pose_ung_138_48.tar.gz', 'pose_ung_138_52.tar.gz', 'pose_ung_138_55.tar.gz', 'pose_ung_139_01.tar.gz', 'pose_ung_139_07.tar.gz', 
    'pose_ung_139_09.tar.gz', 'pose_ung_139_11.tar.gz', 'pose_ung_139_12.tar.gz', 'pose_ung_139_18.tar.gz', 'pose_ung_139_19.tar.gz', 'pose_ung_139_20.tar.gz', 'pose_ung_139_21.tar.gz', 
    'pose_ung_139_23.tar.gz', 'pose_ung_139_27.tar.gz', 'pose_ung_139_32.tar.gz', 'pose_ung_140_02.tar.gz', 'pose_ung_140_09.tar.gz', 'pose_ung_142_01.tar.gz', 'pose_ung_142_10.tar.gz', 
    'pose_ung_142_13.tar.gz', 'pose_ung_142_14.tar.gz', 'pose_ung_142_18.tar.gz', 'pose_ung_142_21.tar.gz', 'pose_ung_142_22.tar.gz', 'pose_ung_143_01.tar.gz', 'pose_ung_143_05.tar.gz', 
    'pose_ung_143_09.tar.gz', 'pose_ung_143_11.tar.gz', 'pose_ung_143_13.tar.gz', 'pose_ung_143_14.tar.gz', 'pose_ung_143_18.tar.gz', 'pose_ung_143_19.tar.gz', 'pose_ung_143_21.tar.gz', 
    'pose_ung_143_24.tar.gz', 'pose_ung_143_26.tar.gz', 'pose_ung_143_27.tar.gz', 'pose_ung_143_32.tar.gz', 'pose_ung_143_34.tar.gz', 'pose_ung_144_07.tar.gz', 'pose_ung_144_08.tar.gz', 
    'pose_ung_144_12.tar.gz', 'pose_ung_144_13.tar.gz', 'pose_ung_144_15.tar.gz', 'pose_ung_144_16.tar.gz', 'pose_ung_144_19.tar.gz', 'pose_ung_144_22.tar.gz', 'pose_ung_144_29.tar.gz', 
    'pose_ung_144_31.tar.gz', 'pose_ung_144_32.tar.gz', 'pose_01_02.tar.gz', 'pose_01_05.tar.gz', 'pose_01_12.tar.gz', 'pose_02_03.tar.gz', 'pose_02_07.tar.gz', 'pose_05_04.tar.gz', 'pose_05_05.tar.gz', 
    'pose_05_08.tar.gz', 'pose_05_13.tar.gz', 'pose_05_14.tar.gz', 'pose_05_18.tar.gz', 'pose_05_19.tar.gz', 'pose_06_05.tar.gz', 'pose_06_06.tar.gz', 'pose_06_11.tar.gz', 'pose_06_15.tar.gz', 'pose_08_05.tar.gz', 
    'pose_08_11.tar.gz', 'pose_09_03.tar.gz', 'pose_09_08.tar.gz', 'pose_09_09.tar.gz', 'pose_09_10.tar.gz', 'pose_09_12.tar.gz', 'pose_10_01.tar.gz', 'pose_15_02.tar.gz', 'pose_15_03.tar.gz', 'pose_15_05.tar.gz', 
    'pose_15_07.tar.gz', 'pose_15_08.tar.gz', 'pose_16_35.tar.gz', 'pose_17_01.tar.gz', 'pose_17_03.tar.gz',
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
