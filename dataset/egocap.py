import os
import numpy as np
import logging
import torchvision.transforms as transforms

from dataset.Joints3DDataset import Joints3DDataset


logger = logging.getLogger(__name__)


class EgoCap(Joints3DDataset):
    def _get_db(self):
        if 'train' in self.image_set:
            root = os.path.join(self.root, 'training_v000')
        else:
            root = os.path.join(self.root, 'validation_v003_2D')

        file_name = os.path.join(root, 'dataset.txt')
                
        with open(file_name, 'r') as anno_file:
            anno = anno_file.read()
            items = anno.split('#')[1:]

        gt_db = []
        for item in items:
            rows = item.split('\n')

            height, width = int(rows[3]), int(rows[4])

            j2d = np.array([np.fromstring(row, dtype=np.int32, sep=' ') for row in rows[6:24]])
            j2d = j2d[1:, 1:]

            gt_db.append({
                'image': os.path.join(root, rows[1][2:]),
                'j2d': j2d,
                'j2d_vis': np.ones_like(j2d),
                'filename': '',
                'imgnum': 0,   
                'center': np.array([width // 2, height // 2]),
                'scale': 2.75,         
            })

        return gt_db

    def __init__(self, cfg, image_set, is_train=True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.ToTensor(), normalize,])

        super().__init__(cfg, cfg.DATASET.ROOT, image_set, is_train, transform)
        self.num_joints = 17

        self.db = self._get_db()

        logger.info('=> load {} samples'.format(len(self.db)))

        
    def evaluate(self, cfg, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        # preds = preds[:, :, 0:2] + 1.0

        # if output_dir:
        #     pred_file = os.path.join(output_dir, 'pred.mat')
        #     savemat(pred_file, mdict={'preds': preds})

        # if 'test' in cfg.DATASET.TEST_SET:
        return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'annot',
                               'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']


def main():
    import sys; sys.path.append('../')
    from settings import config
    
    # EgoCap(config, 'train')
    EgoCap(config, 'test')

    # root = '/CT/3DHandObject/nobackup/EgoCap/training_v000/'
    
    # lines = open('/CT/3DHandObject/nobackup/EgoCap/training_v000/dataset.txt', 'r').read()
    # items = lines.split('#')[1:]

    # file_infos = list()
    # for item in items:
    #     rows = item.split('\n')
        
    #     j2d = np.array([np.fromstring(row, dtype=np.int32, sep=' ') for row in rows[6:24]])
    #     j2d = j2d[:, 1:]

    #     file_infos.append({
    #         'image': os.path.join(root, rows[1][2:]),
    #         'j2d': j2d,

    #         # 'center': c,
    #         # 'scale': s,
    #         # 'joints_3d_vis': joints_3d_vis,

    #         'filename': '',
    #         'imgnum': 0,            
    #     })
        
    #     for idx, row in enumerate(rows):
    #         print(idx, row)
        
    #     print(file_infos)
    #     exit(0)



if __name__ == '__main__':
    main()
