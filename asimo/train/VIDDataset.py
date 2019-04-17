"""
Dataset for VID
"""
import os
import json
from torch.utils.data.dataset import Dataset
from utility.generate_anchors import compute_backbone_shapes,generate_pyramid_anchors

class VIDDataset(Dataset):

    '''
    data_dir = r"D:\workspace\MachineLearning\asimo\OTB_train_crops\img"
    imdb = r"D:\workspace\MachineLearning\asimo\imdb_video_train_otb.json"
    '''

    #def __init__(self, imdb, data_dir, config, z_transforms, x_transforms, mode="Train"):
        #imdb_video = json.load(open(imdb, 'r'))
        #self.videos = imdb_video['videos']
        #self.data_dir = data_dir
        #self.config = config
        #self.num_videos = int(imdb_video['num_videos'])

        #self.z_transforms = z_transforms
        #self.x_transforms = x_transforms

        #if mode == "Train":
        #    self.num = self.config.NUM_PER_EPOCH
        #else:
        #    self.num = self.num_videos
    def __init__(self, db_path, video_names, data_dir, z_transforms, x_transforms, training=True):
        self.video_names = video_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.meta_data = {x[0]: x[1] for x in self.meta_data}
        # filter traj len less than 2
        for key in self.meta_data.keys():
            trajs = self.meta_data[key]
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]

        # self.txn = db.begin(write=False)
        self.db_path = db_path
        self.num = len(self.video_names) if config.num_per_epoch is None or not training \
            else config.num_per_epoch

        # data augmentation
        self.max_stretch = config.scale_resize
        self.max_translate = config.max_translate
        self.random_crop_size = config.instance_size
        self.center_crop_size = config.exemplar_size

        self.training = training

        # 计算出所有的锚标签，共计5*3=15种
        backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE) # array([[256,256],[128,128],[64,64],[32,32],[16,16]])
        self.anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                config.RPN_ANCHOR_RATIOS,
                                                backbone_shapes,
                                                config.BACKBONE_STRIDES,
                                                config.RPN_ANCHOR_STRIDE)        

    def RandomStretch(self, sample, gt_w, gt_h):
        scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        h, w = sample.shape[:2]
        shape = int(w * scale_w), int(h * scale_h)
        scale_w = int(w * scale_w) / w
        scale_h = int(h * scale_h) / h
        gt_w = gt_w * scale_w
        gt_h = gt_h * scale_h
        return cv2.resize(sample, shape, cv2.INTER_LINEAR), gt_w, gt_h

    def CenterCrop(self, sample, ):
        im_h, im_w, _ = sample.shape
        cy = (im_h - 1) / 2
        cx = (im_w - 1) / 2

        ymin = cy - self.center_crop_size / 2 + 1 / 2
        xmin = cx - self.center_crop_size / 2 + 1 / 2
        ymax = ymin + self.center_crop_size - 1
        xmax = xmin + self.center_crop_size - 1

        left = int(round(max(0., -xmin)))
        top = int(round(max(0., -ymin)))
        right = int(round(max(0., xmax - im_w + 1)))
        bottom = int(round(max(0., ymax - im_h + 1)))

        xmin = int(round(xmin + left))
        xmax = int(round(xmax + left))
        ymin = int(round(ymin + top))
        ymax = int(round(ymax + top))

        r, c, k = sample.shape
        if any([top, bottom, left, right]):
            img_mean = tuple(map(int, sample.mean(axis=(0, 1))))
            te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)  # 0 is better than 1 initialization
            te_im[top:top + r, left:left + c, :] = sample
            if top:
                te_im[0:top, left:left + c, :] = img_mean
            if bottom:
                te_im[r + top:, left:left + c, :] = img_mean
            if left:
                te_im[:, 0:left, :] = img_mean
            if right:
                te_im[:, c + left:, :] = img_mean
            im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
        else:
            im_patch_original = sample[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]

        if not np.array_equal(im_patch_original.shape[:2], (self.center_crop_size, self.center_crop_size)):
            im_patch = cv2.resize(im_patch_original,
                                  (self.center_crop_size, self.center_crop_size))  # zzp: use cv to get a better speed
        else:
            im_patch = im_patch_original
        return im_patch

    def RandomCrop(self, sample, ):
        im_h, im_w, _ = sample.shape
        cy_o = (im_h - 1) / 2
        cx_o = (im_w - 1) / 2
        cy = np.random.randint(cy_o - self.max_translate,
                               cy_o + self.max_translate + 1)
        cx = np.random.randint(cx_o - self.max_translate,
                               cx_o + self.max_translate + 1)
        # assert abs(cy - cy_o) <= self.max_translate and \
        #        abs(cx - cx_o) <= self.max_translate
        gt_cx = cx_o - cx
        gt_cy = cy_o - cy

        ymin = cy - self.random_crop_size / 2 + 1 / 2
        xmin = cx - self.random_crop_size / 2 + 1 / 2
        ymax = ymin + self.random_crop_size - 1
        xmax = xmin + self.random_crop_size - 1

        left = int(round(max(0., -xmin)))
        top = int(round(max(0., -ymin)))
        right = int(round(max(0., xmax - im_w + 1)))
        bottom = int(round(max(0., ymax - im_h + 1)))

        xmin = int(round(xmin + left))
        xmax = int(round(xmax + left))
        ymin = int(round(ymin + top))
        ymax = int(round(ymax + top))

        r, c, k = sample.shape
        if any([top, bottom, left, right]):
            img_mean = tuple(map(int, sample.mean(axis=(0, 1))))
            te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)  # 0 is better than 1 initialization
            te_im[top:top + r, left:left + c, :] = sample
            if top:
                te_im[0:top, left:left + c, :] = img_mean
            if bottom:
                te_im[r + top:, left:left + c, :] = img_mean
            if left:
                te_im[:, 0:left, :] = img_mean
            if right:
                te_im[:, c + left:, :] = img_mean
            im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
        else:
            im_patch_original = sample[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]

        if not np.array_equal(im_patch_original.shape[:2], (self.random_crop_size, self.random_crop_size)):
            im_patch = cv2.resize(im_patch_original,
                                  (self.random_crop_size, self.random_crop_size))  # zzp: use cv to get a better speed
        else:
            im_patch = im_patch_original
        return im_patch, gt_cx, gt_cy

    def compute_target(self, anchors, box):
        regression_target = box_transform(anchors, box)

        iou = compute_iou(anchors, box).flatten()
        # print(np.max(iou))
        pos_index = np.where(iou > config.pos_threshold)[0]
        neg_index = np.where(iou < config.neg_threshold)[0]
        label = np.ones_like(iou) * -1
        label[pos_index] = 1
        label[neg_index] = 0
        return regression_target, label        

    def __getitem__bak(self, rand_vid):
        while True:
            '''
            read a pair of images z and x
            '''
            # randomly decide the id of video to get z and x
            rand_vid = rand_vid % self.num_videos

            video_keys = list(self.videos.keys())
            video = self.videos[video_keys[rand_vid]]

            # get ids of this video
            video_ids = video[0]
            # how many ids in this video
            video_id_keys = list(video_ids.keys())

            # randomly pick an id for z
            rand_trackid_z = np.random.choice(list(range(len(video_id_keys))))
            # get the video for this id
            video_id_z = video_ids[video_id_keys[rand_trackid_z]]

            # pick a valid examplar z in the video
            rand_z = np.random.choice(range(len(video_id_z)))

            # pick a valid instance within frame_range frames from the examplar, excluding the examplar itself
            possible_x_pos = list(range(len(video_id_z)))
            rand_x = np.random.choice(possible_x_pos[max(rand_z - self.config.pos_pair_range, 0):rand_z] +
                                      possible_x_pos[(rand_z + 1):min(rand_z + self.config.pos_pair_range, len(video_id_z))])

            # use copy() here to avoid changing dictionary
            z = video_id_z[rand_z].copy()
            x = video_id_z[rand_x].copy()

            # read z and x
            img_z = cv2.imread(os.path.join(self.data_dir, z['instance_path']))
            # print(os.path.join(self.data_dir, z['instance_path']))
            img_z = cv2.cvtColor(img_z, cv2.COLOR_BGR2RGB)
            img_x = cv2.imread(os.path.join(self.data_dir, x['instance_path']))
            img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)

            # do data augmentation;
            # note that we have done center crop for z in the data augmentation
            
            # run_Train_SiamFPN里面定义的数据增强
            #img_z = self.z_transforms(img_z)
            #img_x = self.x_transforms(img_x)

            # 自定义的数据增强
            exemplar_img, _, _ = self.RandomStretch(exemplar_img, 0, 0)
            exemplar_img = self.CenterCrop(exemplar_img, )
            exemplar_img = self.z_transforms(exemplar_img)
            instance_img, gt_w, gt_h = self.RandomStretch(instance_img, gt_w, gt_h)
            instance_img, gt_cx, gt_cy = self.RandomCrop(instance_img, )
            instance_img = self.x_transforms(instance_img)

            regression_target, conf_target = self.compute_target(self.anchors, 
                                                                 np.array(list(map(round, [gt_cx, gt_cy, gt_w, gt_h]))))

            if len(np.where(conf_target == 1)[0]) > 0:
                break
            else:
                idx = np.random.randint(self.num)

        #bbox = z["bbox"]
        #bbox_strs = [str(i) for i in bbox]
        #bbox_str = ','.join(bbox_strs)
        #bboxs = [bbox_str]
        #path_z = [z['instance_path']]
        #path_x = [x['instance_path']]
        #return img_z, img_x, bboxs, path_z, path_x
        return exemplar_img, instance_img, regression_target, conf_target.astype(np.int64)
        
    def __getitem__(self, idx):
        while True:
            idx = idx % len(self.video_names)
            video = self.video_names[idx]
            trajs = self.meta_data[video]
            # sample one trajs
            trkid = np.random.choice(list(trajs.keys()))
            traj = trajs[trkid]
            assert len(traj) > 1, "video_name: {}".format(video)
            # sample exemplar
            exemplar_idx = np.random.choice(list(range(len(traj))))
            # exemplar_name = os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid))
            exemplar_name = \
                glob.glob(os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid)))[0]
            exemplar_img = self.imread(exemplar_name)
            # exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)
            # sample instance
            low_idx = max(0, exemplar_idx - config.frame_range)
            up_idx = min(len(traj), exemplar_idx + config.frame_range)

            # create sample weight, if the sample are far away from center
            # the probability being choosen are high
            weights = self._sample_weights(exemplar_idx, low_idx, up_idx, config.sample_type)
            instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx + 1:up_idx], p=weights)
            instance_name = glob.glob(os.path.join(self.data_dir, video, instance + ".{:02d}.x*.jpg".format(trkid)))[0]
            instance_img = self.imread(instance_name)
            # instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
            gt_w, gt_h = float(instance_name.split('_')[-2]), float(instance_name.split('_')[-1][:-4])

            if np.random.rand(1) < config.gray_ratio:
                exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)
                exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)
                instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
                instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)
            exemplar_img, _, _ = self.RandomStretch(exemplar_img, 0, 0)
            exemplar_img = self.CenterCrop(exemplar_img, )
            exemplar_img = self.z_transforms(exemplar_img)
            instance_img, gt_w, gt_h = self.RandomStretch(instance_img, gt_w, gt_h)
            instance_img, gt_cx, gt_cy = self.RandomCrop(instance_img, )
            instance_img = self.x_transforms(instance_img)
            regression_target, conf_target = self.compute_target(self.anchors,
                                                                 np.array(list(map(round, [gt_cx, gt_cy, gt_w, gt_h]))))

            if len(np.where(conf_target == 1)[0]) > 0:
                break
            else:
                idx = np.random.randint(self.num)
        return exemplar_img, instance_img, regression_target, conf_target.astype(np.int64)

    def __len__(self):
        return self.num

    def draw_img(self, img, boxes, name='1.jpg', color=(0, 255, 0)):
        # boxes (x,y,w,h)
        img = img.copy()
        img_ctx = (img.shape[1] - 1) / 2
        img_cty = (img.shape[0] - 1) / 2
        for box in boxes:
            point_1 = img_ctx - box[2] / 2 + box[0], img_cty - box[3] / 2 + box[1]
            point_2 = img_ctx + box[2] / 2 + box[0], img_cty + box[3] / 2 + box[1]
            img = cv2.rectangle(img, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])),
                                color, 2)
        cv2.imwrite(name, img)

    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'):
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = np.array(weights)
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)
    
    def imread(self, path):
        key = hashlib.md5(path.encode()).digest()
        
        db = lmdb.open(self.db_path, readonly=True, map_size=int(1024*1024*1024))
        self.txn = db.begin(write=False)
        img_buffer = self.txn.get(key)

        img_buffer = np.frombuffer(img_buffer, np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        return img


if __name__=="__main__":

    data_dir = ""

    meta_data_path = os.path.join(data_dir, "meta_data.pkl")
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    all_videos = [x[0] for x in meta_data]

    # split train/valid dataset
    train_videos, valid_videos = train_test_split(all_videos,
                                                  test_size=1 - config.train_ratio, random_state=config.seed)

    # define transforms
    train_z_transforms = transforms.Compose([
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # open lmdb
    # db = lmdb.open(data_dir + '.lmdb', readonly=True, map_size=int(1024*1024*1024))
    db_path = data_dir + '.lmdb'
    train_dataset = VIDDataset(db_path, train_videos, data_dir, train_z_transforms, train_x_transforms)
    valid_dataset = VIDDataset(db_path, valid_videos, data_dir, valid_z_transforms, valid_x_transforms,training=False)

