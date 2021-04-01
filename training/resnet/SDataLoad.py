"""
Mostly taken from ammarah
Custom data loaders
"""
from torch.utils.data.dataset import Dataset
from PIL import Image
from scipy.io import loadmat
import numpy as np
import cv2
from skimage import transform as trans
# -----  Custom Data Loader Class ------ #
class DataLoad(Dataset):
    def __init__(self, csv_file_path, transforms=None, matfile_path=None):

        data = pd.read_csv(csv_file_path, header=None)
        self.img_paths = np.asarray(data.iloc[:, 0])
        self.labels = np.asarray(data.iloc[:, 1])
        self.transforms = transforms


        self.matfile_path = matfile_path
        if self.matfile_path is not None:
            self.data = loadmat(matfile_path)

        #self.mtcnn_data_hr = loadmat("/vol/vssp/facer2vm/data/Casia-Web-From-insight-face/Casia_from_awais/alignment_data/data_1/mtcnn.mat")
        #self.mtcnn_data_lr_20 = loadmat("/vol/vssp/facer2vm/data/Casia-Web-From-insight-face/Casia_from_awais/alignment_data_lr/resized_20_raw_bbox.mat")
        #self.mtcnn_data_lr_16 = loadmat("/vol/vssp/facer2vm/data/Casia-Web-From-insight-face/Casia_from_awais/alignment_data_lr/resized_16_raw_bbox.mat")

        #self.landmarks_hr = loadmat("/vol/vssp/facer2vm/data/Casia-Web-From-insight-face/Casia_from_awais/alignment_data/data_1/zhen_wide-10-0.mat")
        #self.landmarks_lr_20 = loadmat("/vol/vssp/facer2vm/data/Casia-Web-From-insight-face/Casia_from_awais/alignment_data_lr/resized_20_landmarks.mat")
        #self.landmarks_lr_16 = loadmat("/vol/vssp/facer2vm/data/Casia-Web-From-insight-face/Casia_from_awais/alignment_data_lr/resized_16_landmarks.mat")

    def __getitem__(self, index):

        current_path = self.img_paths[index]
        current_label = self.labels[index]
        pudb.set_trace()
        if self.matfile_path is not None:
            img_as_img = Image.fromarray(self.data[current_path]).convert('RGB')
        else:
            img_as_img = Image.open(current_path).convert('RGB')

        if self.transforms is not None:
            img_transformed_hr = self.transforms(img_as_img)

        # img_as_img_extra = self.get_extra(current_path, "hr")
        #
        # if self.transforms is not None:
        #     img_transformed_hr2 = self.transforms(img_as_img_extra)

        # --  making return dictionary
        data = dict()
        data["img"] = img_transformed_hr
        data["label"] = current_label

        return data

    def key_transform(self, key, patch=20):
        split_key = key.split("/")
        if patch == 20:
            split_key[7] = "aligned_lr_20"
        else:
            split_key[7] = "aligned_lr_16"
        key = "/".join(split_key)
        return key

    def get_extra(self, key, type):
        if type == "hr":
            key = "/vol/vssp/datasets/still02/CASIA-WebFace/CASIA-WebFace/" + "/".join(key.split("/")[-2:])
            img = cv2.imread(key)
            bbox = self.mtcnn_data_hr[key]
            keypoints = self.landmarks_hr[key]
        elif type == "lr_20":
            key = "/vol/vssp/facer2vm/data/Casia-Web-From-insight-face/Casia_from_awais/resized_20_raw/" + "/".join(key.split("/")[-2:])
            img = cv2.imread(key)
            bbox = self.mtcnn_data_lr_20[key]
            keypoints = self.landmarks_lr_20[key]
        else:
            key = "/vol/vssp/facer2vm/data/Casia-Web-From-insight-face/Casia_from_awais/resized_16_raw/" + "/".join(key.split("/")[-2:])
            img = cv2.imread(key)
            bbox = self.mtcnn_data_lr_16[key]
            keypoints = self.landmarks_lr_16[key]

        crop_img = self.imgcrop(bbox.squeeze().astype(int), img=img)

        # show_img = crop_img
        # for points in keypoints:
        #     cv2.circle(show_img, (int(points[0]), int(points[1])), 1, (0, 0, 255), 1, 8, 0)
        #
        # self.cv2PIL(show_img).show()

        aligned_image = self.alignment_from_insight(crop_img, keypoints)

        return self.cv2PIL(aligned_image)

    def imgcrop(self, bbox, img=None, percent_scale=10):

        if percent_scale == 0:
            return img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        else:
            (startX, startY, endX, endY) = bbox
            fr_width = endX - startX
            fr_height = endY - startY

            new_fr_width = fr_width * (1 + percent_scale / 100)
            new_fr_height = fr_height * (1 + percent_scale / 100)

            diff_width = int((new_fr_width - fr_width) / 2)
            diff_height = int((new_fr_height - fr_height) / 2)

            startX = startX - diff_width
            endX = endX + diff_width

            startY = startY - diff_height
            endY = endY + diff_height

            fr_width = endX - startX
            fr_height = endY - startY

            crop_img = img[max(0, startY):min(startY + fr_height, img.shape[0]),
                       max(0, startX):min(startX + fr_width, img.shape[1])]

            return crop_img

    def alignment_from_insight(self, src_img, src_pts):
        # --------------------------------- ---------------------- ------------------------ #
        #  Alignment as given by Ho to Junaid. Requires  input img and 5 facial points      #
        #  In its original form, it defines set of reference points and the facial points   #
        #  are supposed to be transformed into those reference points. These reference      #
        #  are for a 96 cross 112 images. So for each crop size, I scale these references   #
        #  accordingly.

        # -- ref points according to 96, 112 imge size
        ref_pts_x = np.array([30.2946, 65.5318, 48.0252, 33.5493, 62.7299])
        ref_pts_y = np.array([51.6963, 51.5014, 71.7366, 92.3655, 92.2041])

        ref_pts_x = ref_pts_x + 8.0

        # -- rescaling reference points according to given crop_size

        # img_width = src_img.shape[1]
        # img_height = src_img.shape[0]
        # fac_x = img_width / 96
        # ref_pts_x = (ref_pts_x * fac_x)
        # fac_y = img_height / 112
        # ref_pts_y = (ref_pts_y * fac_y)
        # ---------------------------------------------------------
        ref_pts = [list(val) for val in zip(ref_pts_x, ref_pts_y)]
        src_pts = np.array(src_pts).reshape(5, 2)
        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(s, r)
        M = tform.params[0:2, :]

        trans_img = cv2.warpAffine(src_img, M, (112, 112), borderValue=0.0)

        return trans_img

    def cv2PIL(self, img):
        b, g, r = cv2.split(img)  # get b,g,r
        rgb_img = cv2.merge([r, g, b])
        pil_image = Image.fromarray(rgb_img)
        return pil_image

    def __len__(self):
        return len(self.img_paths)

