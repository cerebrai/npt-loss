from smodels.model_resnet import getmodel_byname
import torch
import cv2
from skimage import transform as trans
from PIL import Image
import numpy as np

class GetFeats(object):

    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model_name = checkpoint["model_name"]
        num_classes = checkpoint["num_classes"]
        self.transforms = checkpoint["val_trans"]
        model = getmodel_byname(model_name)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint["weight"])
        model.eval()
        if torch.cuda.is_available():
            cuda_flag = True
            print("GPU found Hurray!")
            model.cuda()
        self.model = model
    
    def get_feats(self, img, lm):
        aligned_image = self.alignment_from_insight(img, lm)
        aligned_image_PIL = self.cv2PIL(aligned_image)
        aligned_image_flip = aligned_image_PIL.transpose(Image.FLIP_LEFT_RIGHT)

        img_transformed = self.transforms(aligned_image_PIL)
        img_transformed_flip = self.transforms(aligned_image_flip)
        
        tensor_img = img_transformed.unsqueeze(0).cuda()
        tensor_img_flip = img_transformed_flip.unsqueeze(0).cuda()
        with torch.no_grad():
            feat = self.model.forward(tensor_img)
            feat_flip = self.model.forward(tensor_img_flip)
            feats = torch.cat((feat, feat_flip),1)

        feats = feats.cpu().numpy()
        feats = feats.squeeze()
        feats = feats / (np.linalg.norm(feats))

        return feats

    def alignment_from_insight(self, src_img, src_pts):
        # --------------------------------- ---------------------- ------------------------ #

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
