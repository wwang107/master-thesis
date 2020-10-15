import cv2
import pathlib
from tqdm import tqdm

IMG_FOLDERS = [
    # '160224_haggling1/hdImgs',
    #'160226_haggling1/hdImgs', 
    '170407_haggling_a1/hdImgs'
    ]
ROOT_DIR = '/media/weiwang/Elements/panoptic'

if __name__ == "__main__":
    broken_imgs = []
    root = pathlib.Path(ROOT_DIR)
    for img_folder in IMG_FOLDERS:
        cam_ids = sorted([cam_id for cam_id in root.joinpath(img_folder).iterdir() if cam_id.is_dir()])
        for i,cam_id in enumerate(cam_ids):
            imgs = list(cam_id.glob('*.jpg'))
            for img in tqdm(imgs, desc='seq_{}_cam_{}'.format(img_folder, i)):
                try:
                    file = cv2.imread(str(img))
                    if file.shape[0] < 0 or file.shape[1]<0:
                        print(img)
                        broken_imgs.append(img)
                except AttributeError as identifier:
                    broken_imgs.append(img)
                    print(img)
    print(broken_imgs)
