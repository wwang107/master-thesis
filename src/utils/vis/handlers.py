from .vis import save_batch_joint_and_keypoint, save_batch_maps
import cv2

class VisulizationHandler:
    def __init__(self, input_size, output_size, save_file_base_path=None):
        super().__init__()
        self.scale = input_size/output_size
        self.save_file_base_path = save_file_base_path
        
    def save_batch_joints_and_directional_keypoints_plot(self, batch_images, batch_joints, batch_keypoints, file_name):
        save_batch_joint_and_keypoint(self.save_file_base_path + '/{}.png'.format(file_name),
                                      batch_images, batch_joints, batch_keypoints, self.scale)

    
    def save_batch_heatmaps(self, batch_images, batch_maps, batch_masks, file_name):
        images = save_batch_maps(batch_images, batch_maps, batch_masks)
        cv2.imwrite(self.save_file_base_path +
                    "/{}.png".format(file_name), grid_image)
