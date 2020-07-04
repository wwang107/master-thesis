from .vis import save_batch_joint_and_keypoint, save_batch_maps

class VisulizationHandler:
    def __init__(self, input_size, output_size, save_file_base_path=None):
        super().__init__()
        self.scale = input_size/output_size
        self.save_file_base_path = save_file_base_path
        
    def save_batch_joints_and_directional_keypoints_plot(self, batch_images, batch_joints, batch_keypoints):
        batch_images_permute = batch_images.permute(0,3,1,2)
        save_batch_joint_and_keypoint(self.save_file_base_path + '/test.png',
                                batch_images_permute, batch_joints, batch_keypoints, self.scale)
    
    def save_batch_heatmaps(self, batch_images, batch_maps, batch_masks):
        batch_images_permute = batch_images.permute(0, 3, 1, 2)
        save_batch_maps(batch_images, batch_maps, batch_masks,
                        self.save_file_base_path + '/test_hm.png')
