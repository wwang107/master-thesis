
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# ------------------------------------------------------------------------------
# dataset dependent configuration for visualization
lcolor = (152, 52, 219)
rcolor = (76, 231, 60)

coco_part_labels = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
    'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r'
]
coco_part_idx = {
    b: a for a, b in enumerate(coco_part_labels)
} # e.g. {'nose':1, 'eye_l':2, ...}
coco_part_orders = [
    ('nose', 'eye_l'), ('eye_l', 'eye_r'), ('eye_r', 'nose'),
    ('eye_l', 'ear_l'), ('eye_r', 'ear_r'), ('ear_l', 'sho_l'),
    ('ear_r', 'sho_r'), ('sho_l', 'sho_r'), ('sho_l', 'hip_l'),
    ('sho_r', 'hip_r'), ('hip_l', 'hip_r'), ('sho_l', 'elb_l'),
    ('elb_l', 'wri_l'), ('sho_r', 'elb_r'), ('elb_r', 'wri_r'),
    ('hip_l', 'kne_l'), ('kne_l', 'ank_l'), ('hip_r', 'kne_r'),
    ('kne_r', 'ank_r')
]

coco_part_colors = {'nose': lcolor, 'eye_l': lcolor, 'eye_r': rcolor, 'ear_l':lcolor, 'ear_r':rcolor,
                    'sho_l':lcolor, 'sho_r':rcolor, 'elb_l':lcolor, 'elb_r':rcolor, 'wri_l':lcolor, 'wri_r':rcolor,
                    'hip_l':lcolor, 'hip_r':rcolor, 'kne_l':lcolor, 'kne_r':rcolor, 'ank_l':lcolor, 'ank_r':rcolor}

coco_idx_color = {idx: coco_part_colors[part] for idx, part in enumerate(coco_part_labels)}

VIS_CONFIG = {
    'COCO': {
        'part_labels': coco_part_labels,
        'part_idx': coco_part_idx,
        'part_orders': coco_part_orders,
        'idx_color':coco_idx_color
    }
}
