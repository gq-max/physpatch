import os
import shutil
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval', dataroot='/nuScenes', verbose=True)

output_dir = './data/clean'
os.makedirs(output_dir, exist_ok=True)

camera_channel = 'CAM_FRONT' 


for i, scene in enumerate(nusc.scene):
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)

    cam_token = sample['data'][camera_channel]
    cam_sample_data = nusc.get('sample_data', cam_token)

    src_path = os.path.join(nusc.dataroot, cam_sample_data['filename'])

    filename = os.path.basename(cam_sample_data['filename'])
    dst_path = os.path.join(output_dir, filename)

    shutil.copy(src_path, dst_path)
