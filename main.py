from argparse import ArgumentParser
from YFYF.Alignment.Detection.DlibDetector import DlibDetector
from YFYF.IO.Video import make_long_video
from skimage import io, transform
import skvideo.io
from tqdm import tqdm
from pathlib import Path
import imageio
import numpy as np
import matplotlib.pyplot as plt
import threading
from queue import Queue
from time import sleep
parser = ArgumentParser('Superpose face on another one')
parser.add_argument('--image', type=str, default='bharat.png', help='path to the face image to impose')
parser.add_argument('--video', type=str, default='bitch_lasagna.mp4',help='path to the video')
parser.add_argument('--output', type=str, default='bitch_lasagna_faced.mp4',help='path to the output video')
parser.add_argument('--tmp_dir', type=str, default='tmp', help='temperate directory to store all images')
parser.add_argument('--which_face', type=str, default='all', choices=['all', 'largest'], help='when there are multiple faces, choose one to superpose')
parser.add_argument('--start_from', type=int, default=0, help='Starting from frame')
parser.add_argument('--end_at', type=int, default=-1, help='Starting from frame')
parser.add_argument('--face_scale', type=float, default=1.5, help='Starting from frame')
parser.add_argument('--rotate', action='store_true', help='also match the rotation of the other face, provides better stability, takes longer to compute')
opt = parser.parse_args()
tmp_dir = Path(opt.tmp_dir)
tmp_dir.mkdir(exist_ok=True, parents=True)
image = io.imread(opt.image)
detector = DlibDetector()
if image.shape[-1] == 4:
    mask = image[..., 3].copy()
    image = image[..., :-1]
else:
    mask = None
src_face = image
src_face_size = src_face.shape[1]
reader = imageio.get_reader(opt.video)
frame_paths = []
video_length = reader.get_length()
if opt.end_at == -1:
    opt.end_at = video_length
assert video_length >= opt.end_at
dets = [0]
img_queue = Queue()
def save_img_thread():
    """Save imgs as they come in"""
    total_frame_count = opt.end_at - opt.start_from
    n_digits = len(str(total_frame_count))
    index = 0
    queue_clear = img_queue.empty()
    all_paths = []
    while (not queue_clear) or index < total_frame_count-1: # only stop if img_queue is clear and should stop is True
        if not queue_clear:
            img = img_queue.get()
            file_idx = str(index).rjust(n_digits, '0')
            filename = f'img_{file_idx}.jpg'
            path = tmp_dir / filename
            all_paths.append(path)
            io.imsave(path, img)
            index += 1
        else:
            # print('Sleeping')
            sleep(1)
            # print('Awake')
        queue_clear = img_queue.empty()
    make_long_video(all_paths, 30, format='MPEG', outvid=opt.output)
img_thread = threading.Thread(target=save_img_thread)
img_thread.start()

for i in tqdm(range(opt.start_from, opt.end_at)):
    frame = reader.get_data(i)
    _dets = detector.detect(frame)
    if len(_dets) > 0:
        dets = _dets
    frame = frame.copy()/255
    if opt.which_face == 'largest':
        #TODO: filter the faces
        pass
    for det in dets:
        left, top, right, bottom = det
        centroid = ((bottom + top)/2, (right+left)/2)
        face_size = right - left
        scaled_src_face = transform.rescale(src_face, scale=face_size/src_face_size*opt.face_scale, mode='constant', anti_aliasing=True)
        tar_top, tar_left = centroid[0] - scaled_src_face.shape[0]//2, centroid[1] - scaled_src_face.shape[1]//2
        tar_top, tar_left = map(int, (max(tar_top, 0), max(tar_left, 0)))
        if mask is not None:
            scaled_mask = transform.rescale(mask, scale=face_size/src_face_size*opt.face_scale, preserve_range=True, mode='constant', anti_aliasing=True).astype(np.uint8)
            # print(mask.max())
            background = frame[tar_top:tar_top+scaled_src_face.shape[0], tar_left:tar_left+scaled_src_face.shape[1]].copy()# = scaled_src_face
            background[scaled_mask > 0] = scaled_src_face[scaled_mask > 0]
            frame[tar_top:tar_top+scaled_src_face.shape[0], tar_left:tar_left+scaled_src_face.shape[1]] = background
        else:
            frame[tar_top:tar_top+scaled_src_face.shape[0], tar_left:tar_left+scaled_src_face.shape[1]] = scaled_src_face
    img_queue.put(frame)

img_thread.join()



