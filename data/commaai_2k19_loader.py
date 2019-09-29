import glob
import cv2
import numpy as np
import os
import hashlib

class CommaaiLoader(object):
    def __init__(self,
                 dataset_dir,
                 img_height=256,
                 img_width=832):

        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.scenes = self.collect_train_folders()

    # returns a list of the python directory objects for each sequence
    def collect_train_folders(self):
        segment_dirs = glob.glob(self.dataset_dir + "/*/*/*")
        return segment_dirs

    # returns a list of scence data objects for the given sequence
    # a sequence data object has the following fields:
    #   instrinsics   a numpy array of the camera instrinsics for the dataset
    #   dir           the python directory object for the sequence
    #   rel_path      the relative path of the sequence for writing the data to the output folder
    def collect_scenes(self, sequence):
        intrinsics = np.asarray([[910, 0, 582],
        [0, 910, 437],
        [0,   0,   1]]).astype(np.float32)
        zoom_y = self.img_height/874
        zoom_x = self.img_width/1164
        intrinsics[0,:] *=  zoom_x
        intrinsics[1,:] *=  zoom_y

        return [{'intrinsics': intrinsics, 'dir': sequence, 'rel_path': hashlib.sha224(sequence.encode('utf-8')).hexdigest()[:20] + "_1"}]

    # yields sample objects for the scene
    # a sample object has the following fields:
    #   img           a numpy array containing the frame
    #   id            the number of the frame
    def get_scene_imgs(self, scene_data):
        cam = cv2.VideoCapture(scene_data['dir'] + "/video.hevc")
        frame_velocities =  np.linalg.norm(np.load(scene_data['dir'] + '/global_pose/frame_velocities'),axis=1)
        frame_velocities = list(map(lambda x: x.item(), frame_velocities))

        current_frame = 0
        while(True):
            ret, frame = cam.read()
            if ret:
                current_frame += 1
                img = cv2.resize(frame, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
                img = np.asarray(img)

                yield {"img": img, "id": current_frame, "pose": np.array(frame_velocities[current_frame-1])}
            else:
                break