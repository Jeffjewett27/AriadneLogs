# process Ariadne output into COCO dataset

from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import chain
import json
import os
from typing import List, Tuple

import cv2 
import numpy as np

from parse_ariadne import parse_room_file, generate_time_snapshots
import pyclipr

category_colors = {
    'Knight': (0, 255, 255),
    'Enemy': (0, 0, 200),
    'Terrain': (255, 255, 255),
    'Breakable': (200, 200, 255),
    'Transition': (128, 0, 0),
    'GeoToken': (50, 150, 150),
    'Grass': (128, 220, 128),
    'HotSpring': (225, 180, 180),
    'DreamNail': (200, 200, 200),
    'Bench': (40, 40, 40)
}

class ScreenTransform:
    """
    Transforms from world space into camera space.
    """

    def __init__(self, screen_width: int, screen_height: int, cam_x: float, cam_y: float, 
            cam_width: float, cam_height: float):

        self.screen_width: int = screen_width
        self.screen_height: int = screen_height
        self.cam_x: float = cam_x
        self.cam_y: float = cam_y
        self.cam_height: float = cam_height
        self.cam_width: float = cam_width

        self.w_factor: float = screen_width / cam_width
        self.h_factor: float = screen_height / cam_height

        # a Clipper2 lib path to bound the camera.
        self.clip_path = np.array([(0,0),(0,self.screen_height),(self.screen_width,self.screen_height),
            (self.screen_width,0),(0,0)])

    def transform_xy(self, world_X: float, world_y: float) -> Tuple[int, int]:
        """Transform world coordinates to screen space"""
        return int((world_X - self.cam_x) * self.w_factor + self.screen_width / 2), \
            int((self.cam_y - world_y) * self.h_factor + self.screen_height / 2)
    
    def transform_wh(self, world_w: float, world_h: float) -> Tuple[int, int]:
        """Transform world width and height to screen space"""
        return int(world_w * self.w_factor), int(world_h * self.h_factor)
    
    def clip_to_img_bounds(self, x: int, y: int, w: int, h: int) -> Tuple[bool, int, int, int, int]:
        """
        Clip a screen-space bounding box to coordinates within the screen bounds

        Input:
        - x: bounding box x (screen coordinates)
        - y: bounding box y (screen coordinates)
        - w: bounding box half width (screen coordinates)
        - h: bounding box half height (screen coordinates)

        Output:
        tuple(
        - is_valid: if the area is positive,
        - x1: top left x (int),
        - y1: top left y (int),
        - x2: bottom right x (int),
        - y2: bottom right y (int)
        )
        """
        x1 = max(x - w, 0)
        x2 = min(x + w, self.screen_width - 1)
        y1 = max(y - h, 0)
        y2 = min(y + h, self.screen_height - 1)
        is_valid = x1 < x2 and y1 < y2
        return is_valid, x1, y1, x2, y2

    def clip_segmentations(self, segmentations: List[List[float]]):
        """
        Uses Clipper2 library to clip segmentation path to within the screen bounds

        Inputs:
        - segmentations: list of list of segmentation points in screen space [[(x1,y1), (x2,y2), ...], ...]

        Outputs:
        tuple(
        - clipped segmentations: list of list of segmentation points clipped to the screen boundaries
        - area: area of clipped segmentations
        - bounding box: tuple(
            - x: top left x
            - y: top left y
            - w: full width
            - h: full height
            )
        )
        """
        np_segment = [np.array(path) for path in segmentations]

        pc = pyclipr.Clipper()
        pc.scaleFactor = int(1000)
        pc.addPaths(np_segment, pyclipr.Subject)
        pc.addPath(self.clip_path, pyclipr.Clip)
        # intersect the segmentation with the screen bounds
        clipped = pc.execute(pyclipr.Intersection, pyclipr.FillRule.EvenOdd)
        clipped_polytree = pc.execute2(pyclipr.Intersection, pyclipr.FillRule.EvenOdd)
        area = clipped_polytree.area
        
        min_x = min_y = float('inf')
        max_x = max_y = -float('inf')
        def format_path(path):
            nonlocal min_x, min_y, max_x, max_y
            path = [tuple(point) for point in path.astype(int).tolist()]
            if path[0] != path[-1]:
                path.append(path[0])
            for point in path:
                min_x = min(min_x, point[0])
                min_y = min(min_y, point[1])
                max_x = max(max_x, point[0])
                max_y = max(max_y, point[1])
            return path
        return [format_path(path) for path in clipped], area, (min_x, min_y, max_x - min_x, max_y - min_y)

class RealtimeVideo:

    def __init__(self, video_path, shape=None, offset_ms=0):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)

        if shape is None:
            self.screen_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.screen_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resize = False
        else:
            self.screen_width = shape[0]
            self.screen_height = shape[1]
            self.resize = True

        # Get the video creation time
        creation_time_os = os.path.getctime(video_path)
        self.video_start_time = datetime.fromtimestamp(creation_time_os, tz=timezone.utc) + timedelta(milliseconds=offset_ms)

        # Get the frame rate of the video
        self.video_frame_rate = self.capture.get(cv2.CAP_PROP_FPS)

        # Get the total duration of the video in milliseconds
        # self.video_duration_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC)
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration_ms = frame_count / self.video_frame_rate * 1000
        self.video_end_time = self.video_start_time + timedelta(milliseconds=self.video_duration_ms)
    
    def get_frame(self, t: int, timestamp: datetime = None):
        if timestamp < self.video_start_time:
            raise Exception(f'Cannot get frame at timestamp {timestamp} before {self.video_end_time}')
        if timestamp > self.video_end_time:
            raise Exception(f'Cannot get frame at timestamp {timestamp} that exceeds {self.video_end_time}')
        video_ms = (timestamp - self.video_start_time) / timedelta(milliseconds=1)
        self.capture.set(cv2.CAP_PROP_POS_MSEC, video_ms)
        ret, frame = self.capture.read()
        if not ret:
            return False, None
        if self.resize:
            frame = cv2.resize(frame, (self.screen_width, self.screen_height))
        return True, frame
    
class MockVideo:

    def __init__(self, width, height):
        self.screen_width = width
        self.screen_height = height

    def get_frame(self, t: int, timestamp: datetime = None):
        return f"mock_image_{t}", np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
    
class UnityFrames:
    """
    Extracts screenshots from Ariadne logs
    """

    def __init__(self, screen_height: int, log_dir: str, room_file: str):
        """
        Inputs:
        - screen_height: height of the frame to be resized. Aspect ratio will be maintained
        - log_dir: directory where Ariadne logs are stored
        - room_file: path to the Ariadne .json log file for the room/scene
        """
        self.screen_height = screen_height
        self.log_dir = log_dir

        self.room_file_id = os.path.basename(room_file).replace('.json', '')

    def get_frame(self, t: int):
        """
        Attempts to get the frame from the logs at the timestep.

        Inputs:
        - t (int): the timestep of the image to fetch

        Outputs:
        tuple (
        - path: the path of the retrieved files (or None if not found)
        - frame: the retrieved image, resized to specified height
        )
        """
        path = f"{self.log_dir}/images/{self.room_file_id}_{t}.png"
        if not os.path.exists(path):
            print(f'{path} does not exist')
            return None, None
        img = cv2.imread(path)
        width = int(img.shape[1] * self.screen_height / img.shape[0])
        return path, cv2.resize(img, (width, self.screen_height))
    
class IdManager:
    """Creates an autoincrementing id mapping."""

    def __init__(self):
        self.id_dict = {}
        self.next_id = 0

    def get_id(self, obj):
        """Try to fetch the id for the object, or autoincrement a new id"""
        if obj not in self.id_dict:
            self.id_dict[obj] = self.next_id
            self.next_id += 1
        return self.id_dict[obj]

def generate_entity_annotations(room_file: str, frame_fetcher: UnityFrames, category_id_manager: IdManager):
    """
    Generate object annotations for Ariadne snapshots.

    Args:
    - room_file (str): path to a .json file output by Ariadne
    - frame_fetcher (UnityFrames): gets the frames from the logs
    - category_id_manager (IdManager): track the category ids

    Yields:
    tuple (
    - frame (image): the frame corresponding to the annotations
    - coco_image: the image metadata (path, height, width, etc)
    - coco_objects: list of COCO annotations for the frame
    )
    """

    camera_df, entities_df, info = parse_room_file(room_file)

    entity_id_manager = IdManager()
    image_id_manager = IdManager()

    for timestep, timestamp, camera, entities in generate_time_snapshots(camera_df, entities_df, info):
        
        frame_path, frame = frame_fetcher.get_frame(timestep)
        if frame_path is None:
            # frame was not found in the logs
            continue
        frame_shape = frame.shape
        frame_path = f'{info["room_name"]}/{info["room_file_id"]}_{timestep}.jpg'

        image_id = image_id_manager.get_id(frame_path)
        coco_image = {
            'id': image_id,
            'file_name': frame_path,
            'height': frame_shape[0],
            'width': frame_shape[1],
            'timestamp': timestamp.isoformat(),
            'room_name': info['room_name']
        }

        cam_width = camera['w']
        cam_height = camera['h']
        cam_x = camera['x']
        cam_y = camera['y']
        screen_transform = ScreenTransform(frame_shape[1], frame_shape[0], cam_x, cam_y, cam_width, cam_height)

        coco_objects = []
        for entity in entities.values():
            x, y = screen_transform.transform_xy(entity['x'], entity['y'])
            w, h = screen_transform.transform_wh(entity['w'], entity['h'])
            is_valid, x1, y1, x2, y2 = screen_transform.clip_to_img_bounds(x, y, w, h)
            if not is_valid:
                continue
            box_w = x2 - x1
            box_h = y2 - y1

            # etype = entity['hitboxType']
            # color = category_colors[etype] if etype in category_colors else (0,0,0)

            category_id = category_id_manager.get_id(entity['name'])
            object_id = entity_id_manager.get_id(entity['eid'])
            coco_object = {
                'id': object_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [x1, y1, box_w, box_h],
                'area': box_w * box_h,
                'is_crowd': 0
            }

            segmentation = None
            if 'segmentationBounds' in entity:
                segmentation = entity['segmentationBounds']
                segmentation, area, bbox = transform_and_crop_segmentation(segmentation, screen_transform)
                if len(segmentation) > 0:
                    coco_object['segmentation'] = segmentation
                    coco_object['area'] = area
                    coco_object['bbox'] = bbox

            coco_objects.append(coco_object)
                
            # frame_draw_entity(frame, (x1, y1, x2, y2), color, segmentation=segmentation)
            
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break

        yield frame, coco_image, coco_objects

def transform_and_crop_segmentation(segmentation: List[List[float]], screen_transform: ScreenTransform):
    """
    Transforms segmentations to screen space.

    Inputs:
    - segmentation: list of list of flattened segmentation points [[x1, y1, x2, y2, ...], ...]
    - screen_transform: the ScreenTransform to map coordinates to screen space

    Outputs:
    tuple(
    - transformed segmentation,
    - area of the transformed segmentation,
    - bounding box of the transformed segmentation
    )
    """
    def transform_path(path):
        # start with a path [x1, y1, x2, y2, ...]
        # then transform to [(x1,y1),(x2,y2)]
        points = zip(path[0::2], path[1::2])
        # then transform to screen space [(x1',y1'),(x2',y2')]
        new_points = [screen_transform.transform_xy(*point) for point in points]
        # flatten points back to path [x1', y1', x2', y2', ...]
        return new_points
    
    transformed_segmentation = [transform_path(path) for path in segmentation]
    clipped_segmentation, area, bbox = screen_transform.clip_segmentations(transformed_segmentation)

    return [list(chain.from_iterable(path)) for path in clipped_segmentation], area, bbox

# TODO remove drawing code
def frame_draw_entity(frame, bounds, color, segmentation=None):
    x1, y1, x2, y2 = bounds
    if segmentation is None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    else:
        for path in segmentation:
            points = list(zip(path[0::2], path[1::2]))
            for point1, point2 in zip(points[:-1], points[1:]):
                cv2.line(frame, point1, point2, color, 1)

    
def list_log_files(log_dir, start_log=None):
    min_time = 0
    if start_log is not None:
        log = os.path.join(log_dir, start_log)
        min_time = max(min_time, os.path.getctime(log))
    # Get list of files in the log directory
    files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))]

    files_with_ctime = [(os.path.getctime(file), file) for file in files]
    files_with_ctime = list(filter(lambda file: min_time <= file[0], files_with_ctime))
    files_with_ctime.sort()

    return [file for _, file in files_with_ctime]

def process_ariadne_logs(log_dir, dataset_dir):
    """
    Processes new logs into the dataset. Previously processed logs will not be processed again
    """
    log_files = list_log_files(log_dir)

    already_processed = set()
    partial_path = f'{dataset_dir}/annotations/partial'
    if os.path.exists(partial_path):
        partials = os.listdir(partial_path)
        already_processed = set(partials)

    for room_file in log_files:
        if os.path.basename(room_file) in already_processed:
            continue

        process_single_ariadne_log(log_dir, dataset_dir, room_file)

def process_single_ariadne_log(log_dir, dataset_dir, room_file):
    video = UnityFrames(480, log_dir, room_file)
    room_file_id = os.path.basename(room_file).replace('.json','')
    # ttv = train_test_or_validate(room_file_id)

    category_id_manager = IdManager()
    coco_images = []
    coco_annotations = []

    for image, coco_image, object_annotations in generate_entity_annotations(room_file, video, category_id_manager):
        coco_images.append(coco_image)
        coco_annotations = [*coco_annotations, *object_annotations]

        file_name = coco_image['file_name']
        image_path = f'{dataset_dir}/data/{file_name}'
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        cv2.imwrite(image_path, image)

    coco_categories = [{
        'id': id,
        'name': name
    } for name, id in category_id_manager.id_dict.items()]

    now = datetime.now()
    coco_info = {
        'year': now.year,
        'version': '1',
        'description': f'A COCO JSON annotated file corresponding to the Ariadne-generated {room_file_id}.json',
        'contributors': "FuzzyJeffTheory",
        'source_log': room_file_id,
        'date_created': now.isoformat()
    }

    coco = {
        'info': coco_info,
        'categories': coco_categories,
        'images': coco_images,
        'annotations': coco_annotations
    }

    annotation_dir = f'{dataset_dir}/annotations/partial'
    os.makedirs(annotation_dir, exist_ok=True)
    annotation_file = f'{annotation_dir}/{room_file_id}.json'
    try:
        with open(annotation_file, 'w') as ann_file:
            json.dump(coco, ann_file, indent=2)
        print(f'Successfully processed {annotation_file}')
    except:
        print(f'Failed to process {annotation_file}')

# TODO remove
def train_test_or_validate(file_name):
    validation_weight = 10
    test_weight = 10
    file_hash = hashlib.sha256(file_name.encode()).hexdigest()
    val = int(file_hash, 16) % 100
    if val < test_weight:
        return 'test'
    if val < test_weight + validation_weight:
        return 'val'
    return 'data'
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('logdir', type=str, help="The directory of the Ariadne logs")
    parser.add_argument('--dataset', '-d', default='dataset', type=str, required=False, help="The output dataset directory")
    config = parser.parse_args()

    process_ariadne_logs(config.logdir, config.dataset)
