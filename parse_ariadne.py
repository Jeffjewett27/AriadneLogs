from datetime import datetime, timedelta, timezone
import json
import os
import re

from dateutil import parser
import pandas as pd

def from_json_iso(iso):
    """
    Read JSON ISO 8601 date string into UTC datetime.
    """
    return parser.parse(iso).astimezone(tz=timezone.utc)

def format_name(hitbox_type, name):
    """
    Join hitbox type and name. Strip numbers and parentheses from end of name.
    """
    if len(name) == 0:
        return hitbox_type
    name = re.sub(r'[\s\d\(\)]*$', '', name) 
    name = name.replace(' ', '_')
    return f'{hitbox_type}_{name}'

def parse_room_file(room_file):
    """
    Parse an Ariadne JSON log file.

    Input:
    - room_file: path to the .json file

    Output:
    tuple (
        camera: a dataframe with the timestep as the key and the following columns
            - x: the camera x coordinate
            - y: the camera y coordinate
            - h: the camera half height
            - w: the camera half width
            - time: the real datetime of the snapshot,
        records: dict where keys are timesteps and values are list of entity records
            dict(
                t: list[
                    dict(
                    - eid: any integer index
                    - name: formatted name of the entity (e.g. Enemy_Worm)
                    - hitboxType: the categorization of the hitbox (e.g. Enemy)
                    - x: (optional) the x position
                    - y: (optional) the y position
                    - h: (optional) the half height of the bounding box
                    - w: (optional) the half width of the bounding box
                    - a: (optional) a boolean of whether this hitbox is active (enabled)
                    - segmentationBounds: (optional) the list of list of points on the segmentation bounds (a list of polygons)
                        [ [x1 y1 x2 y2 ... xn yn], ...]
                    )
                ]
            ),
        info: dict(
        - room_file_id: the name of the room_file (without the .json extension)
        - start_time: datetime of room entry
        - end_time: datetime of room exit
        - interval_ms: number of milliseconds between snapshots
        - skipped_frames: a list of [(a,b),...] pairs which are a range of skipped frames (such as from pausing or lag)
        - deviations_ms: the number of milliseconds delayed each frame is from the target. Calculated as the difference between actual_ms - t * interval_ms
        - room_name: the name of the room/scene
        )
    )
    """
    # Load the JSON file
    with open(room_file, 'r') as f:
        data = json.load(f)

    room_basename = os.path.basename(room_file)
    room_id = room_basename[:room_basename.rfind('_')]
    start_time = from_json_iso(data['entryTime'])
    end_time = from_json_iso(data['exitTime'])
    interval_ms = data['intervalMs'] if 'intervalMs' in data else 20

    skipped_frames_raw = data['skippedFrames'] if 'skippedFrames' in data else []
    skipped_frames = list(zip(skipped_frames_raw[::2], skipped_frames_raw[1::2]))
    deviations = data['deviationsMs'] if 'deviationsMs' in data else []

    info = {
        'room_file_id': room_basename.replace('.json',''),
        'start_time': start_time,
        'end_time': end_time,
        'interval_ms': interval_ms,
        'skipped_frames': skipped_frames,
        'deviations_ms': deviations,
        'room_name': room_id
    }

    records = {}
    entity_id = 0
    for category in data['categories']:
        hitbox_type = category['hitboxType']
        for entity in category['entities']:
            name = entity['name']
            name = format_name(hitbox_type, name)
            for record in entity['records']:
                record['hitboxType'] = hitbox_type
                record['name'] = name
                record['eid'] = entity_id
                if 'segmentationBounds' in entity:
                    record['segmentationBounds'] = entity['segmentationBounds']
                if 't' not in record:
                    record['t'] = 0
                t = int(record['t'])
                if t not in records:
                    records[t] = []
                records[t].append(record)
            entity_id += 1

    # Get a pandas DataFrame of camera positions
    camera_data = pd.DataFrame(data['camera'])
    camera_data.fillna(method='ffill', inplace=True)
    camera_data['t'] = camera_data['t'].fillna(0).astype(int)
    camera_data['time'] = start_time + pd.to_timedelta(camera_data['t'] * interval_ms, unit='ms')
    return camera_data, records, info

def generate_time_snapshots(camera_df: pd.DataFrame, records: dict, info: dict, min_t=0):
    """
    Generate snapshots of timesteps and forward fill missing values.

    Inputs:
    - camera_df: the camera dataframe from parse_room_file
    - records: the records dict from parse_room_file
    - info: the info dict from parse_room_file
    - min_t: (optional) the starting timestep

    Yields:
    tuple (
        - t: timestep,
        - time: datetime,
        - camera: dataframe row at timestep,
        - current_entities: dict with eid as the key and entity record (as in records). Missing values are forward filled.
    )
    """
    t = 0
    start_time: datetime = info['start_time']
    interval_ms: float = info['interval_ms']
    deviations_ms: int = info['deviations_ms']
    deviations_idx: int = 0
    deviation = 0

    time_grouped_camera = camera_df.set_index('t')

    current_entities = {}
    camera = camera_df.iloc[0]

    max_t = max(records.keys())

    t = 0
    while t < max_t:
        if t in records.keys():
            for entity in records[t]:
                eid = entity['eid']
                if eid in current_entities:
                    current_entities[eid].update(entity)
                else:
                    current_entities[eid] = entity.copy()

        if t in time_grouped_camera.index:
            camera = time_grouped_camera.loc[t]

        time = start_time + timedelta(milliseconds=t * interval_ms + deviation)

        if t >= min_t:
            yield int(t), time, camera, current_entities.copy()

        if deviations_idx < len(deviations_ms):
            deviation = deviations_ms[deviations_idx]
            # if it deviates more than an interval, skip frame(s)
            t += deviation // interval_ms
            deviation = deviation % interval_ms
            deviations_idx += 1

        t += 1
    
if __name__ == "__main__":
    # example
    cams, entities, info = parse_room_file('assets/Crossroads_47_46817628.json')
    i = 0
    print(info)
    for t, time, cur_cam, cur_entities in generate_time_snapshots(cams, entities, info, 0):
        # cur_cam shows the current camera info
        # cur_entities shows the current entities' snapshots
        pass