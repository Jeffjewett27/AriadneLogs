# AriadneLogs

Ariadne (https://github.com/Jeffjewett27/Ariadne) is a Hollow Knight mod that logs hitboxes and takes screenshots. This repository contains scripts to turn these into an object detection dataset

Ariadne saves each room/scene into individual JSON log files. To create COCO JSON (https://roboflow.com/formats/coco-json) annotations from these log files, run `process_dataset.py`
```
usage: process_dataset.py [-h] [--dataset DATASET] logdir

positional arguments:
  logdir                The directory of the Ariadne logs

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        The output dataset directory
```

Example:
`python process_dataset.py "C:/Users/User/Documents/HollowKnight/logs/25d1cc" -d ./dataset`

This will create a bunch of partial annotation files in `dataset/annotations/partial`. This can be combined with `coco_merge.py`:
```
usage: coco_merge.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        The output dataset directory
```
Example:
`python coco_merge.py -d ./dataset`

This will aggregate all the annotations into `dataset/annotations/labels.json`

An example of an Ariadne log is at `examples/Crossroads_47_46817628.json`, and the corresponding output annotations is at `examples/Partial_Annotation_Crossroads_47_46817628.json`