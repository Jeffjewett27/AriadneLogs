# from https://gradiant.github.io/pyodi/reference/apps/coco-merge/

from argparse import ArgumentParser
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

def read_coco(file_name):
    try:
        with open(file_name, "r") as f:
            return json.load(f)
    except:
        print(f'Failed to load JSON from {file_name}')
        return None

def coco_merge(
    coco_files: List[str], output_file: str, indent: Optional[int] = None, version: int = 1
) -> str:
    """Merge COCO annotation files.

    Args:
        input_extend: Path to input file to be extended.
        input_add: Path to input file to be added.
        output_file : Path to output file with merged annotations.
        indent: Argument passed to `json.dump`. See https://docs.python.org/3/library/json.html#json.dump.
    """
    assert len(coco_files) > 0

    data_adds = []
    for input_add in coco_files:
        if isinstance(input_add, dict):
            coco_data = input_add
        else:
            coco_data = read_coco(input_add)
        if coco_data is not None:
            data_adds.append(coco_data)

    output: Dict[str, Any] = {
        k: data_adds[0][k] for k in data_adds[0] if k not in ("images", "annotations")
    }

    output["images"], output["annotations"] = [], []

    merged_contributors = set()
    merged_source_logs = []

    for i, data in enumerate(data_adds):

        # logger.info(
        #     "Input {}: {} images, {} annotations".format(
        #         i + 1, len(data["images"]), len(data["annotations"])
        #     )
        # )

        cat_id_map = {}
        for new_cat in data["categories"]:
            new_id = None
            for output_cat in output["categories"]:
                if new_cat["name"] == output_cat["name"]:
                    new_id = output_cat["id"]
                    break

            if new_id is not None:
                cat_id_map[new_cat["id"]] = new_id
            else:
                new_cat_id = max(c["id"] for c in output["categories"]) + 1
                cat_id_map[new_cat["id"]] = new_cat_id
                new_cat["id"] = new_cat_id
                output["categories"].append(new_cat)

        img_id_map = {}
        for image in data["images"]:
            n_imgs = len(output["images"])
            img_id_map[image["id"]] = n_imgs
            image["id"] = n_imgs

            output["images"].append(image)

        for annotation in data["annotations"]:
            n_anns = len(output["annotations"])
            annotation["id"] = n_anns
            annotation["image_id"] = img_id_map[annotation["image_id"]]
            annotation["category_id"] = cat_id_map[annotation["category_id"]]

            output["annotations"].append(annotation)

        if "info" in data:
            if "contributors" in data["info"]:
                contributors = data["info"]["contributors"].split(",")
                contributors = [c.strip() for c in contributors]
                merged_contributors.update(contributors)

            if "source_log" in data["info"]:
                merged_source_logs.append(data["info"]["source_log"])
            if "source_logs" in data["info"]:
                print("source_logs", data["info"]["source_logs"])
                merged_source_logs = [*merged_source_logs, *data["info"]["source_logs"]]

    coco_info = {
        'year': datetime.now().year,
        'version': version,
        'description': 'A COCO JSON annotated file corresponding to multiple Ariadne-generated logs',
        'contributors': ", ".join(list(merged_contributors)),
        'source_logs': merged_source_logs,
        'date_updated': datetime.now().isoformat(),
    }

    output["info"] = coco_info

    # logger.info(
    #     "Result: {} images, {} annotations".format(
    #         len(output["images"]), len(output["annotations"])
    #     )
    # )

    return output

def merge_partials(to_coco_file, partial_dir):
    coco_files = [os.path.join(partial_dir, file) for file in os.listdir(partial_dir) if file.endswith('.json')]

    version = 1
    if os.path.exists(to_coco_file):
        existing_merged = read_coco(to_coco_file)
        if existing_merged is not None:
            if "info" in existing_merged:
                existing_info = existing_merged["info"]
                if "source_logs" in existing_info:
                    source_logs = set(existing_info["source_logs"])
                    coco_files = [file for file in coco_files if os.path.basename(file).rstrip(".json") not in source_logs]
                if "version" in existing_info:
                    version = existing_info["version"] + 1

                    # backup the old coco file
                    old_coco_dir = f'{os.path.dirname(to_coco_file)}/old'
                    os.makedirs(old_coco_dir, exist_ok=True)
                    backup_coco_path = f'{old_coco_dir}/labels_v{version-1}.json'
                    with open(backup_coco_path, "w") as f:
                        json.dump(existing_merged, f, indent=2)

            coco_files = [existing_merged, *coco_files]

    if len(coco_files) <= 1:
        print('There are no new partials')
        return
    
    merged_coco = coco_merge(coco_files, to_coco_file, version=version)

    with open(to_coco_file, "w") as f:
        json.dump(merged_coco, f, indent=2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-d', default='dataset', type=str, required=False, help="The output dataset directory")
    config = parser.parse_args()
    
    merge_partials(f'{config.dataset}/annotations/labels.json', f'{config.dataset}/annotations/partial')