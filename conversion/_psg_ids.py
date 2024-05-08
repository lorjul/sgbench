import json


def get_psg_ids(psg_path):
    with open(psg_path) as f:
        psg = json.load(f)

    img_ids = []
    for x in psg["data"]:
        if x["image_id"] in psg["test_image_ids"] and len(x["relations"]) > 0:
            img_ids.append(x["image_id"])

    return img_ids
