# Standard Library
import copy
import os
import math

# Import from third library
import json

# Import from alt
from alt import dump, load
from alt.utils.file_helper import dump_json_lines


def merge_pre_sensebee(case_name, src_path, dst_path):
    for item in os.listdir(src_path):
        if not item.endswith(".json"):
            continue

        pre_path = os.path.join(src_path, item)
        cur_meta = load(pre_path)

        new_cur_meta = copy.deepcopy(cur_meta)

        for fisrt_idx, xx in enumerate(cur_meta["step_1"]["result"]):
            for second_idx, yy in enumerate(xx["rects"]):
                assert yy["imageName"] == new_cur_meta["step_1"]["result"][fisrt_idx]["rects"][second_idx]["imageName"]
                new_cur_meta["step_1"]["result"][fisrt_idx]["rects"][second_idx]["imageName"] = os.path.join(
                    case_name, yy["imageName"]
                )

        for third_idx, zz in enumerate(cur_meta["step_1"]["resultRect"]):
            assert zz["imageName"] == cur_meta["step_1"]["resultRect"][third_idx]["imageName"]
            new_cur_meta["step_1"]["resultRect"][third_idx]["imageName"] = os.path.join(case_name, yy["imageName"])
        dump(os.path.join(dst_path, item), new_cur_meta)


def split_group_labels(label_path, split_num):
    def split_list(input_list, split_num):
        per_num = len(input_list) // split_num + 1
        output = []
        for i in range(split_num):
            output.append(input_list[i * per_num : (i + 1) * per_num])
        return output

    metas = {}
    case_names = set()
    for item in open(label_path).readlines():
        data = json.loads(item)
        cur_case = data["lidar"].split("/")[0]

        if cur_case not in metas:
            metas[cur_case] = []
        case_names.add(cur_case)
        metas[cur_case].append(data)

    split_case_names = split_list(list(case_names), split_num=split_num)
    split_metas = [[] for _ in split_case_names]

    for idx, names in enumerate(split_case_names):
        for name in names:
            split_metas[idx].extend(metas[name])

    for idx, single_metas in enumerate(split_metas):
        dump_path = label_path.replace(".jsonl", f"sub_{idx}.jsonl")
        dump_json_lines(dump_path, single_metas)

        f = open(dump_path + ".tmp.txt", "w")
        calib_paths = set()
        for idx, single_line in enumerate(single_metas):
            f.writelines(single_line["lidar"] + "\n")

            for cam_meta in single_line["cameras"]:
                f.writelines(cam_meta["image"] + "\n")
                calib_paths.add(cam_meta["calib"])

        for item in calib_paths:
            f.writelines(item + "\n")
        f.close()


def split_group_labels_clip_num(label_path, clip_num_per_task):
    def split_list(input_list, clip_num_per_task):
        split_num = math.ceil(len(input_list) / clip_num_per_task)
        output = []
        for i in range(split_num):
            output.append(input_list[i * clip_num_per_task : (i + 1) * clip_num_per_task])
        return output

    metas = {}
    case_names = set()
    for item in open(label_path).readlines():
        data = json.loads(item)
        cur_case = data["lidar"].split("/")[0]

        if cur_case not in metas:
            metas[cur_case] = []
        case_names.add(cur_case)
        metas[cur_case].append(data)

    split_case_names = split_list(list(case_names), clip_num_per_task=clip_num_per_task)
    split_metas = [[] for _ in split_case_names]

    for idx, names in enumerate(split_case_names):
        for name in names:
            split_metas[idx].extend(metas[name])

    labeling_metas = []
    for idx, single_metas in enumerate(split_metas):
        dump_path = label_path.replace(".jsonl", f"sub_{idx}.jsonl")
        dump_json_lines(dump_path, single_metas)

        f = open(dump_path + ".tmp.txt", "w")

        labeling_metas.append([dump_path, dump_path + ".tmp.txt"])
        calib_paths = set()
        for idx, single_line in enumerate(single_metas):
            f.writelines(single_line["lidar"] + "\n")

            for cam_meta in single_line["cameras"]:
                f.writelines(cam_meta["image"] + "\n")
                calib_paths.add(cam_meta["calib"])

        for item in calib_paths:
            f.writelines(item + "\n")
        f.close()
    return labeling_metas
