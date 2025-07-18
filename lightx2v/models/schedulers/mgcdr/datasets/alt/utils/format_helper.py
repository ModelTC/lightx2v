class FormatUpdator:
    prefix = "visible_1.0"

    @classmethod
    def update_txt(cls, txt_path):
        assert "trainlist.tmp.txt" in txt_path

        dst_path = txt_path.replace("trainlist.tmp.txt", f"{cls.prefix}.trainlist.tmp.txt")

        return dst_path

    @classmethod
    def update_json(cls, json_path):
        assert json_path.endswith(".json")

        dst_path = json_path[:-4] + f"{cls.prefix}.json"
        return dst_path

    @classmethod
    def get_vis_path(cls, txt_path):
        assert "trainlist.tmp.txt" in txt_path
        vis_path = txt_path[: -len("trainlist.tmp.txt") - 1].replace("trainlist", "visualization")
        return vis_path


if __name__ == "__main__":
    txt_path = "s3://sdc3-gt-qc-2/pap/macj88_0726rebrush/trainlist/onlyStopLine/2024_06_14_10_58_01_AutoCollect.trainlist.tmp.txt"
    new_txt_path = FormatUpdator.update_txt(txt_path)
    print(txt_path)
    print(new_txt_path)

    json_path = "ad_system_common_auto:s3://sdc3-gt-qc-2/pap/macj88_0723j6e/refresh_000_20240710_label/parse/2024_07_09_11_22_36_AutoCollect/2024_07_09_10_35_09_j6eGtParser/1720492525096018944/stop_line.json"
    new_json_path = FormatUpdator.update_json(json_path)
    print(json_path)
    print(new_json_path)
