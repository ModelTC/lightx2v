import os
import numpy as np
import pandas as pd
from alt import load

from .data_parser import data_parser_factory
from .evaluator import Evaluator

def get_default_parameters():
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    path = os.path.join(father_path, "../../configs/stable_eval.yaml")
    assert os.path.exists(path), path
    return load(path)['eval_parameters']

class StableEvaluator(object):
    def __init__(self, parameters=get_default_parameters(), ratio_only=False, **kwargs):
        self.ratio_only = ratio_only
        self.parameters = parameters

        if parameters["eval_global_params"].get("data_load"):
            data_load = parameters["eval_global_params"]["data_load"]
        else:
            raise ValueError("data_load must be assigned")

        self.cfg_eval_obj_types = parameters["eval_global_params"]["eval_object_types"]

        self.data_parser = data_parser_factory.creat_data_parser(data_load, None)
        self.evaluator = Evaluator(parameters)

    def filter_result(self, result):
        fileter_res = {}
        for cls in result:
            fileter_res[cls] = {}
            for metric_key in result[cls].keys():
                if self.ratio_only:
                    if "ratio" in metric_key:
                        fileter_res[cls][metric_key] = result[cls][metric_key]
                else:
                    fileter_res[cls][metric_key] = result[cls][metric_key]
        return fileter_res

    def evaluate(self, metas):
        # data load
        self.data_parser.data_parse(self.cfg_eval_obj_types, metas)
        eval_obj_types = self.data_parser.get_eval_types()

        all_types_eval_result = {}
        for obj_type in eval_obj_types:
            datas_org_by_frame = self.data_parser.get_data_org_by_frame(obj_type)
            datas_org_by_track = self.data_parser.get_data_org_by_track(obj_type)
            eval_result = self.evaluator.eval(obj_type, datas_org_by_frame, datas_org_by_track)
            all_types_eval_result[obj_type] = eval_result

        final_result = self.filter_result(all_types_eval_result)
        return final_result

    @classmethod
    def save_metrics_as_csv(cls, eval_result, save_file):
        rows_name = list(eval_result.keys())
        if len(rows_name) > 0:
            cols_name = list(eval_result[rows_name[0]].keys())
        else:
            return

        datas = []
        for r_name in rows_name:
            line = []
            for c_name in cols_name:
                line.append(eval_result[r_name][c_name])
            datas.append(line)
        datas = np.array(datas)
        df = pd.DataFrame(datas, columns=cols_name)
        df_row = pd.DataFrame({"eval_type_name": rows_name})
        df = pd.concat([df_row, df], axis=1)
        df.to_csv(save_file, index=False, float_format="%.3f")

if __name__ == "__main__":
    # Standard Library
    import pickle as pkl

    # Import from pipeflow
    import yaml

    meta = "kestrel_structural_test_data/ut_meta/stable_eval.pkl"
    cfg_file = "config/eval/stable_eval.yaml"

    targets = pkl.load(open(meta, "rb"))

    eval_params = yaml.load(open(cfg_file, "r"), Loader=yaml.SafeLoader)
    metrics = StableEvaluator(eval_params["eval_parameters"]).evaluate(targets)
    print(metrics)
