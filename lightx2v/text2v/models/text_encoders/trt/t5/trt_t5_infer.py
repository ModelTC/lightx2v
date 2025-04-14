import os
from pathlib import Path
from subprocess import Popen

import torch
import tensorrt as trt
from loguru import logger
import numpy as np
from torch.nn.modules import Module

from lightx2v.common.backend_infer.trt import common
from lightx2v.common.backend_infer.trt.trt_infer_base import TrtModelInferBase, np_torch_dtype_map

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class T5TrtModelInfer(TrtModelInferBase):
    def __init__(self, engine_path, **kwargs):
        super().__init__(engine_path, **kwargs)
        import onnxruntime as ort

    def __call__(self, ids, mask, *args, **kwargs):
        device = ids.device
        ids = ids.cpu().numpy()
        mask = mask.cpu().numpy()
        shp_dict = {i["name"]: i["shape"] for i in self.inp_list}
        shp_dict.update({i["name"]: i["shape"] for i in self.out_list})
        self.alloc(shp_dict)

        out_list = []
        for o in self.outputs:
            out_list.append(np.zeros(o["shape"], o["dtype"]))
        for inp, data in zip(self.inputs, [ids, mask]):
            common.memcpy_host_to_device(inp["allocation"], np.ascontiguousarray(data))
        self.context.execute_v2(self.allocations)
        outs = []
        for i, out in enumerate(out_list):
            common.memcpy_device_to_host(out, self.outputs[i]["allocation"])
            out = torch.from_numpy(out).to(device)
            out = out.type(torch.bfloat16)
            outs.append(out)
        return outs[0]

    @staticmethod
    def export_to_onnx(model: Module, model_dir, *args, **kwargs):
        ids = kwargs.get("ids")
        mask = kwargs.get("mask")
        onnx_dir = Path(model_dir) / "onnx/t5"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = str(onnx_dir / "t5.onnx")
        torch.onnx.export(model, (ids, mask), onnx_path, opset_version=14)
        return onnx_path

    @staticmethod
    def convert_to_trt_engine(onnx_path, engine_path, *args, **kwargs):
        logger.info("Start to convert ONNX to tensorrt engine.")
        cmd = f"trtexec --onnx={onnx_path} --saveEngine={engine_path} --bf16 "
        p = Popen(cmd, shell=True)
        p.wait()
        if not Path(engine_path).exists():
            raise RuntimeError(f"Convert onnx({onnx_path}) to tensorrt engine failed.")
        logger.info("Finish tensorrt converting.")
        return engine_path
