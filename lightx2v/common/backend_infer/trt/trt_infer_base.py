from pathlib import Path

import numpy as np
import torch
import tensorrt as trt
from cuda import cudart
import torch.nn as nn
from loguru import logger

from lightx2v.common.backend_infer.trt import common

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


np_torch_dtype_map = {"float16": torch.float16, "float32": torch.float32}


class TrtModelInferBase(nn.Module):
    """
    Implements inference for the TensorRT engine.
    """

    def __init__(self, engine_path, **kwargs):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"Tensorrt engine `{str(engine_path)}` not exists.")
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context
        logger.info(f"Loaded tensorrt engine from `{engine_path}`")
        self.inp_list = []
        self.out_list = []
        self.get_io_properties()

    def alloc(self, shape_dict):
        """
        Setup I/O bindings
        """
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = shape_dict[name]
            if is_input:
                self.context.set_input_shape(name, shape)
                self.batch_size = shape[0]
            if dtype == trt.DataType.BF16:
                dtype = trt.DataType.HALF
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def get_io_properties(self):
        for bind in self.engine:
            mode = self.engine.get_tensor_mode(bind)
            if mode.name == "INPUT":
                self.inp_list.append({"name": bind, "shape": self.engine.get_tensor_shape(bind), "dtype": self.engine.get_tensor_dtype(bind).name})
            else:
                self.out_list.append({"name": bind, "shape": self.engine.get_tensor_shape(bind), "dtype": self.engine.get_tensor_dtype(bind).name})
        return

    def __call__(self, batch, *args, **kwargs):
        pass

    @staticmethod
    def export_to_onnx(model: torch.nn.Module, model_dir):
        pass

    @staticmethod
    def convert_to_trt_engine(onnx_path, engine_path):
        pass
