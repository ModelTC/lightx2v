import os
from pathlib import Path
from subprocess import Popen

import torch
import numpy as np
import tensorrt as trt
from loguru import logger

from lightx2v.common.backend_infer.trt import common
from lightx2v.common.backend_infer.trt.trt_infer_base import TrtModelInferBase

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class HyVaeTrtModelInfer(TrtModelInferBase):
    """
    Implements hunyuan vae inference for the TensorRT engine.
    """

    def __init__(self, engine_path):
        super().__init__(engine_path)

    def __call__(self, batch, *args, **kwargs):
        """
        Execute inference
        """
        # Prepare the output data
        device = batch.device
        dtype = batch.dtype
        batch = batch.cpu().numpy()

        def get_output_shape(shp):
            b, c, t, h, w = shp
            out = (b, 3, 4 * (t - 1) + 1, h * 8, w * 8)
            return out

        vae_out_shape = get_output_shape(batch.shape)
        shp_dict = {"inp": batch.shape, "out": vae_out_shape}
        self.alloc(shp_dict)
        output = np.zeros(vae_out_shape, self.out_list[0]["dtype"])

        # Process I/O and execute the network
        common.memcpy_host_to_device(self.inputs[0]["allocation"], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        common.memcpy_device_to_host(output, self.outputs[0]["allocation"])
        output = torch.from_numpy(output).to(device).type(dtype)
        return output

    @staticmethod
    def export_to_onnx(decoder: torch.nn.Module, model_dir):
        logger.info("Start to do VAE onnx exporting.")
        device = next(decoder.parameters())[0].device
        example_inp = torch.rand(1, 16, 17, 32, 32).to(device).type(next(decoder.parameters())[0].dtype)
        out_path = str(Path(str(model_dir)) / "vae_decoder.onnx")
        torch.onnx.export(
            decoder.eval().half(),
            example_inp.half(),
            out_path,
            input_names=["inp"],
            output_names=["out"],
            opset_version=14,
            dynamic_axes={"inp": {1: "c1", 2: "c2", 3: "c3", 4: "c4"}, "out": {1: "c1", 2: "c2", 3: "c3", 4: "c4"}},
        )
        os.system(f"onnxsim {out_path} {out_path}")
        logger.info("Finish VAE onnx exporting.")
        return out_path

    @staticmethod
    def convert_to_trt_engine(onnx_path, engine_path):
        logger.info("Start to convert VAE ONNX to tensorrt engine.")
        cmd = (
            "trtexec "
            f"--onnx={onnx_path} "
            f"--saveEngine={engine_path} "
            "--allowWeightStreaming "
            "--stronglyTyped "
            "--fp16 "
            "--weightStreamingBudget=100 "
            "--minShapes=inp:1x16x9x18x16 "
            "--optShapes=inp:1x16x17x32x16 "
            "--maxShapes=inp:1x16x17x32x32 "
        )
        p = Popen(cmd, shell=True)
        p.wait()
        if not Path(engine_path).exists():
            raise RuntimeError(f"Convert vae onnx({onnx_path}) to tensorrt engine failed.")
        logger.info("Finish VAE tensorrt converting.")
        return engine_path
