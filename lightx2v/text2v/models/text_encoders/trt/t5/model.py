import logging

import torch

from ...hf.t5.tokenizer import HuggingfaceTokenizer
from .trt_t5_infer import T5TrtModelInfer


class T5EncoderModel:
    def __init__(self, text_len, dtype=torch.bfloat16, device=torch.cuda.current_device(), engine_path=None, checkpoint_path=None, tokenizer_path=None, **kwargs):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        self.model = T5TrtModelInfer(engine_path=engine_path)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=text_len, clean="whitespace")

    def to_cpu(self):
        self.model = self.model.to("cpu")

    def to_cuda(self):
        self.model = self.model.to("cuda")

    def infer(self, texts, args):
        if args.cpu_offload:
            self.to_cuda()

        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.cuda()
        mask = mask.cuda()
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)

        if args.cpu_offload:
            self.to_cpu()

        return [u[:v] for u, v in zip(context, seq_lens)]
