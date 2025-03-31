import torch
from transformers import CLIPTextModel, AutoTokenizer


class TextEncoderHFClipModel:
    def __init__(self, model_path, device):
        self.device = device
        self.model_path = model_path
        self.init()
        self.load()

    def init(self):
        self.max_length = 77

    def load(self):
        self.model = (
            CLIPTextModel.from_pretrained(self.model_path).to(torch.float16).to(self.device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="right")

    def to_cpu(self):
        self.model = self.model.to("cpu")

    def to_cuda(self):
        self.model = self.model.to("cuda")

    @torch.no_grad()
    def infer(self, text, args):
        if args.cpu_offload:
            self.to_cuda()
        tokens = self.tokenizer(
            text,
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ).to("cuda")

        outputs = self.model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=False,
        )

        last_hidden_state = outputs["pooler_output"]
        if args.cpu_offload:
            self.to_cpu()
        return last_hidden_state, tokens["attention_mask"]


if __name__ == "__main__":
    model = TextEncoderHFClipModel(
        "/mnt/nvme0/yongyang/projects/hy/HunyuanVideo/ckpts/text_encoder_2", torch.device("cuda")
    )
    text = "A cat walks on the grass, realistic style."
    outputs = model.infer(text)
    print(outputs)
