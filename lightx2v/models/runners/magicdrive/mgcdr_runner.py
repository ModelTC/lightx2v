import os
import gc
import einops
import json
import copy
from copy import deepcopy
import torch
import torch.nn.functional as F
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.utils import save_image
from einops import repeat, rearrange
from typing import Optional, Tuple, List, Dict, Any
from lightx2v.models.input_encoders.hf.t5.model import T5EncoderModel
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.mgcdr.scheduler import MagicDriverScheduler
from lightx2v.models.schedulers.mgcdr.datasets.carla import CARLAVariableDataset
from lightx2v.models.networks.mgcdr.model import MagicDriveModel
from lightx2v.models.video_encoders.hf.mgcdr.vae import VideoAutoencoderKLCogVideoX
from lightx2v.models.input_encoders.hf.t5_v1_1_xxl.model import T5EncoderModel_v1_1_xxl
from lightx2v.utils.mgcdr.utils import apply_mask_strategy
from lightx2v.utils.profiler import ProfilingContext4Debug
from loguru import logger


VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def is_img(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in IMG_EXTENSIONS


def is_vid(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in VID_EXTENSIONS


def save_sample(x, save_path=None, fps=8, normalize=True, value_range=(-1, 1), force_video=False, 
                high_quality=False, verbose=True, with_postfix=True, force_image=False, save_per_n_frame=-1, tag=None):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    try:
        assert x.ndim == 4, f"Input dim is {x.ndim}/{x.shape}"
        x = x.to("cpu")
        if with_postfix:
            save_path += f"_{x.shape[-2]}x{x.shape[-1]}"
        if tag is not None:
            save_path += f"_{tag}"

        if not force_video and x.shape[1] == 1:  # T = 1: save as image
            if not is_img(save_path):
                save_path += ".png"
            x = x.squeeze(1)
            save_image([x], save_path, normalize=normalize, value_range=value_range)
        else:
            if with_postfix:
                save_path += f"_f{x.shape[1]}_fps{fps}.mp4"
            elif not is_vid(save_path):
                save_path += ".mp4"
            if normalize:
                low, high = value_range
                x.clamp_(min=low, max=high)
                x.sub_(low).div_(max(high - low, 1e-5))

            x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to(torch.uint8)
            imgList = [xi for xi in x.numpy()]
            if force_image:
                os.makedirs(save_path)
                for xi, _x in enumerate(imgList):
                    _save_path = os.path.join(save_path, f"f{xi:05d}.png")
                    Image.fromarray(_x).save(_save_path)
            elif high_quality:
                if save_per_n_frame > 0 and len(imgList) > save_per_n_frame:
                    single_value = len(imgList) % 2
                    for i in range(0, len(imgList) - single_value, save_per_n_frame):
                        if i == 0:
                            _save_path = f"_f0-{save_per_n_frame + 1}".join(os.path.splitext(save_path))
                            vid_len = save_per_n_frame + single_value
                        else:
                            vid_len = save_per_n_frame
                            _save_path = f"_f{i + 1}-{i + 1 + vid_len}".join(os.path.splitext(save_path))
                            i += single_value
                        if len(imgList[i:i+vid_len]) < vid_len:
                            logging.warning(f"{len(imgList)} will stop at frame {i}.")
                            break
                        clip = ImageSequenceClip(imgList[i:i+vid_len], fps=fps)
                        clip.write_videofile(
                            _save_path, verbose=verbose, bitrate="4M",
                            logger='bar' if verbose else None)
                        clip.close()
                else:
                    clip = ImageSequenceClip(imgList, fps=fps)
                    clip.write_videofile(
                        save_path, verbose=verbose, bitrate="4M",
                        logger='bar' if verbose else None)
                    clip.close()
            else:
                write_video(save_path, x, fps=fps, video_codec="h264")
        # if verbose:
        if True:
            print(f"Saved to {save_path}")
    except Exception as e:
        traceback.print_exc()
    return save_path


def move_to(obj, device, dtype=None, filter=lambda x: True):
    if torch.is_tensor(obj):
        if filter(obj):
            if dtype is None:
                dtype = obj.dtype
            return obj.to(device, dtype)
        else:
            return obj
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device, dtype, filter)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device, dtype, filter))
        return res
    elif obj is None:
        return obj
    else:
        raise TypeError(f"Invalid type {obj.__class__} for move_to.")


def unsqueeze_tensors_in_dict(in_dict: Dict[str, Any], dim) -> Dict[str, Any]:
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, torch.Tensor):
            out_dict[k] = v.unsqueeze(dim)
        elif isinstance(v, dict):
            out_dict[k] = unsqueeze_tensors_in_dict(v, dim)
        elif isinstance(v, list):
            if dim == 0:
                out_dict[k] = [v]
            elif dim == 1:
                out_dict[k] = [[vi] for vi in v]
            else:
                raise ValueError(
                    f"cannot handle {k}:{v} ({v.__class__}) with dim={dim}")
        elif v is None:
            out_dict[k] = None
        else:
            raise TypeError(f"Unknow dtype for {k}:{v} ({v.__class__})")
    return out_dict


def stack_tensors_in_dicts(
        dicts: List[Dict[str, Any]], dim, holder=None) -> Dict[str, Any]:
    """stack any Tensor in list of dicts. If holder is provided, dicts will be
    stacked ahead of holder tensor. Make sure no dict is changed in place.

    Args:
        dicts (List[Dict[str, Any]]): dicts to stack, without the desired dim.
        dim (int): dim to add for stack.
        holder (_type_, optional): dict to hold, with the desired dim. Defaults
        to None. 

    Raises:
        TypeError: if the datatype for values are not Tensor or dict.

    Returns:
        Dict[str, Any]: stacked dict.
    """
    if len(dicts) == 1:
        if holder is None:
            return unsqueeze_tensors_in_dict(dicts[0], dim)
        else:
            this_dict = dicts[0]
            final_dict = deepcopy(holder)
    else:
        this_dict = dicts[0]  # without dim
        final_dict = stack_tensors_in_dicts(dicts[1:], dim)  # with dim
    for k, v in final_dict.items():
        if isinstance(v, torch.Tensor):
            # for v in this_dict, we need to add dim before concat.
            if this_dict[k].shape != v.shape[1:]:
                print("Error")
            final_dict[k] = torch.cat([this_dict[k].unsqueeze(dim), v], dim=dim)
        elif isinstance(v, dict):
            final_dict[k] = stack_tensors_in_dicts(
                [this_dict[k]], dim, holder=v)
        elif isinstance(v, list):
            if dim == 0:
                final_dict[k] = [this_dict[k]] + v
            elif dim == 1:
                final_dict[k] = [
                    [this_vi] + vi for this_vi, vi in zip(this_dict[k], v)]
            else:
                raise ValueError(
                    f"cannot handle {k}:{v} ({v.__class__}) with dim={dim}")
        elif v is None:
            assert final_dict[k] is None
        else:
            raise TypeError(f"Unknow dtype for {k}:{v} ({v.__class__})")
    return final_dict


def pad_bboxes_to_maxlen(
        bbox_shape, max_len, bboxes=None, classes=None, masks=None, **kwargs):
    # NOTE: our latest mask has 0: none, 1: use, -1: drop
    B, N_out = bbox_shape[:2]  # only mask always has NC dim
    ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape[3:], dtype=torch.float32)
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.int32)
    ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.int32)
    if bboxes is not None:
        for _b in range(B):
            # box and classes
            _bboxes = bboxes[_b]
            _classes = classes[_b]
            if len(_bboxes) == N_out:
                for _n in range(N_out):
                    if _bboxes[_n] is None:  # never happen
                        continue  # empty for this view
                    this_box_num = len(_bboxes[_n])
                    ret_bboxes[_b, _n, :this_box_num] = _bboxes[_n]
                    ret_classes[_b, _n, :this_box_num] = _classes[_n]
                    if masks is not None:
                        ret_masks[_b, _n, :this_box_num] = masks[_b, _n]
                    else:
                        ret_masks[_b, _n, :this_box_num] = 1
            elif len(_bboxes) == 1:
                this_box_num = len(_bboxes[0])
                ret_bboxes[_b, :, :this_box_num] = _bboxes
                ret_classes[_b, :, :this_box_num] = _classes
                if masks is not None:
                    ret_masks[_b, :, :this_box_num] = masks[_b]
                else:
                    ret_masks[_b, :, :this_box_num] = 1
            else:
                raise RuntimeError(f"Wrong bboxes shape: {bboxes.shape}")

    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,
        "classes": ret_classes,
        "masks": ret_masks
    }
    return ret_dict


def collate_bboxes_to_maxlen(bbox, device, dtype, NC, T) -> None or dict:
    bbox_maxlen = 0
    bbox_shape = [T, NC, None, 8, 3]  # TODO: hard-coded bbox shape
    for bboxes_3d_data in bbox:  # loop over B
        if bboxes_3d_data is not None:
            mask_shape = bboxes_3d_data['masks'].shape
            bbox_maxlen = max(bbox_maxlen, mask_shape[2])  # T, NC, len, ...
    if bbox_maxlen == 0:
        # return None
        # HACK: training cannot take None bbox, we add one padding box
        bbox_maxlen = 1
    ret_dicts = []
    for bboxes_3d_data in bbox:
        bboxes_3d_data = {} if bboxes_3d_data is None else bboxes_3d_data
        new_data = pad_bboxes_to_maxlen(
            bbox_shape, bbox_maxlen, **bboxes_3d_data)  # treat T as B
        ret_dicts.append(new_data)
    ret = stack_tensors_in_dicts(ret_dicts, dim=0)  # add B dim
    ret = move_to(ret, device, dtype)
    return ret


@RUNNER_REGISTER("mgcdr")
class MagicDriverRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_class = CARLAVariableDataset
        self.dtype = torch.bfloat16
        self.save_fps = self.config.get("save_video_fps", 8)
        self.num_sampling_steps = self.config.get("num_sampling_steps", 1)
        self.num_timesteps = self.config.get("num_timesteps", 1000)
        
    def init_modules(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.vae = self.load_vae()
    
    def load_transformer(self):
        model = MagicDriveModel(config=self.config, device=self.init_device)
        return model
    
    def load_text_encoder(self):
        text_encoder = T5EncoderModel_v1_1_xxl(
            self.config
        )
        text_encoders = [text_encoder]
        return text_encoders
    
    def load_vae(self):
        vae_path = f'{self.config.get('vae_path')}'
        vae_micro_frame_size = self.config.get('micro_frame_size', 32)
        vae_micro_batch_size = self.config.get('micro_batch_size', 1)
        vae_config = {
            'from_pretrained': vae_path, 
            'subfolder': 'vae', 
            'micro_frame_size': vae_micro_frame_size, 
            'micro_batch_size': vae_micro_batch_size
        }
        vae = VideoAutoencoderKLCogVideoX(
            **vae_config
        )
        return vae
    
    def run_text_encoder(self, text, neg_text):
        n = len(text)
        text_encoder_output = {}
        y, mask = self.text_encoders[0].infer([text], self.config, return_attention_mask=True)
        text_encoder_output["y"] = y
        if neg_text is None:
            text_encoder_output["y_null"] = self.model.pre_weight_class.y_embedder_y_embedding.tensor[None].repeat(n, 1, 1)[:, None].to(self.init_device)
        else:
            text_encoder_output["y_null"] = self.text_encoders[0].infer([neg_text], self.config)
        text_encoder_output["mask"] = mask
        return text_encoder_output
    
    def init_scheduler(self):
        scheduler = MagicDriverScheduler(self.config)
        self.model.set_scheduler(scheduler)
    
    def run_vae_encoder(self, img):
        pass
    
    def set_target_shape(self):
        pass
    
    def save_video_func(self, video):
        save_sample(
            video,
            fps=self.save_fps, # 8
            save_path=self.save_path,
            high_quality=True,
            verbose=False,
            save_per_n_frame=self.config.get("save_per_n_frame", -1), # -1
            force_image=self.config.get("force_image", False), # False
        )
        
    def replace_with_null_condition(self, _model_args, uncond_cam, uncond_rel_pos,
                                uncond_y, keys, append=False):
        unchanged_keys = ["mv_order_map", "t_order_map", "height", "width", "num_frames", "fps"]
        handled_keys = []
        model_args = {}
        if "y" in keys and "y" in _model_args:
            handled_keys.append("y")
            if append:
                model_args["y"] = torch.cat([_model_args["y"], uncond_y], 0)
            else:
                model_args['y'] = uncond_y
            keys.remove("y")

        if "bbox" in keys and "bbox" in _model_args:
            handled_keys.append("bbox")
            _bbox = _model_args['bbox']
            bbox = {}
            for k in _bbox.keys():
                null_item = torch.zeros_like(_bbox[k])
                if append:
                    bbox[k] = torch.cat([_bbox[k], null_item], dim=0)
                else:
                    bbox[k] = null_item
            model_args['bbox'] = bbox
            keys.remove("bbox")

        if "cams" in keys and "cams" in _model_args:
            handled_keys.append("cams")
            cams = _model_args['cams']  # BxNC, T, 1, 3, 7
            null_cams = torch.zeros_like(cams)
            BNC, T, L = null_cams.shape[:3]
            null_cams = null_cams.reshape(-1, 3, 7)
            null_cams[:] = uncond_cam[None]
            null_cams = null_cams.reshape(BNC, T, L, 3, 7)
            if append:
                model_args['cams'] = torch.cat([cams, null_cams], dim=0)
            else:
                model_args['cams'] = null_cams
            keys.remove("cams")

        if "rel_pos" in keys and "rel_pos" in _model_args:
            handled_keys.append("rel_pos")
            rel_pos = _model_args['rel_pos'][..., :-1, :]  # BxNC, T, 1, 4, 4
            null_rel_pos = torch.zeros_like(rel_pos)
            BNC, T, L = null_rel_pos.shape[:3]
            null_rel_pos = null_rel_pos.reshape(-1, 3, 4)
            null_rel_pos[:] = uncond_rel_pos[None]
            null_rel_pos = null_rel_pos.reshape(BNC, T, L, 3, 4)
            if append:
                model_args['rel_pos'] = torch.cat([rel_pos, null_rel_pos], dim=0)
            else:
                model_args['rel_pos'] = null_rel_pos
            keys.remove("rel_pos")

        if "maps" in keys and "maps" in _model_args:
            handled_keys.append("maps")
            maps = _model_args["maps"]
            null_maps = torch.zeros_like(maps)
            if append:
                model_args['maps'] = torch.cat([maps, null_maps], dim=0)
            else:
                model_args['maps'] = null_maps
            keys.remove("maps")

        if len(keys) > 0:
            raise RuntimeError(f"{keys} left unhandled with {_model_args.keys()}")
        for k in _model_args.keys():
            if k in handled_keys:
                continue
            elif k in unchanged_keys:
                model_args[k] = _model_args[k]
            elif k == "bbox":
                _bbox = _model_args['bbox']
                bbox = {}
                for kb in _bbox.keys():
                    bbox[kb] = repeat(_bbox[kb], "b ... -> (2 b) ...")
                model_args['bbox'] = bbox
            else:
                if append:
                    model_args[k] = repeat(_model_args[k], "b ... -> (2 b) ...")
                else:
                    model_args[k] = deepcopy(_model_args[k])
        return model_args

    def get_additional_inputs(self):
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]      
        dataset_params_json = self.config.get("dataset_params_json")
        with open(dataset_params_json, 'r') as file:
            dataset_args_dict = json.load(file)
        cam_params_json = self.config.get("cam_params_json")
        raw_meta_files = [self.config.get("raw_meta_files")]
        scene_description_file = [self.config.get("scene_description_file")]
        dataset_args_dict.update(
            {
                "pap_cam_init_path": cam_params_json,
                "raw_meta_files": raw_meta_files,
                "scene_description_file": scene_description_file
            }
        )
        self.dataset = self.dataset_class(**dataset_args_dict)
        batch = self.dataset[0]
        
        B, T, NC = batch["pixel_values"].shape[:3]
        latent_size = self.vae.get_latent_size((T, *batch["pixel_values"].shape[-2:]))
        x = batch.pop("pixel_values").to(self.init_device, self.dtype)
        x = rearrange(x, "B T NC C ... -> (B NC) C T ...")
        y = batch.pop("captions")[0]
        maps = batch.pop("bev_map_with_aux").to(self.init_device, self.dtype)
        bbox = batch.pop("bboxes_3d_data")
        bbox = [bbox_i.data for bbox_i in bbox]
        bbox = collate_bboxes_to_maxlen(bbox, self.init_device, self.dtype, NC, T)
        cams = batch.pop("camera_param").to(self.init_device, self.dtype)
        cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")  # BxNC, T, 1, 3, 7
        rel_pos = batch.pop("frame_emb").to(self.init_device, self.dtype)
        trans_scale = self.config.get("trans_scale", 1)
        rel_pos, pose_vis = edit_pos(rel_pos, self.config.get("traj", None), trans_scale,
            edit_param1=self.config.get("traj_param1", None),
            edit_param2=self.config.get("traj_param2", None),
            edit_param3=self.config.get("traj_param3", None),
        ) # time-consuming pre op
        rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)
        condition_frame_length = self.config.get("condition_frame_length", 0)
        prompts = y
        neg_prompts = None
        refs = [""] * len(y)
        ms = [""] * len(y)
        
        model_args = {}
        model_args["maps"] = maps
        model_args["bbox"] = bbox
        model_args["cams"] = cams
        model_args["rel_pos"] = rel_pos
        model_args["fps"] = batch.pop('fps')
        model_args['drop_cond_mask'] = torch.ones((B))  # camera
        model_args['drop_frame_mask'] = torch.ones((B, T))  # box & rel_pos
        model_args["height"] = batch.pop("height")
        model_args["width"] = batch.pop("width")
        model_args["num_frames"] = batch.pop("num_frames")
        model_args = move_to(model_args, device=self.init_device, dtype=self.dtype)
        # no need to move these
        model_args["mv_order_map"] = self.config.get(
            "mv_order_map", 
            {
                0: [5, 1, 6],
                1: [0, 2, 6],
                2: [1, 3],
                3: [2, 4],
                4: [3, 5],
                5: [4, 0, 6],
                6: [5, 1, 0]
            }
        )
        model_args["t_order_map"] = self.config.get("t_order_map", None)
        
        bbox = self.add_box_latent(bbox, B, NC, T)
        new_bbox = {}
        for k, v in bbox.items():
            new_bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")  # BxNC, T, len, 3, 7
        model_args["bbox"] = move_to(new_bbox, device=self.init_device, dtype=self.dtype)
        z = torch.randn(len(prompts), self.config.get("in_channels", 16) * NC, *latent_size, device=self.init_device, dtype=self.dtype)
        mask = apply_mask_strategy(z, refs, ms, 0, align=None)
        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)
        mask_t = mask * self.num_timesteps
        x0 = z.clone()
        x_noise = self.add_noise(x0, torch.randn_like(x0), timesteps)
        mask_t_upper = mask_t >= timesteps.unsqueeze(1)
        mask_add_noise = mask_t_upper & ~noise_added
        z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
        noise_added = mask_t_upper
        model_args["x_mask"] = mask_t_upper
        model_args["x"] = z
        model_args["timestep"] = timesteps
        
        return prompts, neg_prompts, model_args
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise
    
    def add_box_latent(self, bbox, B, NC, T, len_dim=3):
        # == add latent, NC and T share the same set of latents ==
        max_len = bbox['bboxes'].shape[len_dim]
        
        # _bbox_latent = sample_func(B * max_len)
        _bbox_latent = torch.randn(
            (B*max_len, self.config.get["hidden_size", 1152])
        ) 
        
        if _bbox_latent is not None:
            _bbox_latent = _bbox_latent.view(B, max_len, -1)
            # finally, add to bbox
            bbox['box_latent'] = einops.repeat(
                _bbox_latent, "B ... -> B T NC ...", NC=NC, T=T)
        return bbox
    
    def run_input_encoder(self):
        self.cond_inputs = {}
        prompts, neg_prompts, model_args = self.get_additional_inputs()
        self.cond_inputs.update(model_args)
        text_encoder_outputs = self.run_text_encoder(prompts, neg_prompts)
        uncond_y = text_encoder_outputs.pop('y_null')
        model_args.update(text_encoder_outputs)
        self.uncond_inputs = copy.deepcopy(self.cond_inputs)
        uncond_cam = self.model.pre_weight.camera_embedder_uncond_cam
        uncond_rel_pos = self.model.pre_weight.frame_embedder_uncond_cam
        self.uncond_inputs = self.replace_with_null_condition(self.uncond_inputs, uncond_cam, uncond_rel_pos, uncond_y, keys=["y", "bbox", "cams", "rel_pos", "maps"], append=False)
        
    def run_vae_decoer(self, samples):
        pass
    
    def set_model_args(self):
        pass
    
    def run(self):
        for step_index in range(self.model.scheduler.infer_steps):
            logger.info(f"==> step_index: {step_index + 1} / {self.model.scheduler.infer_steps}")

            with ProfilingContext4Debug("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)

            with ProfilingContext4Debug("cond infer"):
                self.model.infer(self.cond_inputs)
                
            with ProfilingContext4Debug("uncond infer"):
                self.model.infer(self.uncond_inputs)

            with ProfilingContext4Debug("step_post"):
                self.model.scheduler.step_post()
    
    def run_dit(self):
        self.init_scheduler()
        self.model.scheduler.prepare()
        samples, generator = self.run()
        return samples, generator
    
    async def run_pipeline(self):
        await self.run_input_encoder()
        samples, generator = await self.run_dit()
        samples = await self.run_vae_decoer(samples)
        self.save_video(samples)
        del samples, generator
        torch.cuda.empty_cache()
        gc.collect()
