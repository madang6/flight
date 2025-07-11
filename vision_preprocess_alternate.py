import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage.segmentation import slic
from skimage.metrics import structural_similarity as ssim
from scipy.stats import mode

import torch
import torch.nn.functional as F
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cpu")
import onnxruntime as ort

import cv2
import imageio
from PIL import Image
import albumentations as A

from typing import Any, List, Optional, Tuple, Type, Union

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class CLIPSegHFModel:
    def __init__(
        self,
        hf_model: str = "CIDAS/clipseg-rd64-refined",
        device: Optional[str] = None,
        cmap: str = "turbo",
        onnx_model_path: Optional[str] = None,
        onnx_model_fp16_path: Optional[str] = None
    ):
        """
        HuggingFace CLIPSeg wrapper for torch inference, with optional ONNX fallback.
        If you pass onnx_model_path and it doesn’t exist, we’ll export it automatically.
        """
        # select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # load HF processor & model
        self.processor = CLIPSegProcessor.from_pretrained(hf_model, use_fast=True)
        self.model = CLIPSegForImageSegmentation.from_pretrained(hf_model)
        self.model.to(self.device).eval()

        # color‐lut, running bounds, caches, etc.
        self.lut = get_colormap_lut(cmap_name=cmap)
        self.running_min = float('inf')
        self.running_max = float('-inf')
        self.prev_image = None
        self.prev_output = None
        self.segmentation_ema = None
        self.ema_alpha = 0.7
        self.last_superpixel_mask = None
        self.superpixel_every = 1
        self.frame_counter = 0

        # ONNX support
        self.use_onnx = False
        self.ort_session = None
        self.using_fp16 = False
        if onnx_model_path is not None:
            if ort is None:
                raise ImportError(
                    "onnxruntime is not installed; pip install onnxruntime to use ONNX inference."
                )
            # if the .onnx file is missing, export it:
            if not os.path.isfile(onnx_model_path):
                self._export_onnx(onnx_model_path)
            if onnx_model_fp16_path:
                if (not os.path.isfile(onnx_model_fp16_path)):
                    self._convert_to_fp16(onnx_model_path, onnx_model_fp16_path)
                onnx_model_path = onnx_model_fp16_path
                self.using_fp16 = True

            # load the session
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            
            so.log_severity_level = 1
            self.ort_session = ort.InferenceSession(
                onnx_model_path,
                sess_options=so,
                providers=["TensorrtExecutionProvider"]#, "CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.io_binding = self.ort_session.io_binding()
            self.use_onnx = True

            self._io_binding = None
            self._input_gpu  = None
            self._output_gpu = None



    def _export_onnx(self, onnx_path: str):
        """
        Exports the HF CLIPSeg model to ONNX at onnx_path.
        Uses a dummy text+image input from the processor.
        """
        # pick a dummy prompt and image
        dummy_prompt = "a photo of a cat"
        dummy_img = Image.new("RGB", (224, 224), color="white")

        # prepare torch inputs
        torch_inputs = self.processor(
            images=dummy_img,
            text=dummy_prompt,
            return_tensors="pt"
        )
        torch_inputs = {k: v.to(self.device) for k, v in torch_inputs.items()}

        # names must match the processor/model
        # input_names = ["pixel_values", "input_ids", "attention_mask"]
        # output_names = ["logits"]
        # dynamic_axes = {
        #     "pixel_values":   {0: "batch", 2: "height", 3: "width"},
        #     "input_ids":      {0: "batch", 1: "seq_len"},
        #     "attention_mask": {0: "batch", 1: "seq_len"},
        #     "logits":         {0: "batch", 2: "height", 3: "width"},
        # }

        # export
        torch.onnx.export(
            self.model,
            (
                torch_inputs["input_ids"],
                torch_inputs["pixel_values"],
                torch_inputs["attention_mask"],
            ),
            onnx_path,
            input_names=["input_ids", "pixel_values", "attention_mask"],
            # input_names=["input_ids", "pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {1: "seq_len"},
                # "pixel_values":   {2: "height", 3: "width"},
                "attention_mask": {1: "seq_len"},
                # "logits":         {2: "height", 3: "width"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    
    # def _convert_to_fp16(self, onnx_path: str, fp16_path: str):
    #     import onnx
    #     from onnxconverter_common import float16

    #     model = onnx.load("clipseg_model.onnx")

    #     model_fp16 = float16.convert_float_to_float16(
    #         model,
    #         keep_io_types=False,       # keep inputs/outputs in float32
    #         disable_shape_infer=True,  # skip ONNX shape inference
    #         op_block_list=[],
    #         check_fp16_ready=False
    #     )

    #     onnx.save_model(model_fp16, "clipseg_model_fp16.onnx")

    def _convert_to_fp16(self, onnx_path: str, fp16_path: str):
        import onnx
        from onnxconverter_common import float16
        """Convert an ONNX model to FP16 using the standard converter."""

        model = onnx.load(onnx_path)

        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=False,
        )

        onnx.save(model_fp16, fp16_path)

    def _rescale_global(self, arr: np.ndarray) -> np.ndarray:
        """
        Rescale `arr` to [0,1] using the min/max seen so far (updated here).
        """
        cur_min = float(arr.min())
        cur_max = float(arr.max())
        self.running_min = min(self.running_min, cur_min)
        self.running_max = max(self.running_max, cur_max)
        # print(f"Running bounds: min={self.running_min}, max={self.running_max}")
        # print(f"Current bounds: min={cur_min}, max={cur_max}")
        span = self.running_max - self.running_min
        if span < 1e-9:
            scaled = np.zeros_like(arr, dtype=np.float32)
        else:
            scaled = (arr - self.running_min) / span
        return scaled
    
    # def _run_onnx_model(self, img: Image.Image, prompt: str) -> np.ndarray:
    #     """
    #     Runs forward pass via onnxruntime and returns the raw logits as a float32 numpy array.
    #     """
    #     # 1) Get PyTorch tensors from the HF processor...
    #     torch_inputs = self.processor(images=img, text=prompt, return_tensors="pt")
    #     # 2) Move them to CPU & convert to numpy for ONNX runtime
    #     ort_inputs = {}
    #     for inp in self.ort_session.get_inputs():
    #         name = inp.name
    #         tensor = torch_inputs.get(name)
    #         if tensor is None:
    #             continue
    #         # detach, move to CPU, numpy
    #         ort_inputs[name] = tensor.cpu().numpy()
    #     # 3) Run the ONNX session
    #     ort_outs = self.ort_session.run(None, ort_inputs)
    #     # assume first output is logits [1,1,H,W]
    #     logits = ort_outs[0]
    #     return logits.squeeze().astype(np.float32)


    def _run_onnx_model(self, img: Image.Image, prompt: str) -> np.ndarray:
        import numpy as np
        import torch

        # 1) Preprocess on GPU
        torch_inputs = self.processor(images=img, text=prompt, return_tensors="pt")
        if self.using_fp16:
            # convert to FP16 if needed and move to device
            torch_inputs = {
                k: (v.half().to(self.device) if k == "pixel_values" else v.to(self.device))
                for k, v in torch_inputs.items()
            }
        else:
            torch_inputs = {k: v.to(self.device) for k, v in torch_inputs.items()}

        # 2) Fresh IOBinding
        io_binding = self.ort_session.io_binding()

        # 3) Bind inputs zero-copy
        sess_input_names = {inp.name for inp in self.ort_session.get_inputs()}
        for name, tensor in torch_inputs.items():
            if name not in sess_input_names:
                continue
            elem_type = np.float32 if tensor.dtype == torch.float32 else np.int64
            io_binding.bind_input(
                name=name,
                device_type=self.device,   # e.g. "cuda"
                device_id=0,
                element_type=elem_type,
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )

        # 4) Figure out the ONNX output shape
        out_meta = self.ort_session.get_outputs()[0]
        B, _, H, W = torch_inputs["pixel_values"].shape
        if len(out_meta.shape) == 3:
            # [batch, height, width]
            out_shape = (B, H, W)
        elif len(out_meta.shape) == 4:
            # [batch, channels, height, width]
            C = out_meta.shape[1] if isinstance(out_meta.shape[1], int) else 1
            out_shape = (B, C, H, W)
        else:
            raise RuntimeError(f"Unsupported logits rank: {len(out_meta.shape)}")

        # 5) Allocate & bind output buffer on GPU
        out_dtype = torch.float16 if self.using_fp16 else torch.float32
        output_gpu = torch.empty(out_shape, dtype=out_dtype, device=self.device)
        io_binding.bind_output(
            name=out_meta.name,
            device_type=self.device,
            device_id=0,
            element_type=(np.float16 if self.using_fp16 else np.float32),
            shape=out_shape,
            buffer_ptr=output_gpu.data_ptr(),
        )

        # 6) Run
        self.ort_session.run_with_iobinding(io_binding)

        # 7) Fetch & squeeze
        result = output_gpu.cpu().numpy()
        # if it’s [B, 1, H, W], drop the channel axis
        if result.ndim == 4 and result.shape[1] == 1:
            result = result[:, 0]
        # if batch‐size is 1, you can also drop that:
        return result.squeeze()

    def clipseg_hf_inference(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
        resize_output_to_input: bool = True,
        use_refinement: bool = False,
        use_smoothing: bool = False,
        scene_change_threshold: float = 1.00,
        verbose=False
    ) -> np.ndarray:
        """
        Run CLIPSeg on a PIL image or numpy array, return colorized mask as (H,W,3) uint8.
        """
        def log(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        # --- Step 1: Normalize input to PIL + NumPy ---
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
            image_np = image
        elif isinstance(image, Image.Image):
            img = image
            image_np = np.array(image)
        else:
            raise TypeError(f"Unsupported image type {type(image)}")

        # --- Step 2: Determine whether to reuse ---
        should_reuse = False
        ssim_score = None
        if scene_change_threshold < 1.0 and self.prev_image is not None and self.prev_output is not None:
            # Compare resized grayscale SSIM
            prev_small = cv2.resize(self.prev_image, (64, 64))
            curr_small = cv2.resize(image_np, (64, 64))
            prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            ssim_score = ssim(prev_gray, curr_gray, data_range=1.0)
            should_reuse = ssim_score >= scene_change_threshold
            log(f"[DEBUG] SSIM = {ssim_score:.4f}, Threshold = {scene_change_threshold}, Reuse = {should_reuse}")

        # --- Step 3: Reuse path (warp previous mask) ---
        if should_reuse:
            mask_u8 = warp_mask(self.prev_image, image_np, self.prev_output)
            # Optional: fast filtering
            mask_u8 = cv2.bilateralFilter(mask_u8, d=7, sigmaColor=75, sigmaSpace=75)
            colorized = colorize_mask_fast(mask_u8, self.lut)
            overlayed = blend_overlay_gpu(image_np, colorized)
            return overlayed

        # --- Step 4: Run inference (scene has changed) ---
        start = time.time()
        if self.use_onnx:
            arr = self._run_onnx_model(img, prompt)
        else:
            torch_inputs = self.processor(images=img, text=prompt, return_tensors="pt")
            torch_inputs = {k: v.to(self.device) for k, v in torch_inputs.items()}
            with torch.no_grad():
                logits = self.model(**torch_inputs).logits
            arr = logits.cpu().squeeze().numpy().astype(np.float32)

        skip_norm = prompt.strip().lower() == "null"
        if skip_norm:
            prob = 1.0 / (1.0 + np.exp(-arr))
            mask_u8 = (prob * 255).astype(np.uint8)
        else:
            scaled = self._rescale_global(arr)
            mask_u8 = (scaled * 255).astype(np.uint8)

        if resize_output_to_input:
            mask_u8 = np.array(Image.fromarray(mask_u8).resize(img.size, resample=Image.BILINEAR))

        # --- Step 5: Post-processing only on fresh inference ---
        if use_smoothing:
            # Temporal EMA
            mask_f = mask_u8.astype(np.float32)
            if self.segmentation_ema is None or self.segmentation_ema.shape != mask_f.shape:
                self.segmentation_ema = mask_f.copy()
            else:
                self.segmentation_ema = (
                    self.ema_alpha * mask_f + (1 - self.ema_alpha) * self.segmentation_ema
                )
            mask_u8 = np.clip(self.segmentation_ema, 0, 255).astype(np.uint8)

            # Optional filtering
            mask_u8 = cv2.bilateralFilter(mask_u8, d=7, sigmaColor=75, sigmaSpace=75)

        # Optional: superpixel refinement
        if use_refinement:
            self.frame_counter += 1
            if self.frame_counter % self.superpixel_every == 0:
                # self.last_superpixel_mask = superpixel_smoothing(image_np, mask_u8)
                self.last_superpixel_mask = fast_superpixel_seeds(image_np, mask_u8)
            if self.last_superpixel_mask is not None:
                mask_u8 = self.last_superpixel_mask

        # --- Step 6: Render and cache ---
        colorized = colorize_mask_fast(mask_u8, self.lut)
        overlayed = blend_overlay_gpu(image_np, colorized)

        # Store just raw mask + image for reuse
        self.prev_image = image_np.copy()
        self.prev_output = mask_u8.copy()

        end = time.time()
        log(f"CLIPSeg inference time: {end - start:.3f} seconds")
        return overlayed

################################################
# 2. Lookup Table for Semantic Probability Map #
################################################

# Precompute the LUT once
def get_colormap_lut(cmap_name="turbo", lut_size=256):
    cmap = plt.get_cmap(cmap_name, lut_size)
    lut = (cmap(np.linspace(0, 1, lut_size))[:, :3] * 255).astype(np.uint8)  # shape: (256, 3)
    return lut  # Each row is an RGB color

# Fast LUT-based colorizer
def colorize_mask_fast(mask_np, lut):
    """
    Convert a (H, W) uint8 mask into a (H, W, 3) RGB image using the provided LUT.
    """
    return lut[mask_np]  # Very fast NumPy indexing: shape (H, W, 3)

################################################
# 3. Utility Functions for Image Processing #
################################################
    

def blend_overlay_gpu(base: np.ndarray,
                      overlay: np.ndarray,
                      alpha: float = 1.00) -> np.ndarray:
    """
    Convert `base`→mono-gray on GPU (if needed), resize `overlay` on GPU,
    then blend:  result = α·overlay + (1−α)·gray_base.  Entirely on CUDA.

    Parameters:
        base (np.ndarray): 
            • If 3-channel: H×W×3 BGR/uint8. 
            • If single-channel: H×W/uint8. 
        overlay (np.ndarray): H'×W'×3 BGR/uint8 image to overlay.
        alpha (float): opacity of overlay in [0,1].

    Returns:
        np.ndarray: H×W×3 BGR/uint8 blended result (on CPU).
    """
    # 1. Upload base to GPU and convert to float
    if base.ndim == 2:
        # already gray
        base_gray = torch.from_numpy(base).float().cuda()             # shape [H, W]
    elif base.ndim == 3 and base.shape[2] == 3:
        B = torch.from_numpy(base[:, :, 0]).float().cuda()            # BGR channels
        G = torch.from_numpy(base[:, :, 1]).float().cuda()
        R = torch.from_numpy(base[:, :, 2]).float().cuda()
        # Standard Rec. 601 luma-weights for BGR→gray:
        base_gray = 0.114 * B + 0.587 * G + 0.299 * R               # shape [H, W]
    else:
        raise ValueError(
            "`base` must be H×W (gray) or H×W×3 (BGR).")

    H, W = base_gray.shape

    # 2. Upload overlay to GPU as a float tensor [3, H', W']
    if overlay.ndim != 3 or overlay.shape[2] != 3:
        raise ValueError("`overlay` must be H'×W'×3 (BGR/uint8).")
    ov = torch.from_numpy(overlay).float().permute(2, 0, 1).cuda()    # [3, H', W']
    ov = ov.unsqueeze(0)                                             # [1, 3, H', W']

    # 3. Resize overlay to match base’s H×W (bilinear on GPU)
    ov_resized = F.interpolate(
        ov, size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(0)                                                     # [3, H, W]

    # 4. Stack gray→3 channels: [3, H, W]
    gray3 = base_gray.unsqueeze(0).repeat(3, 1, 1)                    # [3, H, W]

    # 5. Blend on GPU: α·overlay + (1−α)·gray3
    blended = alpha * ov_resized + (1.0 - alpha) * gray3              # [3, H, W]

    # 6. Clamp to [0,255], cast → uint8, move to CPU, return as H×W×3
    blended = blended.clamp(0, 255).round().byte()                    # [3, H, W]
    return blended.permute(1, 2, 0).cpu().numpy()   

def guided_smoothing(rgb: np.ndarray, seg_mask: np.ndarray, radius=4, eps=1e-3) -> np.ndarray:
    rgb_float = rgb.astype(np.float32) / 255.0
    seg_float = seg_mask.astype(np.float32) / 255.0
    guided = cv2.ximgproc.guidedFilter(guide=rgb_float, src=seg_float, radius=radius, eps=eps)
    return (guided * 255).astype(np.uint8)

def superpixel_smoothing(image: np.ndarray, seg_mask: np.ndarray, n_segments=500) -> np.ndarray:
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=0)
    smoothed = np.zeros_like(seg_mask)
    for label in np.unique(segments):
        mask = segments == label
        values = seg_mask[mask]
        if values.size == 0:
            continue  # skip empty segment
        majority = mode(values, axis=None)[0]
        smoothed[mask] = majority.item()  # safely extract scalar
    return smoothed

def fast_superpixel_seeds(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    num_superpixels = 400  # Adjust as needed
    num_levels = 4
    prior = 2
    histogram_bins = 5
    double_step = False

    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        w, h, image.shape[2],
        num_superpixels, num_levels, prior,
        histogram_bins, double_step
    )

    seeds.iterate(image, num_iterations=2)  # Keep small
    labels = seeds.getLabels()
    out = np.zeros_like(mask)

    for label in np.unique(labels):
        region = mask[labels == label]
        if region.size > 0:
            out[labels == label] = np.bincount(region).argmax()
    return out


def scene_changed(prev: np.ndarray, curr: np.ndarray, threshold=0.02) -> bool:
    # resize if needed
    if prev.shape != curr.shape:
        curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]))

    # normalize
    prev = prev.astype(np.float32) / 255.0
    curr = curr.astype(np.float32) / 255.0

    diff = np.mean(np.abs(prev - curr))  # Mean absolute difference
    return diff > threshold

def scene_changed_ssim(prev: np.ndarray, curr: np.ndarray, threshold=0.95) -> bool:
    if prev.shape != curr.shape:
        curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]))

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    score = ssim(prev_gray, curr_gray, data_range=1.0)

    return score < threshold

def compute_ssim(im1: np.ndarray, im2: np.ndarray) -> float:
    gray1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def warp_mask(prev_rgb, curr_rgb, prev_mask):
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    h, w = prev_mask.shape
    flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
    remap = flow_map + flow
    warped = cv2.remap(prev_mask, remap[..., 0], remap[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped.astype(np.uint8)


def render_rescale(self, srgb_mono):
    '''This function takes a single channel semantic similarity and rescales it globally'''
    # Maintain running min/max
    if not hasattr(self, "running_min"):
        self.running_min = -1.0
    if not hasattr(self, "running_max"):
        self.running_max = 1.0

    current_min = srgb_mono.min().item()
    current_max = srgb_mono.max().item()
    self.running_min = min(self.running_min, current_min)
    self.running_max = max(self.running_max, current_max)

    similarity_clip = (srgb_mono - self.running_min) / (self.running_max - self.running_min + 1e-10)

    return similarity_clip
