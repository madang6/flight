"""
Benchmark ONNX vs PyTorch CLIPSeg inference on a video.

Configuration is loaded from a JSON file at CONFIG_PATH.
"""
import os
import time
import statistics
import json

import numpy as np
import imageio
import cv2

import sousvide.flight.vision_preprocess_alternate as vp
# import sousvide.flight.zed_command_helper as zed

# Path to the external JSON config file
CONFIG_PATH = (
    "/home/admin/StanfordMSL/SousVide-Semantic/"
    "configs/perception/onnx_benchmark_config.json"
)


def load_config(path):
    """
    Load benchmark configuration from JSON.
    Expected keys:
      - input_video_path: str
      - prompt: str
      - hf_model: str (HuggingFace model name)
      - onnx_model_path: str or null
    """
    with open(path, 'r') as f:
        return json.load(f)


def main():
    # Load user configuration
    cfg = load_config(CONFIG_PATH)
    prompt = cfg.get('prompt', '')
    hf_model = cfg.get('hf_model', 'CIDAS/clipseg-rd64-refined')
    onnx_model_path = cfg.get('onnx_model_path')
    camera_mode = cfg.get('camera_mode', False)

    # Initialize CLIPSeg model (may export & load ONNX)
    if onnx_model_path is None:
        print("Initializing CLIPSegHFModel...")
        model = vp.CLIPSegHFModel(hf_model=hf_model)
    else:
        print("Initializing ONNX CLIPSegHFModel (this may export ONNX)...")
        model = vp.CLIPSegHFModel(
            hf_model=hf_model,
            onnx_model_path=onnx_model_path,
            onnx_model_fp16_path=cfg.get('onnx_model_fp16_path', None)
        )

    times = []
    frames = []
    frame_count = 0

    if camera_mode:
        input_video_path = cfg['input_video_path']
        video_dir = os.path.dirname(input_video_path)
        base, ext = os.path.splitext(os.path.basename(input_video_path))
        output_path = os.path.join(video_dir, f"live_onnx_benchmark{ext}")

        # Live camera benchmarking
        fps_cam = cfg.get('camera_fps', 30)
        duration = cfg.get('camera_duration', 10.0)
        width = cfg.get('camera_width', 640)
        height = cfg.get('camera_height', 480)

        camera = zed.get_camera(height=height, width=width, fps=fps_cam)
        if camera is None:
            raise RuntimeError("Unable to initialize camera.")

        print(f"Capturing live for {duration:.1f}s at {fps_cam} FPS...")
        start_time = time.time()

        while (time.time() - start_time) < duration:
            frame, timestamp = zed.get_image(camera)
            if frame is None:
                continue

            # Convert ZED frame BGR→RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Inference
            t0 = time.time()
            overlay = model.clipseg_hf_inference(
                frame_rgb,
                prompt,
                resize_output_to_input=True,
                use_refinement=False,
                use_smoothing=False,
                scene_change_threshold=1.0,
                verbose=False,
            )
            t1 = time.time()
            times.append(t1 - t0)
            frames.append(overlay)
            frame_count += 1

        zed.close_camera(camera)
        print("Live camera benchmarking completed. Saving processed frames...")

    else:
        # File-based video benchmarking
        input_video_path = cfg['input_video_path']
        video_dir = os.path.dirname(input_video_path)
        base, ext = os.path.splitext(os.path.basename(input_video_path))
        if onnx_model_path is None:
            output_path = os.path.join(video_dir, f"{base}_default_benchmark{ext}")
        else:
            output_path = os.path.join(video_dir, f"{base}_onnx_benchmark{ext}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {total_frames} frames at {fps:.2f} FPS...")

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            overlay = model.clipseg_hf_inference(
                frame_rgb,
                prompt,
                resize_output_to_input=True,
                use_refinement=False,
                use_smoothing=False,
                scene_change_threshold=1.0,
                verbose=False,
            )
            t1 = time.time()
            times.append(t1 - t0)
            frame_count += 1

            if overlay.ndim == 3 and overlay.shape[2] == 3:
                out_frame = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            else:
                out_frame = overlay
            out.write(out_frame)

            if frame_count % 50 == 0:
                avg_ms = statistics.mean(times[-50:]) * 1000
                print(f"  Frame {frame_count}/{total_frames}  avg {avg_ms:.1f} ms/frame")

        cap.release()
        out.release()
        print(f"Output video: {output_path}")
        print("File-based video benchmark completed.")

    # Print timing stats
    if frame_count > 0:
        total_time = sum(times)
        avg_fps = frame_count / total_time
        print(f"Total frames processed: {frame_count}")
        print(f"Total inference time: {total_time:.2f} s")
        print(f"Average time/frame: {statistics.mean(times)*1000:.1f} ms")
        print(f"Median time/frame: {statistics.median(times)*1000:.1f} ms")
        print(f"Min time/frame: {min(times)*1000:.1f} ms")
        print(f"Max time/frame: {max(times)*1000:.1f} ms")

        print(f"Average FPS: {avg_fps:.2f}")
        imageio.mimsave(output_path, frames, fps=avg_fps)
    else:
        print("No frames were processed.")


if __name__ == "__main__":
    main()
