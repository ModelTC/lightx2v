from tqdm import tqdm
import argparse
import glob
import os
import requests
import time


def post_i2v(image_path, output_path):
    url = "http://localhost:8000"

    file_name = os.path.basename(image_path)
    prompt = os.path.splitext(file_name)[0]
    save_video_path = os.path.join(output_path, f"{prompt}.mp4")

    message = {
        "prompt": prompt,
        "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "image_path": image_path,
        "save_video_path": save_video_path,
    }

    while True:
        response = requests.get(f"{url}/v1/service/status").json()
        if response["service_status"] == "idle":
            response = requests.post(f"{url}/v1/tasks/", json=message)
            return
        time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to img files.")
    parser.add_argument("--output_path", type=str, default="./vbench_i2v", help="output video path.")
    args = parser.parse_args()

    if os.path.exists(args.data_path):
        img_files = glob.glob(os.path.join(args.data_path, "*.jpg"))
        print(f"Found {len(img_files)} image files.")

        with tqdm(total=len(img_files)) as progress_bar:
            for idx, img_path in enumerate(img_files):
                post_i2v(img_path, args.output_path)
                progress_bar.update()
