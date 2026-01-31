import streamlit as st
from PIL import Image
import torch
import os
import argparse
import yaml
import utils as util
from train import inference 
import tempfile
from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(
    repo_id="ntmy777/dehazedet",
    filename="best.pt"
)


# Load arguments and parameters
def load_args_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model_path', default=MODEL_PATH, type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_test', action='store_true')
    parser.add_argument('--data', default='rtts', type=str)
    parser.add_argument('--detection_weight', default=0.1, type=int)
    parser.add_argument('--dehazing_weight', default=0.9, type=int)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args([])

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    with open(os.path.join('args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    return args, params

# Streamlit UI
st.set_page_config(page_title="DehazeDET Inference", layout="centered")
st.title("DehazeDet - Object Detection with Image Restoration")

st.write("Upload a hazy image and the model will return a dehazed image with object detection results.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    args, params = load_args_params()

    # Save the uploaded PIL image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        image.save(tmp_file.name)  # Save the image to disk
        temp_path = tmp_file.name  # Get the path

    # Run inference with the temp file path
    result_img = inference(args.model_path, temp_path, args, params, device='cpu')

    # Show the result
    st.image(result_img, caption="Inference Result", use_container_width=True)


    

