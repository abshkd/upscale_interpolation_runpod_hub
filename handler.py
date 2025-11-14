import runpod
from runpod.serverless.utils import rp_upload
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii # Import for Base64 error handling
import time

# Logging setup
logging.basicConfig(level=logging.WARN, force=True)
logger = logging.getLogger(__name__)

# CUDA check and setup
def check_cuda_availability():
    """Check CUDA availability and set environment variables."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("✅ CUDA is available and working")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            return True
        else:
            logger.error("❌ CUDA is not available")
            raise RuntimeError("CUDA is required but not available")
    except Exception as e:
        logger.error(f"❌ CUDA check failed: {e}")
        raise RuntimeError(f"CUDA initialization failed: {e}")

# Execute CUDA check
try:
    cuda_available = check_cuda_availability()
    if not cuda_available:
        raise RuntimeError("CUDA is not available")
except Exception as e:
    logger.error(f"Fatal error: {e}")
    logger.error("Exiting due to CUDA requirements not met")
    exit(1)


server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())
    
def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                # Encode bytes object to base64 for JSON serialization
                if isinstance(image_data, bytes):
                    import base64
                    image_data = base64.b64encode(image_data).decode('utf-8')
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

    
def get_video_path(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'gifs' in node_output:
            for video in node_output['gifs']:
                # Return first video file path
                return video['fullpath']
    
    return None

def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)


def handler(job):
    job_input = job.get("input", {})
    logger.info(f"Received job input: {job_input}")
    task_id = f"task_{uuid.uuid4()}"

    # Check task type
    task_type = job_input.get("task_type", "upscale")
    
    # Process video input (one of: video_path, video_url, video_base64)
    video_path_input = job_input.get("video_path")
    video_url_input = job_input.get("video_url")
    video_base64_input = job_input.get("video_base64")
    
    if not (video_path_input or video_url_input or video_base64_input):
        return {"error": "Video input required (one of: video_path, video_url, video_base64)"}
    
    # Obtain video file path
    if video_path_input:
        # If file path
        if video_path_input == "/example_video.mp4":
            video_path = "/example_video.mp4"
            return {"video": "test"}
        else:
            video_path = video_path_input
    elif video_url_input:
        # If URL, download it
        try:
            import urllib.request
            # Save to ComfyUI input directory with unique filename
            video_filename = f"{task_id}_input.mp4"
            video_path = os.path.join("/ComfyUI/input", video_filename)
            os.makedirs("/ComfyUI/input", exist_ok=True)
            urllib.request.urlretrieve(video_url_input, video_path)
            logger.info(f"Video downloaded from URL: {video_url_input}")
        except Exception as e:
            return {"error": f"Video URL download failed: {e}"}
    elif video_base64_input:
        # If Base64, decode and save
        try:
            # Save to ComfyUI input directory with unique filename
            video_filename = f"{task_id}_input.mp4"
            video_path = os.path.join("/ComfyUI/input", video_filename)
            os.makedirs("/ComfyUI/input", exist_ok=True)
            
            # Strip data URI prefix if present
            if video_base64_input.startswith('data:'):
                video_base64_input = video_base64_input.split(',', 1)[1]
            
            decoded_data = base64.b64decode(video_base64_input)
            with open(video_path, 'wb') as f:
                f.write(decoded_data)
            logger.info(f"Base64 video saved to file '{video_path}'.")
        except Exception as e:
            return {"error": f"Base64 video decoding failed: {e}"}
    
    # Load and configure workflow
    if task_type == "upscale":
        # Upscaling only
        prompt = load_workflow("/upscale.json")
        prompt["8"]["inputs"]["video"] = video_path
    elif task_type == "upscale_and_interpolation":
        # Upscaling + frame interpolation
        prompt = load_workflow("/upscale_and_interpolation.json")
        prompt["8"]["inputs"]["video"] = video_path
    else:
        return {"error": f"Unsupported task type: {task_type}"}

    # Connect to ComfyUI server and process
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")
    
    # First check if HTTP connection is available
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")
    
    # Check HTTP connection (up to 1 minute)
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            import urllib.request
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP connection successful (attempt {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(f"HTTP connection failed (attempt {http_attempt+1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("Cannot connect to ComfyUI server. Please check if the server is running.")
            time.sleep(1)
    
    ws = websocket.WebSocket()
    # Attempt WebSocket connection (up to 3 minutes)
    max_attempts = int(180/5)  # 3 minutes (attempt every second)
    for attempt in range(max_attempts):
        import time
        try:
            ws.connect(ws_url)
            logger.info(f"WebSocket connection successful (attempt {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"WebSocket connection failed (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("WebSocket connection timeout (3 minutes)")
            time.sleep(5)
    
    video_path = get_video_path(ws, prompt)
    ws.close()

    # Handle case when video is not available
    if not video_path:
        return {"error": "Unable to generate video."}
    
    # Check network_volume parameter
    use_network_volume = job_input.get("network_volume", False)
    
    if use_network_volume:
        # Using network volume: return file path
        try:
            # Create output video file path
            output_filename = f"{task_type}_{task_id}.mp4"
            output_path = f"/runpod-volume/{output_filename}"
            
            # Copy original file to output path
            import shutil
            shutil.copy2(video_path, output_path)
            
            logger.info(f"Result video saved to '{output_path}'.")
            return {"video_path": output_path}
            
        except Exception as e:
            logger.error(f"Video save failed: {e}")
            return {"error": f"Video save failed: {e}"}
    else:
        # Not using network volume: Base64 encode and return
        try:
            with open(video_path, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
            
            logger.info("Video encoded to Base64 and returned.")
            return {"video": video_data}
            
        except Exception as e:
            logger.error(f"Video Base64 encoding failed: {e}")
            return {"error": f"Video encoding failed: {e}"}

runpod.serverless.start({"handler": handler})