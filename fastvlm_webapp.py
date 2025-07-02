#!/usr/bin/env python3
"""
FastVLM Web App with Webcam Integration - Optimized Performance
A FastAPI web application that processes webcam frames using FastVLM
"""

import os
import asyncio
import base64
import io
import json
import gc
import threading
import queue
from typing import Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Import FastVLM components
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class FastVLMPredictor:
    def __init__(self, model_path: str, model_base: str = None, conv_mode: str = "qwen_2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing FastVLM on device: {self.device}")
        
        # Load model
        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, device=self.device
        )
        
        self.conv_mode = conv_mode
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        # Performance optimizations
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        print("FastVLM model loaded successfully!")
    
    def predict(self, image: Image.Image, prompt: str, temperature: float = 0.2) -> str:
        try:
            # Clear GPU cache before prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Construct prompt
            qs = prompt
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            # Tokenize prompt
            torch_device = torch.device(self.device)
            input_ids = tokenizer_image_token(
                prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(torch_device)
            
            # Process image
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            
            # Run inference with optimizations
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half(),
                    image_sizes=[image.size],
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=128,  # Reduced for faster inference
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                
                # Clean up tensors
                del input_ids, image_tensor, output_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return outputs
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return f"Error: {str(e)}"


class OptimizedWebcamManager:
    def __init__(self):
        self.cap = None
        self.frame_count = 0
        self.process_every_n_frames = 100
        self.last_processed_frame = 0
        self.frame_queue = queue.Queue(maxsize=2)  # Small buffer to prevent memory buildup
        self.processing_lock = threading.Lock()
    
    def start_camera(self, camera_index: int = 0):
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {camera_index}")
            
            # Optimize camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
            
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def get_frame(self):
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.frame_count += 1
        
        # Clear old frames from queue to prevent buildup
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Add current frame
        try:
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass
        
        return frame
    
    def should_process_frame(self):
        with self.processing_lock:
            should_process = (self.frame_count - self.last_processed_frame) >= self.process_every_n_frames
            if should_process:
                self.last_processed_frame = self.frame_count
            return should_process
    
    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break


# Global instances
predictor = None
webcam_manager = OptimizedWebcamManager()
executor = ThreadPoolExecutor(max_workers=1)  # Single worker for AI processing

app = FastAPI(title="FastVLM Webcam App", description="Real-time vision-language processing with webcam")

class PredictionRequest(BaseModel):
    system_prompt: str = "Describe what you see in this image."
    temperature: float = 0.2

class InitRequest(BaseModel):
    model_path: str
    model_base: Optional[str] = None

@app.post("/initialize")
async def initialize_model(request: InitRequest):
    """Initialize the FastVLM model"""
    global predictor
    try:
        predictor = FastVLMPredictor(request.model_path, request.model_base)
        return {"status": "success", "message": "Model initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")

async def run_prediction_async(image: Image.Image, prompt: str, temperature: float):
    """Run prediction in thread pool to avoid blocking"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, predictor.predict, image, prompt, temperature)

@app.websocket("/webcam")
async def websocket_webcam(websocket: WebSocket):
    """WebSocket endpoint for webcam streaming and processing"""
    await websocket.accept()
    
    if not webcam_manager.start_camera():
        await websocket.send_json({"error": "Could not start camera"})
        return
    
    current_prediction_task = None
    
    try:
        while True:
            # Check for new client message (non-blocking)
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                system_prompt = message.get("system_prompt", "Describe what you see in this image.")
                temperature = message.get("temperature", 0.2)
            except asyncio.TimeoutError:
                # Use defaults if no new message
                system_prompt = "Describe what you see in this image."
                temperature = 0.2
            
            # Get frame from camera
            frame = webcam_manager.get_frame()
            if frame is None:
                await websocket.send_json({"error": "Could not capture frame"})
                continue
            
            # Convert frame to base64 (optimized)
            # Resize frame for faster encoding
            display_frame = cv2.resize(frame, (480, 360))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # Lower quality for speed
            _, buffer = cv2.imencode('.jpg', display_frame, encode_param)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Check if we have a completed prediction
            prediction = ""
            processed = False
            
            if current_prediction_task and current_prediction_task.done():
                try:
                    prediction = current_prediction_task.result()
                    processed = True
                    current_prediction_task = None
                except Exception as e:
                    prediction = f"Prediction error: {str(e)}"
                    processed = True
                    current_prediction_task = None
            
            # Start new prediction if needed and no task is running
            if (webcam_manager.should_process_frame() and 
                predictor is not None and 
                current_prediction_task is None):
                try:
                    # Convert BGR to RGB and create PIL Image
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize for faster processing
                    rgb_frame = cv2.resize(rgb_frame, (384, 384))
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Start async prediction
                    current_prediction_task = asyncio.create_task(
                        run_prediction_async(pil_image, system_prompt, temperature)
                    )
                except Exception as e:
                    print(f"Error starting prediction: {e}")
            
            # Send response immediately (don't wait for prediction)
            response = {
                "frame": frame_b64,
                "prediction": prediction,
                "frame_count": webcam_manager.frame_count,
                "processed": processed,
                "timestamp": datetime.now().isoformat(),
                "processing": current_prediction_task is not None and not current_prediction_task.done()
            }
            
            await websocket.send_json(response)
            
            # Control frame rate - reduced sleep for smoother streaming
            await asyncio.sleep(0.033)  # ~30 FPS
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup
        if current_prediction_task:
            current_prediction_task.cancel()
        webcam_manager.stop_camera()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FastVLM Webcam App - Optimized</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: white;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .setup-section {
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 20px;
            }
            .video-section {
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }
            .video-container {
                flex: 1;
                min-width: 300px;
            }
            .predictions-container {
                flex: 1;
                min-width: 300px;
            }
            #videoElement {
                width: 100%;
                max-width: 640px;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            .controls {
                margin: 20px 0;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                align-items: center;
            }
            input, textarea, button, select {
                padding: 10px;
                border: none;
                border-radius: 8px;
                font-size: 14px;
            }
            input, textarea, select {
                background: rgba(255, 255, 255, 0.9);
                color: #333;
                flex: 1;
                min-width: 200px;
            }
            button {
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                color: white;
                cursor: pointer;
                font-weight: bold;
                transition: transform 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            .status {
                padding: 10px;
                border-radius: 8px;
                margin: 10px 0;
                font-weight: bold;
            }
            .status.success { background: rgba(46, 204, 113, 0.3); }
            .status.error { background: rgba(231, 76, 60, 0.3); }
            .status.info { background: rgba(52, 152, 219, 0.3); }
            .status.processing { background: rgba(243, 156, 18, 0.3); }
            .prediction-box {
                background: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                border-left: 4px solid #3498db;
                max-height: 150px;
                overflow-y: auto;
            }
            .frame-info {
                font-size: 12px;
                opacity: 0.8;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .processing-indicator {
                font-size: 10px;
                background: rgba(243, 156, 18, 0.7);
                padding: 2px 6px;
                border-radius: 4px;
                color: white;
            }
            .prediction-text {
                line-height: 1.6;
            }
            .performance-stats {
                font-size: 12px;
                opacity: 0.7;
                margin-top: 10px;
            }
            .fps-counter {
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 5px;
                border-radius: 5px;
                font-size: 12px;
            }
            .video-wrapper {
                position: relative;
            }
            @media (max-width: 768px) {
                .video-section {
                    flex-direction: column;
                }
                .controls {
                    flex-direction: column;
                }
                input, textarea, button {
                    width: 100%;
                    margin: 5px 0;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 FastVLM Webcam App - Optimized</h1>
            
            <div class="setup-section">
                <h3>Model Setup</h3>
                <div class="controls">
                    <input type="text" id="modelPath" placeholder="Model path (e.g., ./checkpoints/llava-fastvithd_0.5b_stage2/llava-fastvithd_0.5b_stage2)" 
                           value="./checkpoints/llava-fastvithd_0.5b_stage2/llava-fastvithd_0.5b_stage2">
                    <button onclick="initializeModel()">Initialize Model</button>
                </div>
                <div id="modelStatus" class="status info">Model not initialized</div>
            </div>
            
            <div class="setup-section">
                <h3>System Prompt & Settings</h3>
                <div class="controls">
                    <textarea id="systemPrompt" rows="2" placeholder="Enter your system prompt...">Describe what you see in this image briefly.</textarea>
                </div>
                <div class="controls">
                    <label style="margin-right: 10px;">Temperature:</label>
                    <input type="range" id="temperature" min="0" max="1" step="0.1" value="0.2" style="flex: 1;">
                    <span id="tempValue">0.2</span>
                </div>
            </div>
            
            <div class="video-section">
                <div class="video-container">
                    <h3>📹 Live Webcam Feed</h3>
                    <div class="video-wrapper">
                        <img id="videoElement" alt="Webcam feed will appear here" style="background: #000;">
                        <div id="fpsCounter" class="fps-counter">FPS: --</div>
                    </div>
                    <div class="controls">
                        <button id="startBtn" onclick="startWebcam()">Start Webcam</button>
                        <button id="stopBtn" onclick="stopWebcam()" disabled>Stop Webcam</button>
                    </div>
                    <div id="webcamStatus" class="status info">Webcam not started</div>
                    <div class="performance-stats">
                        <div>Frames processed: <span id="processedCount">0</span></div>
                        <div>Total frames: <span id="totalFrames">0</span></div>
                        <div>Processing rate: <span id="processingRate">--</span></div>
                    </div>
                </div>
                
                <div class="predictions-container">
                    <h3>🤖 AI Predictions</h3>
                    <div id="predictionsContainer">
                        <p style="text-align: center; opacity: 0.7;">AI predictions will appear here...</p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let websocket = null;
            let isModelInitialized = false;
            let isWebcamRunning = false;
            let frameCount = 0;
            let processedCount = 0;
            let lastFrameTime = Date.now();
            let fpsArray = [];

            // Update temperature display
            document.getElementById('temperature').addEventListener('input', function() {
                document.getElementById('tempValue').textContent = this.value;
            });

            async function initializeModel() {
                const modelPath = document.getElementById('modelPath').value;
                const statusDiv = document.getElementById('modelStatus');
                
                if (!modelPath) {
                    statusDiv.innerHTML = 'Please enter a model path';
                    statusDiv.className = 'status error';
                    return;
                }
                
                statusDiv.innerHTML = 'Initializing model...';
                statusDiv.className = 'status info';
                
                try {
                    const response = await fetch('/initialize', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            model_path: modelPath
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        statusDiv.innerHTML = '✅ Model initialized successfully!';
                        statusDiv.className = 'status success';
                        isModelInitialized = true;
                    } else {
                        statusDiv.innerHTML = `❌ Error: ${result.detail}`;
                        statusDiv.className = 'status error';
                        isModelInitialized = false;
                    }
                } catch (error) {
                    statusDiv.innerHTML = `❌ Network error: ${error.message}`;
                    statusDiv.className = 'status error';
                    isModelInitialized = false;
                }
            }

            function startWebcam() {
                if (!isModelInitialized) {
                    alert('Please initialize the model first!');
                    return;
                }
                
                const statusDiv = document.getElementById('webcamStatus');
                statusDiv.innerHTML = 'Connecting to webcam...';
                statusDiv.className = 'status info';
                
                websocket = new WebSocket(`ws://localhost:8000/webcam`);
                
                websocket.onopen = function(event) {
                    statusDiv.innerHTML = '✅ Webcam connected and streaming!';
                    statusDiv.className = 'status success';
                    isWebcamRunning = true;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    
                    // Reset counters
                    frameCount = 0;
                    processedCount = 0;
                    lastFrameTime = Date.now();
                    fpsArray = [];
                    
                    // Start receiving frames
                    requestNextFrame();
                };
                
                websocket.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.error) {
                        statusDiv.innerHTML = `❌ ${data.error}`;
                        statusDiv.className = 'status error';
                        return;
                    }
                    
                    // Update FPS calculation
                    const now = Date.now();
                    const timeDiff = now - lastFrameTime;
                    lastFrameTime = now;
                    
                    if (timeDiff > 0) {
                        const fps = 1000 / timeDiff;
                        fpsArray.push(fps);
                        if (fpsArray.length > 10) fpsArray.shift();
                        
                        const avgFps = fpsArray.reduce((a, b) => a + b, 0) / fpsArray.length;
                        document.getElementById('fpsCounter').textContent = `FPS: ${avgFps.toFixed(1)}`;
                    }
                    
                    // Update video feed
                    if (data.frame) {
                        document.getElementById('videoElement').src = `data:image/jpeg;base64,${data.frame}`;
                        frameCount++;
                        document.getElementById('totalFrames').textContent = frameCount;
                    }
                    
                    // Update processing indicator
                    if (data.processing) {
                        statusDiv.innerHTML = '🔄 Processing frame...';
                        statusDiv.className = 'status processing';
                    } else {
                        statusDiv.innerHTML = '✅ Webcam connected and streaming!';
                        statusDiv.className = 'status success';
                    }
                    
                    // Update predictions
                    if (data.processed && data.prediction) {
                        processedCount++;
                        document.getElementById('processedCount').textContent = processedCount;
                        document.getElementById('processingRate').textContent = 
                            `${((processedCount / frameCount) * 100).toFixed(1)}%`;
                        
                        addPrediction(data.prediction, data.frame_count, data.timestamp);
                    }
                    
                    // Continue the loop if webcam is still running
                    if (isWebcamRunning) {
                        setTimeout(requestNextFrame, 10); // Small delay for smooth streaming
                    }
                };
                
                websocket.onerror = function(error) {
                    statusDiv.innerHTML = `❌ WebSocket error`;
                    statusDiv.className = 'status error';
                };
                
                websocket.onclose = function(event) {
                    statusDiv.innerHTML = 'Webcam disconnected';
                    statusDiv.className = 'status info';
                    isWebcamRunning = false;
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                };
            }

            function requestNextFrame() {
                if (websocket && websocket.readyState === WebSocket.OPEN && isWebcamRunning) {
                    const systemPrompt = document.getElementById('systemPrompt').value;
                    const temperature = parseFloat(document.getElementById('temperature').value);
                    
                    websocket.send(JSON.stringify({
                        system_prompt: systemPrompt,
                        temperature: temperature
                    }));
                }
            }

            function stopWebcam() {
                isWebcamRunning = false;
                if (websocket) {
                    websocket.close();
                }
            }

            function addPrediction(prediction, frameCount, timestamp) {
                const container = document.getElementById('predictionsContainer');
                
                // Create new prediction element
                const predictionDiv = document.createElement('div');
                predictionDiv.className = 'prediction-box';
                
                const timeStr = new Date(timestamp).toLocaleTimeString();
                predictionDiv.innerHTML = `
                    <div class="frame-info">
                        <span>Frame ${frameCount} • ${timeStr}</span>
                    </div>
                    <div class="prediction-text">${prediction}</div>
                `;
                
                // Add to top of container
                if (container.children.length === 1 && container.children[0].tagName === 'P') {
                    container.innerHTML = '';
                }
                
                container.insertBefore(predictionDiv, container.firstChild);
                
                // Keep only the last 3 predictions to save memory
                while (container.children.length > 3) {
                    container.removeChild(container.lastChild);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastVLM Webcam Web App - Optimized")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind to")
    parser.add_argument("--model-path", default=None, help="Path to FastVLM model")
    
    args = parser.parse_args()
    
    print(f"Starting Optimized FastVLM Webcam App on {args.host}:{args.port}")
    print("Open your browser and go to: http://localhost:8000")
    
    # Initialize model if provided
    if args.model_path:
        try:
            predictor = FastVLMPredictor(args.model_path)
            print("Model pre-loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not pre-load model: {e}")
    
    uvicorn.run(app, host=args.host, port=args.port)
