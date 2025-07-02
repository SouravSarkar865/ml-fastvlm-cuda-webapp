# FastVLM: Efficient Vision Encoding for Vision Language Models

This is the official repository of
**[FastVLM: Efficient Vision Encoding for Vision Language Models](https://www.arxiv.org/abs/2412.13303). (CVPR 2025)**

[//]: # (![FastViTHD Performance]&#40;docs/acc_vs_latency_qwen-2.png&#41;)
<p align="center">
<img src="docs/acc_vs_latency_qwen-2.png" alt="Accuracy vs latency figure." width="400"/>
</p>

### Highlights
* We introduce FastViTHD, a novel hybrid vision encoder designed to output fewer tokens and significantly reduce encoding time for high-resolution images.  
* Our smallest variant outperforms LLaVA-OneVision-0.5B with 85x faster Time-to-First-Token (TTFT) and 3.4x smaller vision encoder.
* Our larger variants using Qwen2-7B LLM outperform recent works like Cambrian-1-8B while using a single image encoder with a 7.9x faster TTFT.
* Demo iOS app to demonstrate the performance of our model on a mobile device.
* **🚀 NEW**: Real-time webcam web interface for live vision-language processing.

<table>
<tr>
    <td><img src="docs/fastvlm-counting.gif" alt="FastVLM - Counting"></td>
    <td><img src="docs/fastvlm-handwriting.gif" alt="FastVLM - Handwriting"></td>
    <td><img src="docs/fastvlm-emoji.gif" alt="FastVLM - Emoji"></td>
</tr>
</table>

## Getting Started
We use LLaVA codebase to train FastVLM variants. In order to train or finetune your own variants, 
please follow instructions provided in [LLaVA](https://github.com/haotian-liu/LLaVA) codebase. 
We provide instructions for running inference with our models.   

### Setup
```bash
conda create -n fastvlm python=3.10
conda activate fastvlm
pip install -e .
```

### Model Zoo
For detailed information on various evaluations, please refer to our [paper](https://www.arxiv.org/abs/2412.13303).

| Model        | Stage |                                            Pytorch Checkpoint (url)                                             |
|:-------------|:-----:|:---------------------------------------------------------------------------------------------------------------:|
| FastVLM-0.5B |   2   | [fastvlm_0.5b_stage2](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage2.zip) |
|              |   3   | [fastvlm_0.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip) |
| FastVLM-1.5B |   2   | [fastvlm_1.5b_stage2](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage2.zip) |
|              |   3   | [fastvlm_1.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip)  |
| FastVLM-7B   |   2   | [fastvlm_7b_stage2](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage2.zip)  |
|              |   3   | [fastvlm_7b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3.zip)  |

To download all the pretrained checkpoints run the command below (note that this might take some time depending on your connection so might be good to grab ☕️ while you wait).

```bash
bash get_models.sh   # Files will be downloaded to `checkpoints` directory.
```

## Usage

### Command Line Inference
To run inference of PyTorch checkpoint, follow the instruction below
```bash
python predict.py --model-path /path/to/checkpoint-dir \
                  --image-file /path/to/image.png \
                  --prompt "Describe the image."
```

### 🌐 Real-time Webcam Web Interface

We've added a powerful web-based interface that allows real-time vision-language processing using your webcam. This feature processes every 10th frame from your webcam stream and generates AI descriptions based on custom prompts.

#### Features
- **📹 Live Webcam Streaming**: Real-time video feed processing
- **🤖 AI-Powered Analysis**: Custom system prompts for different use cases
- **⚡ Optimized Performance**: Asynchronous processing to maintain smooth video streaming
- **📊 Performance Monitoring**: Real-time FPS counter and processing statistics
- **🎨 Modern UI**: Responsive design with glassmorphism effects

#### Setup for Web Interface
Install additional dependencies:
```bash
conda activate fastvlm
pip install fastapi uvicorn websockets opencv-python python-multipart
```

#### Running the Web Interface
Start the web application:
```bash
# Option 1: Pre-load a specific model
python fastvlm_webapp.py --model-path ./checkpoints/llava-fastvithd_0.5b_stage2/llava-fastvithd_0.5b_stage2

# Option 2: Initialize model through web UI
python fastvlm_webapp.py
```

Then open your browser and navigate to: **http://localhost:8000**

#### Web Interface Usage
1. **Initialize Model**: Enter your model path and click "Initialize Model"
2. **Set System Prompt**: Customize the AI prompt (e.g., "Describe what you see", "Count objects", "Identify safety concerns")
3. **Adjust Settings**: Set temperature for generation creativity
4. **Start Webcam**: Click "Start Webcam" to begin live processing
5. **View Results**: AI predictions appear in real-time alongside the video feed

#### Example System Prompts
- `"Describe what you see in this image briefly"`
- `"Count the number of people in the image"`
- `"Identify any objects on the desk"`
- `"Describe the scene and any activities happening"`
- `"What safety concerns do you notice?"`

#### Performance Optimization
The web interface includes several optimizations:
- **Smart Frame Processing**: Only every 10th frame is processed to balance performance and real-time response
- **Asynchronous AI Processing**: Predictions run in background without blocking video stream
- **Memory Management**: Automatic GPU cache clearing and garbage collection
- **Adaptive Quality**: Optimized image encoding for smooth streaming

#### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended for best performance)
- **RAM**: 8GB+ recommended
- **Browser**: Modern browser with WebSocket support
- **Camera**: USB webcam or built-in camera

### Inference on Apple Silicon
To run inference on Apple Silicon, pytorch checkpoints have to be exported to format 
suitable for running on Apple Silicon, detailed instructions and code can be found [`model_export`](model_export/) subfolder.
Please see the README there for more details.

For convenience, we provide 3 models that are in Apple Silicon compatible format: [fastvlm_0.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3_llm.fp16.zip), 
[fastvlm_1.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3_llm.int8.zip), 
[fastvlm_7b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3_llm.int4.zip). 
We encourage developers to export the model of their choice with the appropriate quantization levels following 
the instructions in [`model_export`](model_export/).

### Inference on Apple Devices
To run inference on Apple devices like iPhone, iPad or Mac, see [`app`](app/) subfolder for more details.

## File Structure
```
FastVLM/
├── predict.py                 # Command-line inference script
├── fastvlm_webapp.py         # Real-time webcam web interface
├── get_models.sh             # Model download script
├── checkpoints/              # Downloaded model checkpoints
├── llava/                    # Core LLaVA components
├── app/                      # iOS/macOS app code
├── model_export/             # Apple Silicon export tools
└── docs/                     # Documentation and figures
```

## Use Cases

### Real-time Applications
The webcam interface enables various real-time use cases:
- **Security Monitoring**: Describe scenes and identify potential issues
- **Accessibility**: Real-time scene description for visually impaired users
- **Education**: Live object identification and counting
- **Quality Control**: Real-time product inspection and defect detection
- **Interactive Demos**: Showcase vision-language capabilities

### Research Applications
- **Multimodal AI Research**: Test vision-language models in real-time scenarios
- **Human-Computer Interaction**: Develop gesture and activity recognition systems
- **Computer Vision**: Benchmark model performance on live video streams

## Citation
If you found this code useful, please cite the following paper:
```
@InProceedings{fastvlm2025,
  author = {Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari},
  title = {FastVLM: Efficient Vision Encoding for Vision Language Models},
  booktitle = {Proceedings of the IEEE/CVV Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2025},
}
```

## Contributing
We welcome contributions! If you've built interesting applications or improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

### Community Extensions
- **Web Interface**: Real-time webcam processing with FastAPI (this fork)
- Feel free to add your own extensions and submit PRs!

## Acknowledgements
Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details. 

## License
Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.