# JP-CosyVoice2 Demo

[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=CosyVoice🎙️&text2=Japanese%20Text-to-Speech%20Model&width=800&height=210)](https://github.com/Nhatto20/JP-Cosyvoice2-Demo)

## 🌟 Overview

This is a Japanese-optimized implementation of CosyVoice2, a state-of-the-art text-to-speech system. This repository provides fine-tuned models and tools specifically designed for Japanese language synthesis with natural prosody and high fidelity.

## ✨ Features

- 🎯 **Zero-shot Voice Cloning**: Clone any voice with just 3-10 seconds of audio
- 🎤 **SFT (Speaker Fine-Tuning)**: Use pre-trained Japanese speaker voices
- 🌍 **Cross-lingual Synthesis**: Generate Japanese speech with voice characteristics from other languages
- 📝 **Instruction-based Control**: Fine-grained control over speaking style and emotion
- 🔄 **Voice Conversion**: Convert one voice to sound like another
- ⚡ **Ultra-low Latency**: Streaming support with 150ms first packet latency
- 🎨 **High Quality**: Natural prosody and accurate pronunciation for Japanese

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- Conda package manager

### Installation

1. **Clone the repository**

```bash
git clone --recursive https://github.com/Nhatto20/JP-Cosyvoice2-Demo.git

# If submodule cloning fails, run:
cd JP-Cosyvoice2-Demo
git submodule update --init --recursive
```

2. **Create and activate conda environment**

```bash
conda create -n cosyvoice_jp python=3.10 -y
conda activate cosyvoice_jp
```

3. **Install dependencies**

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# Install sox for audio processing
# Ubuntu/Debian:
sudo apt-get install sox libsox-dev

# CentOS/RHEL:
sudo yum install sox sox-devel
```

4. **Download the pre-trained model**

```python
from modelscope import snapshot_download

# Download Japanese fine-tuned model
snapshot_download('o6Dool/JP_CosyVoice2_finetune', 
                  local_dir='pretrained_models/JP_CosyVoice2_finetune')

# Optional: Download text processing resources
snapshot_download('iic/CosyVoice-ttsfrd', 
                  local_dir='pretrained_models/CosyVoice-ttsfrd')
```

Or using git:

```bash
mkdir -p pretrained_models
git clone https://huggingface.co/o6Dool/JP_CosyVoice2_finetune pretrained_models/JP_CosyVoice2_finetune
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

5. **Optional: Install ttsfrd for better text normalization**

```bash
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
cd ../..
```

## 📖 Usage Guide

### Basic Setup

```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# Initialize the model
cosyvoice = CosyVoice2(
    './pretrained_models/JP_CosyVoice2_finetune',
    load_jit=False,
    load_trt=False,
    load_vllm=False,
    fp16=False
)
```

### 1. Zero-shot Voice Cloning

Clone any voice using a short audio sample and its transcript.

**Example:**

```python
# Load prompt audio (3-10 seconds recommended)
prompt_speech_16k = load_wav('./examples/prompt_japanese.wav', 16000)

# Generate speech with cloned voice
for i, j in enumerate(cosyvoice.inference_zero_shot(
    tts_text='今日はとても良い天気ですね。散歩に出かけませんか。',
    prompt_text='今片思いの人がいるのですが、片思いの人は今忙しくて、メールが返ってきません。',
    prompt_speech_16k=prompt_speech_16k,
    stream=False
)):
    torchaudio.save(f'zero_shot_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

**Save speaker for reuse:**

```python
# Register the voice
cosyvoice.add_zero_shot_spk(
    prompt_text='今片思いの人がいるのですが、片思いの人は今忙しくて、メールが返ってきません。',
    prompt_speech_16k=prompt_speech_16k,
    zero_shot_spk_id='voice_example_1'
)

# Generate using saved voice
for i, j in enumerate(cosyvoice.inference_zero_shot(
    tts_text='明日は雨が降るそうです。',
    prompt_text='',
    prompt_speech_16k='',
    zero_shot_spk_id='voice_example_1',
    stream=False
)):
    torchaudio.save(f'zero_shot_saved_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)

# Save speaker information to disk
cosyvoice.save_spkinfo()
```

### 2. SFT (Speaker Fine-Tuning)

Use pre-trained speaker voices from the model.

**Example:**

```python
# List available speakers
print("Available speakers:", cosyvoice.list_available_spks())

# Generate speech with selected speaker
for i, j in enumerate(cosyvoice.inference_sft(
    tts_text='おはようございます。今日も一日頑張りましょう。',
    spk_id='voice_example_1',  # Replace with actual speaker ID
    stream=False
)):
    torchaudio.save(f'sft_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### 3. Cross-lingual Synthesis

Generate Japanese speech using voice characteristics from audio in any language.

**Example:**

```python
# Load prompt audio (can be in any language)
prompt_speech_16k = load_wav('./examples/english_prompt.wav', 16000) # Replace with actual audio path

# Generate Japanese speech with the voice characteristics
for i, j in enumerate(cosyvoice.inference_cross_lingual(
    tts_text='春の訪れとともに、桜の花が満開になりました。',
    prompt_speech_16k=prompt_speech_16k,
    stream=False
)):
    torchaudio.save(f'cross_lingual_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

**With fine-grained control (laughter, emphasis):**

```python
for i, j in enumerate(cosyvoice.inference_cross_lingual(
    tts_text='その話を聞いて、彼は[laughter]笑い出しました[laughter]。',
    prompt_speech_16k=prompt_speech_16k,
    stream=False
)):
    torchaudio.save(f'fine_grained_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### 4. Instruction-based Synthesis

Control speaking style using natural language instructions.

**Example:**

```python
prompt_speech_16k = load_wav('./examples/prompt_japanese.wav', 16000)

# Generate with style instruction
for i, j in enumerate(cosyvoice.inference_instruct2(
    tts_text='この本はとても面白いです。ぜひ読んでみてください。',
    instruct_text='優しく穏やかに話してください',  # "Please speak gently and calmly"
    prompt_speech_16k=prompt_speech_16k,
    stream=False
)):
    torchaudio.save(f'instruct_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

**Different style examples:**

```python
# Energetic style
for i, j in enumerate(cosyvoice.inference_instruct2(
    tts_text='明日は楽しいイベントがあります。',
    instruct_text='元気よく話してください',  # "Please speak energetically"
    prompt_speech_16k=prompt_speech_16k,
    stream=False
)):
    torchaudio.save(f'instruct_energetic_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)

# Slow and clear style
for i, j in enumerate(cosyvoice.inference_instruct2(
    tts_text='説明書をよく読んでください。',
    instruct_text='ゆっくりはっきりと話してください',  # "Please speak slowly and clearly"
    prompt_speech_16k=prompt_speech_16k,
    stream=False
)):
    torchaudio.save(f'instruct_slow_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### 5. Voice Conversion

Convert the voice in one audio file to sound like another.

**Example:**

```python
# Load source audio (voice to convert)
source_speech_16k = load_wav('./examples/source_voice.wav', 16000)

# Load target audio (desired voice)
target_speech_16k = load_wav('./examples/target_voice.wav', 16000)

# Perform voice conversion
for i, j in enumerate(cosyvoice.inference_vc(
    source_speech_16k=source_speech_16k,
    prompt_speech_16k=target_speech_16k,
    stream=False
)):
    torchaudio.save(f'vc_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### 6. Streaming Inference

For real-time applications with low latency.

**Example:**

```python
prompt_speech_16k = load_wav('./examples/prompt_japanese.wav', 16000)

# Enable streaming mode
for i, j in enumerate(cosyvoice.inference_zero_shot(
    tts_text='これはストリーミング推論のテストです。',
    prompt_text='今片思いの人がいるのですが、片思いの人は今忙しくて、メールが返ってきません。',
    prompt_speech_16k=prompt_speech_16k,
    stream=True,  # Enable streaming
    speed=1.0
)):
    # Process each chunk as it arrives
    torchaudio.save(f'stream_chunk_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### 7. Bi-directional Streaming with Text Generator

Useful when integrating with language models.

**Example:**

```python
def text_generator():
    """Generator that yields text chunks"""
    yield '今日は良い天気です。'
    yield '公園に行きました。'
    yield '友達と会いました。'
    yield '楽しい時間を過ごしました。'

prompt_speech_16k = load_wav('./examples/prompt_japanese.wav', 16000)

for i, j in enumerate(cosyvoice.inference_zero_shot(
    tts_text=text_generator(),  # Use generator as input
    prompt_text='今片思いの人がいるのですが、片思いの人は今忙しくて、メールが返ってきません。',
    prompt_speech_16k=prompt_speech_16k,
    stream=False
)):
    torchaudio.save(f'bistream_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

## 🎨 Advanced Features

### Speed Control

Adjust the speaking speed from 0.5x to 2.0x:

```python
for i, j in enumerate(cosyvoice.inference_zero_shot(
    tts_text='速度を変えて話すテストです。',
    prompt_text='今片思いの人がいるのですが、片思いの人は今忙しくて、メールが返ってきません。',
    prompt_speech_16k=prompt_speech_16k,
    stream=False,
    speed=1.5  # 1.5x speed
)):
    torchaudio.save(f'fast_speech_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### Text Frontend Control

Disable automatic text normalization if needed:

```python
for i, j in enumerate(cosyvoice.inference_zero_shot(
    tts_text='2024年12月25日',
    prompt_text='今片思いの人がいるのですが、片思いの人は今忙しくて、メールが返ってきません。',
    prompt_speech_16k=prompt_speech_16k,
    stream=False,
    text_frontend=False  # Disable text normalization
)):
    torchaudio.save(f'no_frontend_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

## 🖥️ Web Interface

Launch the Gradio web interface for easy testing:

```bash
python webui.py --port 7860 --model_dir pretrained_models/JP_CosyVoice2_finetune
# Then open your browser and navigate to `http://localhost:7860`
or
python app.py
```

## 📊 Performance Tips

1. **Audio Quality**: Use clean audio with minimal background noise for best results
2. **Prompt Length**: 3-10 seconds is optimal for voice cloning
3. **Prompt Text**: Provide prompt text for better speech synthesis result
4. **GPU Memory**: FP16 mode reduces memory usage by ~50%
5. **Streaming**: Use streaming mode for real-time applications
6. **Text Length**: Split very long texts into smaller chunks for better quality

## 🛠️ Configuration Options

When initializing the model, you can configure:

```python
cosyvoice = CosyVoice2(
    model_dir='./pretrained_models/JP_CosyVoice2_finetune',
    load_jit=False,      # Load JIT-compiled models for faster inference
    load_trt=False,      # Load TensorRT models for GPU acceleration
    load_vllm=False,     # Load vLLM for efficient LLM inference
    fp16=False           # Use FP16 precision (reduces memory, slight quality trade-off)
)
```

## 🔧 Troubleshooting

**Out of Memory Error:**
- Enable FP16 mode: `fp16=True`
- Reduce audio length
- Process in smaller batches

**Poor Audio Quality:**
- Check input audio sample rate (should be 16kHz for prompt audio)
- Ensure prompt audio is clean and clear
- Try different prompt audio samples

**Slow Inference:**
- Enable JIT compilation: `load_jit=True`
- Use GPU acceleration: `load_trt=True`
- Enable streaming mode for incremental output

**Installation Issues:**
- Ensure CUDA is properly installed
- Check Python version (requires 3.10+)
- Try creating a fresh conda environment

## 📚 Additional Resources

- **Original CosyVoice**: [GitHub](https://github.com/FunAudioLLM/CosyVoice)
- **CosyVoice2 Paper**: [arXiv:2412.10117](https://arxiv.org/abs/2412.10117)
- **Online Demo**: [CosyVoice2 Demos](https://colab.research.google.com/drive/1wwcLaavxeFVvRDyI2-2lHLoL3TyxIIIy?usp=sharing)
- **Model on Huggingface**: [o6Dool/JP_CosyVoice2_finetune](https://huggingface.co/o6Dool/JP_CosyVoice2_finetune)

## 🙏 Acknowledgments

This project builds upon the excellent work of:
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - Original TTS framework
- [FunASR](https://github.com/modelscope/FunASR) - ASR toolkit
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) - Flow matching TTS
- [WeNet](https://github.com/wenet-e2e/wenet) - Speech recognition toolkit

## 📄 License

This project is licensed under the Apache License 2.0. See LICENSE file for details.

## 📮 Contact & Support

For questions, issues, or contributions:
- Open an issue on [GitHub](https://github.com/Nhatto20/JP-Cosyvoice2-Demo/issues)
- Check existing issues for solutions
- Contribute via pull requests

## 🌟 Citation
```bibtex
@article{du2024cosyvoice,
  title={Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens},
  author={Du, Zhihao and Chen, Qian and Zhang, Shiliang and Hu, Kai and Lu, Heng and Yang, Yexin and Hu, Hangrui and Zheng, Siqi and Gu, Yue and Ma, Ziyang and others},
  journal={arXiv preprint arXiv:2407.05407},
  year={2024}
}

@article{du2024cosyvoice,
  title={Cosyvoice 2: Scalable streaming speech synthesis with large language models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and Shi, Xian and Lv, Xiang and Zhao, Tianyu and Gao, Zhifu and Yang, Yexin and Gao, Changfeng and Wang, Hui and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}

@inproceedings{lyu2025build,
  title={Build LLM-Based Zero-Shot Streaming TTS System with Cosyvoice},
  author={Lyu, Xiang and Wang, Yuxuan and Zhao, Tianyu and Wang, Hao and Liu, Huadai and Du, Zhihao},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--2},
  year={2025},
  organization={IEEE}
}
```

---