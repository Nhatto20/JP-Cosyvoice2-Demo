import gradio as gr
import torch
import torchaudio
import numpy as np
from cosyvoice.cli.cosyvoice import CosyVoice2
import tempfile
import os
import time
from pathlib import Path
from typing import Optional, Dict

import sys
sys.path.append('third_party/Matcha-TTS')

# Metrics imports (only loaded if needed)
try:
    from jiwer import wer, cer
    from speechbrain.pretrained import SpeakerRecognition
    from faster_whisper import WhisperModel
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: Metrics libraries not available. Install: pip install faster-whisper speechbrain jiwer")

# Initialize CosyVoice2 model
MODEL_DIR = "./pretrained_models/JP_CosyVoice2_finetune"
cosyvoice = CosyVoice2(MODEL_DIR)

# Initialize metrics models (lazy loading)
ASR_MODEL = None
SPEAKER_MODEL = None

def init_metrics_models():
    """Lazy initialization of metrics models"""
    global ASR_MODEL, SPEAKER_MODEL
    
    if not METRICS_AVAILABLE:
        return False
    
    if ASR_MODEL is None:
        try:
            ASR_MODEL = WhisperModel(
                "large-v3",
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="float16" if torch.cuda.is_available() else "int8",
                num_workers=4,
                cpu_threads=4
            )
        except Exception as e:
            print(f"Failed to load ASR model: {e}")
            return False
    
    if SPEAKER_MODEL is None:
        try:
            SPEAKER_MODEL = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec"
            )
        except Exception as e:
            print(f"Failed to load Speaker model: {e}")
            return False
    
    return True

def load_audio(audio_path, target_sr=16000):
    """Load and resample audio to 16kHz"""
    if audio_path is None:
        return None
    speech, sr = torchaudio.load(audio_path)
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        speech = resampler(speech)
    return speech

def save_audio(audio_tensor, sample_rate=22050):
    """Save audio tensor to temporary file"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    torchaudio.save(temp_file.name, audio_tensor, sample_rate)
    return temp_file.name

def run_asr(wav_path: str, language: str = "ja") -> str:
    """Run ASR using Faster-Whisper"""
    if ASR_MODEL is None:
        return ""
    
    segments, info = ASR_MODEL.transcribe(
        wav_path,
        beam_size=5,
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    text = " ".join([segment.text for segment in segments])
    return text.strip()

def get_embedding(wav_path: str):
    """Get speaker embedding"""
    if SPEAKER_MODEL is None:
        return None
    
    wav, sr = torchaudio.load(wav_path)
    emb = SPEAKER_MODEL.encode_batch(wav).squeeze()
    if emb.dim() > 1:
        emb = emb.squeeze()
    return emb.detach()

def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings"""
    if a is None or b is None:
        return 0.0
    
    a = a.view(-1).unsqueeze(0)
    b = b.view(-1).unsqueeze(0)
    return torch.nn.functional.cosine_similarity(a, b, dim=1).item()

def calculate_metrics(generated_audio_path: str, reference_text: str, reference_audio_path: Optional[str], language: str = "ja") -> Dict:
    """
    Calculate CER, SIM metrics
    
    Args:
        generated_audio_path: Path to generated audio
        reference_text: Ground truth text
        reference_audio_path: Path to reference audio (for speaker similarity)
        language: Language code for ASR
    
    Returns:
        Dictionary with metrics
    """
    metrics = {
        "cer": None,
        "sim": None,
        "error": None
    }
    
    if not init_metrics_models():
        metrics["error"] = "Metrics models not available"
        return metrics
    
    try:
        # Calculate CER
        if reference_text and reference_text.strip():
            hypothesis = run_asr(generated_audio_path, language)
            if hypothesis:
                cer_value = cer(reference_text, hypothesis)
                metrics["cer"] = round(cer_value * 100, 2)  # Convert to percentage
        
        # Calculate Speaker Similarity
        if reference_audio_path and os.path.exists(reference_audio_path):
            gen_emb = get_embedding(generated_audio_path)
            ref_emb = get_embedding(reference_audio_path)
            
            if gen_emb is not None and ref_emb is not None:
                sim_value = cosine_similarity(gen_emb, ref_emb)
                metrics["sim"] = round(sim_value * 100, 2)  # Convert to percentage
    
    except Exception as e:
        metrics["error"] = str(e)
    
    return metrics

def format_metrics(metrics: Dict) -> str:
    """Format metrics dictionary to readable string"""
    if metrics.get("error"):
        return f"⚠️ Error: {metrics['error']}"
    
    output = "📊 **Metrics Results:**\n\n"
    
    if metrics.get("cer") is not None:
        output += f"- **CER (Character Error Rate)**: {metrics['cer']}%\n"
    
    if metrics.get("sim") is not None:
        output += f"- **SIM (Speaker Similarity)**: {metrics['sim']}%\n"
    
    if metrics.get("cer") is None and metrics.get("sim") is None:
        output = "⚠️ No metrics available"
    
    return output

def format_timing(first_chunk_time: float, total_time: float, stream_mode: str) -> str:
    """Format timing information"""
    output = "⏱️ **Timing Information:**\n\n"
    
    if stream_mode == "Streaming" and first_chunk_time > 0:
        output += f"- **Time to First Chunk**: {first_chunk_time:.2f}s\n"
    
    output += f"- **Total Generation Time**: {total_time:.2f}s\n"
    
    return output

def generate_with_metrics(
    generate_fn,
    stream_mode: str,
    enable_metrics: bool,
    reference_text: str,
    reference_audio: Optional[str],
    language: str,
    *args
):
    """
    Wrapper function to add metrics and timing to any generation function
    
    Args:
        generate_fn: The actual generation function
        stream_mode: "Streaming" or "Non-streaming"
        enable_metrics: Whether to calculate metrics
        reference_text: Reference text for CER calculation
        reference_audio: Reference audio for SIM calculation
        language: Language code for ASR
        *args: Arguments to pass to generate_fn
    """
    start_time = time.time()
    first_chunk_time = 0
    generated_audio_path = None
    
    # Track first chunk time for streaming
    chunk_count = 0
    
    for result in generate_fn(*args):
        chunk_count += 1
        
        # Record time to first chunk
        if chunk_count == 1 and stream_mode == "Streaming":
            first_chunk_time = time.time() - start_time
        
        # Extract audio path if it's the final result
        if isinstance(result, tuple) and len(result) == 2:
            audio_data, audio_path = result
            if audio_path:
                generated_audio_path = audio_path
            yield audio_data, None, "", ""
        else:
            yield result, None, "", ""
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Format timing
    timing_info = format_timing(first_chunk_time, total_time, stream_mode)
    
    # Calculate metrics if enabled
    metrics_info = ""
    if enable_metrics and generated_audio_path:
        metrics = calculate_metrics(
            generated_audio_path,
            reference_text,
            reference_audio,
            language
        )
        metrics_info = format_metrics(metrics)
    
    # Yield final result with metrics and timing
    if generated_audio_path:
        final_audio, _ = torchaudio.load(generated_audio_path)
        final_audio = final_audio.numpy().flatten()
        yield (cosyvoice.sample_rate, final_audio), generated_audio_path, timing_info, metrics_info
    else:
        yield None, None, timing_info, metrics_info

def generate_zero_shot(text, prompt_text, prompt_audio, stream_mode, speed, text_frontend, enable_metrics, ref_text, language):
    """Zero-shot inference with metrics and auto ASR for prompt text"""
    if prompt_audio is None:
        raise gr.Error("Please upload a prompt audio file")
    
    # Auto-generate prompt_text using ASR if not provided
    auto_generated = False
    if not prompt_text or prompt_text.strip() == "":
        if not init_metrics_models():
            raise gr.Error("ASR model not available. Please provide prompt text manually.")
        
        gr.Info("🎤 Auto-generating prompt text using ASR...")
        prompt_text = run_asr(prompt_audio, language)
        auto_generated = True
        
        if not prompt_text:
            raise gr.Error("Failed to transcribe prompt audio. Please provide prompt text manually.")
        
        gr.Info(f"✅ Generated prompt text: {prompt_text[:100]}...")
    
    def _generate():
        prompt_speech = load_audio(prompt_audio)
        stream = stream_mode == "Streaming"
        output_audios = []
        
        for i, output in enumerate(cosyvoice.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_speech_16k=prompt_speech,
            stream=stream,
            speed=speed,
            text_frontend=text_frontend
        )):
            audio = output['tts_speech'].numpy().flatten()
            output_audios.append(audio)
            
            if stream and i < 10:
                temp_audio = np.concatenate(output_audios)
                yield (cosyvoice.sample_rate, temp_audio), None
        
        final_audio = np.concatenate(output_audios)
        final_path = save_audio(torch.from_numpy(final_audio).unsqueeze(0), cosyvoice.sample_rate)
        yield (cosyvoice.sample_rate, final_audio), final_path
    
    # Use tts_text as reference if no ref_text provided
    final_ref_text = ref_text if (ref_text and ref_text.strip()) else text
    
    yield from generate_with_metrics(
        _generate,
        stream_mode,
        enable_metrics,
        final_ref_text if enable_metrics else "",
        prompt_audio if enable_metrics else None,
        language,
    )

# Build Gradio Interface
with gr.Blocks(title="CosyVoice2 Zero-Shot TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ CosyVoice2 Zero-Shot Voice Cloning
    
    Clone any voice with just a short audio sample (3-10 seconds).
    
    ⚠️ **Note:** The first inference will take some time to load the model. Subsequent generations will be faster.
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Settings")
            
            zs_text = gr.Textbox(
                label="Text to synthesize", 
                placeholder="Enter the text you want to generate...", 
                lines=3
            )
            zs_prompt_text = gr.Textbox(
                label="Prompt text (optional)", 
                placeholder="Leave empty for auto-transcription...", 
                lines=2
            )
            zs_prompt_audio = gr.Audio(
                label="Prompt audio (3-10s)", 
                type="filepath"
            )
            
            gr.Markdown("### Generation Settings")
            
            with gr.Row():
                zs_stream = gr.Radio(
                    choices=["Streaming", "Non-streaming"], 
                    value="Non-streaming", 
                    label="Mode"
                )
                zs_speed = gr.Slider(
                    minimum=0.5, 
                    maximum=2.0, 
                    value=1.0, 
                    step=0.1, 
                    label="Speed"
                )
            
            zs_frontend = gr.Checkbox(
                value=True, 
                label="Enable text normalization"
            )
            
            with gr.Accordion("📊 Advanced Metrics (Optional)", open=False):
                gr.Markdown("""
                **Instructions:**
                - Enable metrics to calculate CER (Character Error Rate) and Speaker Similarity
                - Reference text is optional - if not provided, will use your input text
                - Select the correct language for accurate transcription
                - Note: Enabling metrics will increase processing time
                """)
                
                zs_enable_metrics = gr.Checkbox(
                    value=False, 
                    label="Enable metrics"
                )
                zs_ref_text = gr.Textbox(
                    label="Reference text (optional)", 
                    placeholder="Leave empty to use input text..."
                )
                zs_language = gr.Dropdown(
                    choices=["ja", "en", "zh", "ko"], 
                    value="ja", 
                    label="Language"
                )
            
            zs_btn = gr.Button("🎤 Generate Voice", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("### Output")
            
            zs_audio_stream = gr.Audio(
                label="🔊 Streaming Preview", 
                autoplay=True,
                visible=True
            )
            zs_audio_final = gr.Audio(
                label="✅ Final Output",
                visible=True
            )
            zs_timing = gr.Markdown(
                label="⏱️ Performance",
                value=""
            )
            zs_metrics = gr.Markdown(
                label="📊 Quality Metrics",
                value=""
            )
    
    zs_btn.click(
        fn=generate_zero_shot,
        inputs=[
            zs_text, 
            zs_prompt_text, 
            zs_prompt_audio, 
            zs_stream, 
            zs_speed, 
            zs_frontend, 
            zs_enable_metrics, 
            zs_ref_text, 
            zs_language
        ],
        outputs=[
            zs_audio_stream, 
            zs_audio_final, 
            zs_timing, 
            zs_metrics
        ]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)