
# # import sys
# # import torch
# # sys.path.append('third_party/Matcha-TTS')

# # import streamlit as st
# # import torchaudio
# # import tempfile
# # import time
# # import base64
# # import numpy as np

# # from cosyvoice.cli.cosyvoice import CosyVoice2
# # from cosyvoice.utils.file_utils import load_wav

# # # -------------------------
# # # Streamlit UI
# # # -------------------------
# # st.set_page_config(page_title="CosyVoice2 Streaming Demo", layout="wide")

# # st.title("🎤 CosyVoice2 Real-time Streaming Demo")
# # st.markdown("### Test zero-shot + streaming inference with real-time playback")

# # # -------------------------
# # # Sidebar settings
# # # -------------------------
# # st.sidebar.header("⚙️ Settings")

# # default_model_path = r"C:\Users\japan\datasets\pretrained_models\JP_CosyVoice2_finetune"

# # model_path = st.sidebar.text_input(
# #     "Model directory",
# #     value=default_model_path,
# #     help="Enter path to CosyVoice2 pretrained model"
# # )

# # uploaded_ref = st.sidebar.file_uploader(
# #     "Reference Audio (16kHz WAV)",
# #     type=["wav"],
# #     help="Optional: Reference speaker audio for zero-shot"
# # )

# # ref_text = st.sidebar.text_area(
# #     "Reference Text (optional)",
# #     "",
# #     help="If empty, model will use only the audio"
# # )

# # # -------------------------
# # # Input text
# # # -------------------------
# # text = st.text_area(
# #     "📌 Input Text for TTS",
# #     "今日はとても良い天気ですね。\nゆっくり話しますので、最後まで聞いてください。",
# #     height=160
# # )

# # btn_run = st.button("🚀 Start Streaming")

# # # -------------------------
# # # Load model (lazy)
# # # -------------------------
# # @st.cache_resource(show_spinner=True)
# # def load_model(path):
# #     return CosyVoice2(
# #         path,
# #         load_jit=False,
# #         load_trt=False,
# #         load_vllm=False,
# #         fp16=False
# #     )

# # if btn_run:

# #     if text.strip() == "":
# #         st.error("❗ Please enter some text.")
# #         st.stop()

# #     cosyvoice = load_model(model_path)
# #     st.success("✅ Model loaded!")

# #     # ---------------------------------------
# #     # Prepare reference audio
# #     # ---------------------------------------
# #     if uploaded_ref is not None:
# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
# #             f.write(uploaded_ref.read())
# #             ref_path = f.name
# #         prompt_speech_16k = load_wav(ref_path, 16000)
# #     else:
# #         prompt_speech_16k = None
# #         st.warning("⚠️ No reference audio provided. Using model default voice.")

# #     st.write("---")
# #     st.subheader("🔊 Real-time Streaming Output")

# #     audio_placeholder = st.empty()
# #     log_placeholder = st.empty()

# #     # ---------------------------------------
# #     # Streaming inference
# #     # ---------------------------------------
# #     start_t = time.time()
# #     first_packet_t = None

# #     log_placeholder.write("⏳ Streaming started...")

# #     # Buffer to accumulate audio for continuous playback
# #     audio_buffer = np.array([], dtype=np.float32)

# #     # Run streaming TTS
# #     for idx, out in enumerate(
# #         cosyvoice.inference_zero_shot(
# #             text,
# #             ref_text or "",
# #             prompt_speech_16k,
# #             stream=True
# #         )
# #     ):
# #         t_now = time.time()

# #         if first_packet_t is None:
# #             first_packet_t = t_now
# #             log_placeholder.write(
# #                 f"🚀 **First packet latency:** {first_packet_t - start_t:.3f} sec"
# #             )

# #         chunk = out["tts_speech"].numpy().flatten()
# #         audio_buffer = np.concatenate([audio_buffer, chunk])

# #         # Convert to WAV binary buffer for playback
# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
# #             torchaudio.save(f.name, out["tts_speech"], cosyvoice.sample_rate)
# #             audio_bytes = open(f.name, "rb").read()

# #         # 🔥 Real-time audio playback
# #         audio_placeholder.audio(audio_bytes, format="audio/wav")

# #     end_t = time.time()
# #     log_placeholder.write(
# #         f"✔️ **Total time:** {end_t - start_t:.3f} sec\n\n🎉 **Streaming completed!**"
# #     )

# #     # Save final combined audio
# #     st.write("### 💾 Download Full Audio Output")
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
# #         torchaudio.save(f.name,
# #                         torch.tensor(audio_buffer).unsqueeze(0),
# #                         cosyvoice.sample_rate)
# #         st.download_button(
# #             "Download Full WAV",
# #             data=open(f.name, "rb").read(),
# #             file_name="cosyvoice_stream_output.wav",
# #             mime="audio/wav"
# #         )



# import gradio as gr
# import torch
# import torchaudio
# import numpy as np
# from cosyvoice.cli.cosyvoice import CosyVoice2
# import tempfile
# import os
# from pathlib import Path
# import sys
# sys.path.append('third_party/Matcha-TTS')


# # Initialize CosyVoice2 model
# #MODEL_DIR = r"C:\Users\japan\datasets\pretrained_models\JP_CosyVoice2_finetune"  # or your model path
# MODEL_DIR = "/mnt/c/Users/japan/datasets/pretrained_models/JP_CosyVoice2_finetune"
# cosyvoice = CosyVoice2(MODEL_DIR)

# # Get available speakers
# available_spks = cosyvoice.list_available_spks()

# def load_audio(audio_path, target_sr=16000):
#     """Load and resample audio to 16kHz"""
#     if audio_path is None:
#         return None
#     speech, sr = torchaudio.load(audio_path)
#     if speech.shape[0] > 1:
#         speech = speech.mean(dim=0, keepdim=True)
#     if sr != target_sr:
#         resampler = torchaudio.transforms.Resample(sr, target_sr)
#         speech = resampler(speech)
#     return speech

# def save_audio(audio_tensor, sample_rate=22050):
#     """Save audio tensor to temporary file"""
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
#     torchaudio.save(temp_file.name, audio_tensor, sample_rate)
#     return temp_file.name

# def generate_sft(text, spk_id, stream_mode, speed, text_frontend):
#     """SFT inference with pre-defined speaker"""
#     try:
#         stream = stream_mode == "Streaming"
#         output_audios = []
        
#         for i, output in enumerate(cosyvoice.inference_sft(
#             tts_text=text,
#             spk_id=spk_id,
#             stream=stream,
#             speed=speed,
#             text_frontend=text_frontend
#         )):
#             audio = output['tts_speech'].numpy().flatten()
#             output_audios.append(audio)
            
#             # Yield intermediate results for streaming
#             if stream and i < 10:  # Limit streaming chunks displayed
#                 temp_audio = np.concatenate(output_audios)
#                 yield (cosyvoice.sample_rate, temp_audio), None
        
#         # Final output
#         final_audio = np.concatenate(output_audios)
#         final_path = save_audio(torch.from_numpy(final_audio).unsqueeze(0), cosyvoice.sample_rate)
#         yield (cosyvoice.sample_rate, final_audio), final_path
        
#     except Exception as e:
#         raise gr.Error(f"Error: {str(e)}")

# def generate_zero_shot(text, prompt_text, prompt_audio, stream_mode, speed, text_frontend):
#     """Zero-shot inference with custom voice"""
#     try:
#         if prompt_audio is None:
#             raise gr.Error("Please upload a prompt audio file")
        
#         # Load prompt audio
#         prompt_speech = load_audio(prompt_audio)
#         stream = stream_mode == "Streaming"
#         output_audios = []
        
#         for i, output in enumerate(cosyvoice.inference_zero_shot(
#             tts_text=text,
#             prompt_text=prompt_text,
#             prompt_speech_16k=prompt_speech,
#             stream=stream,
#             speed=speed,
#             text_frontend=text_frontend
#         )):
#             audio = output['tts_speech'].numpy().flatten()
#             output_audios.append(audio)
            
#             if stream and i < 10:
#                 temp_audio = np.concatenate(output_audios)
#                 yield (cosyvoice.sample_rate, temp_audio), None
        
#         final_audio = np.concatenate(output_audios)
#         final_path = save_audio(torch.from_numpy(final_audio).unsqueeze(0), cosyvoice.sample_rate)
#         yield (cosyvoice.sample_rate, final_audio), final_path
        
#     except Exception as e:
#         raise gr.Error(f"Error: {str(e)}")

# def generate_cross_lingual(text, prompt_audio, stream_mode, speed, text_frontend):
#     """Cross-lingual inference"""
#     try:
#         if prompt_audio is None:
#             raise gr.Error("Please upload a prompt audio file")
        
#         prompt_speech = load_audio(prompt_audio)
#         stream = stream_mode == "Streaming"
#         output_audios = []
        
#         for i, output in enumerate(cosyvoice.inference_cross_lingual(
#             tts_text=text,
#             prompt_speech_16k=prompt_speech,
#             stream=stream,
#             speed=speed,
#             text_frontend=text_frontend
#         )):
#             audio = output['tts_speech'].numpy().flatten()
#             output_audios.append(audio)
            
#             if stream and i < 10:
#                 temp_audio = np.concatenate(output_audios)
#                 yield (cosyvoice.sample_rate, temp_audio), None
        
#         final_audio = np.concatenate(output_audios)
#         final_path = save_audio(torch.from_numpy(final_audio).unsqueeze(0), cosyvoice.sample_rate)
#         yield (cosyvoice.sample_rate, final_audio), final_path
        
#     except Exception as e:
#         raise gr.Error(f"Error: {str(e)}")

# def generate_instruct2(text, instruct_text, prompt_audio, stream_mode, speed, text_frontend):
#     """Instruction-guided inference"""
#     try:
#         if prompt_audio is None:
#             raise gr.Error("Please upload a prompt audio file")
        
#         prompt_speech = load_audio(prompt_audio)
#         stream = stream_mode == "Streaming"
#         output_audios = []
        
#         for i, output in enumerate(cosyvoice.inference_instruct2(
#             tts_text=text,
#             instruct_text=instruct_text,
#             prompt_speech_16k=prompt_speech,
#             stream=stream,
#             speed=speed,
#             text_frontend=text_frontend
#         )):
#             audio = output['tts_speech'].numpy().flatten()
#             output_audios.append(audio)
            
#             if stream and i < 10:
#                 temp_audio = np.concatenate(output_audios)
#                 yield (cosyvoice.sample_rate, temp_audio), None
        
#         final_audio = np.concatenate(output_audios)
#         final_path = save_audio(torch.from_numpy(final_audio).unsqueeze(0), cosyvoice.sample_rate)
#         yield (cosyvoice.sample_rate, final_audio), final_path
        
#     except Exception as e:
#         raise gr.Error(f"Error: {str(e)}")

# def generate_vc(source_audio, prompt_audio, stream_mode, speed):
#     """Voice conversion"""
#     try:
#         if source_audio is None or prompt_audio is None:
#             raise gr.Error("Please upload both source and prompt audio files")
        
#         source_speech = load_audio(source_audio)
#         prompt_speech = load_audio(prompt_audio)
#         stream = stream_mode == "Streaming"
#         output_audios = []
        
#         for i, output in enumerate(cosyvoice.inference_vc(
#             source_speech_16k=source_speech,
#             prompt_speech_16k=prompt_speech,
#             stream=stream,
#             speed=speed
#         )):
#             audio = output['tts_speech'].numpy().flatten()
#             output_audios.append(audio)
            
#             if stream and i < 10:
#                 temp_audio = np.concatenate(output_audios)
#                 yield (cosyvoice.sample_rate, temp_audio), None
        
#         final_audio = np.concatenate(output_audios)
#         final_path = save_audio(torch.from_numpy(final_audio).unsqueeze(0), cosyvoice.sample_rate)
#         yield (cosyvoice.sample_rate, final_audio), final_path
        
#     except Exception as e:
#         raise gr.Error(f"Error: {str(e)}")

# # Build Gradio Interface
# with gr.Blocks(title="CosyVoice2 TTS") as demo:
#     gr.Markdown("""
#     # 🎙️ CosyVoice2 Text-to-Speech System
    
#     Multi-functional TTS system with support for SFT, Zero-shot, Cross-lingual, Instruct-guided, and Voice Conversion.
#     """)
    
#     with gr.Tabs():
#         # Tab 1: SFT (Pre-defined Speaker)
#         with gr.Tab("🎤 SFT - Pre-defined Speaker"):
#             gr.Markdown("Use pre-trained speaker voices")
#             with gr.Row():
#                 with gr.Column():
#                     sft_text = gr.Textbox(
#                         label="Text to synthesize",
#                         placeholder="Enter text here...",
#                         lines=3
#                     )
#                     sft_spk = gr.Dropdown(
#                         choices=available_spks,
#                         label="Speaker ID",
#                         value=available_spks[0] if available_spks else None
#                     )
#                     with gr.Row():
#                         sft_stream = gr.Radio(
#                             choices=["Streaming", "Non-streaming"],
#                             value="Non-streaming",
#                             label="Mode"
#                         )
#                         sft_speed = gr.Slider(
#                             minimum=0.5,
#                             maximum=2.0,
#                             value=1.0,
#                             step=0.1,
#                             label="Speed"
#                         )
#                     sft_frontend = gr.Checkbox(
#                         value=True,
#                         label="Enable text frontend processing"
#                     )
#                     sft_btn = gr.Button("Generate", variant="primary")
                
#                 with gr.Column():
#                     sft_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
#                     sft_audio_final = gr.Audio(label="Final Output")
            
#             sft_btn.click(
#                 fn=generate_sft,
#                 inputs=[sft_text, sft_spk, sft_stream, sft_speed, sft_frontend],
#                 outputs=[sft_audio_stream, sft_audio_final]
#             )
        
#         # Tab 2: Zero-shot
#         with gr.Tab("🎯 Zero-shot - Custom Voice"):
#             gr.Markdown("Clone any voice with a short audio sample + text")
#             with gr.Row():
#                 with gr.Column():
#                     zs_text = gr.Textbox(
#                         label="Text to synthesize",
#                         placeholder="Enter text here...",
#                         lines=3
#                     )
#                     zs_prompt_text = gr.Textbox(
#                         label="Prompt text (transcript of prompt audio)",
#                         placeholder="Enter the transcript of your audio sample...",
#                         lines=2
#                     )
#                     zs_prompt_audio = gr.Audio(
#                         label="Prompt audio (3-10 seconds recommended)",
#                         type="filepath"
#                     )
#                     with gr.Row():
#                         zs_stream = gr.Radio(
#                             choices=["Streaming", "Non-streaming"],
#                             value="Non-streaming",
#                             label="Mode"
#                         )
#                         zs_speed = gr.Slider(
#                             minimum=0.5,
#                             maximum=2.0,
#                             value=1.0,
#                             step=0.1,
#                             label="Speed"
#                         )
#                     zs_frontend = gr.Checkbox(
#                         value=True,
#                         label="Enable text frontend processing"
#                     )
#                     zs_btn = gr.Button("Generate", variant="primary")
                
#                 with gr.Column():
#                     zs_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
#                     zs_audio_final = gr.Audio(label="Final Output")
            
#             zs_btn.click(
#                 fn=generate_zero_shot,
#                 inputs=[zs_text, zs_prompt_text, zs_prompt_audio, zs_stream, zs_speed, zs_frontend],
#                 outputs=[zs_audio_stream, zs_audio_final]
#             )
        
#         # Tab 3: Cross-lingual
#         with gr.Tab("🌍 Cross-lingual"):
#             gr.Markdown("Synthesize text in different language using voice characteristics")
#             with gr.Row():
#                 with gr.Column():
#                     cl_text = gr.Textbox(
#                         label="Text to synthesize (any language)",
#                         placeholder="Enter text in target language...",
#                         lines=3
#                     )
#                     cl_prompt_audio = gr.Audio(
#                         label="Prompt audio (voice sample)",
#                         type="filepath"
#                     )
#                     with gr.Row():
#                         cl_stream = gr.Radio(
#                             choices=["Streaming", "Non-streaming"],
#                             value="Non-streaming",
#                             label="Mode"
#                         )
#                         cl_speed = gr.Slider(
#                             minimum=0.5,
#                             maximum=2.0,
#                             value=1.0,
#                             step=0.1,
#                             label="Speed"
#                         )
#                     cl_frontend = gr.Checkbox(
#                         value=True,
#                         label="Enable text frontend processing"
#                     )
#                     cl_btn = gr.Button("Generate", variant="primary")
                
#                 with gr.Column():
#                     cl_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
#                     cl_audio_final = gr.Audio(label="Final Output")
            
#             cl_btn.click(
#                 fn=generate_cross_lingual,
#                 inputs=[cl_text, cl_prompt_audio, cl_stream, cl_speed, cl_frontend],
#                 outputs=[cl_audio_stream, cl_audio_final]
#             )
        
#         # Tab 4: Instruct-guided
#         with gr.Tab("📝 Instruct2 - Style Control"):
#             gr.Markdown("Control speaking style with instructions (e.g., 'speak slowly and sadly')")
#             with gr.Row():
#                 with gr.Column():
#                     inst_text = gr.Textbox(
#                         label="Text to synthesize",
#                         placeholder="Enter text here...",
#                         lines=3
#                     )
#                     inst_instruct = gr.Textbox(
#                         label="Style instruction",
#                         placeholder="E.g., 'speak in a cheerful and energetic way'",
#                         lines=2
#                     )
#                     inst_prompt_audio = gr.Audio(
#                         label="Prompt audio (voice sample)",
#                         type="filepath"
#                     )
#                     with gr.Row():
#                         inst_stream = gr.Radio(
#                             choices=["Streaming", "Non-streaming"],
#                             value="Non-streaming",
#                             label="Mode"
#                         )
#                         inst_speed = gr.Slider(
#                             minimum=0.5,
#                             maximum=2.0,
#                             value=1.0,
#                             step=0.1,
#                             label="Speed"
#                         )
#                     inst_frontend = gr.Checkbox(
#                         value=True,
#                         label="Enable text frontend processing"
#                     )
#                     inst_btn = gr.Button("Generate", variant="primary")
                
#                 with gr.Column():
#                     inst_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
#                     inst_audio_final = gr.Audio(label="Final Output")
            
#             inst_btn.click(
#                 fn=generate_instruct2,
#                 inputs=[inst_text, inst_instruct, inst_prompt_audio, inst_stream, inst_speed, inst_frontend],
#                 outputs=[inst_audio_stream, inst_audio_final]
#             )
        
#         # Tab 5: Voice Conversion
#         with gr.Tab("🔄 Voice Conversion"):
#             gr.Markdown("Convert source voice to target voice")
#             with gr.Row():
#                 with gr.Column():
#                     vc_source = gr.Audio(
#                         label="Source audio (voice to convert)",
#                         type="filepath"
#                     )
#                     vc_prompt = gr.Audio(
#                         label="Target audio (target voice)",
#                         type="filepath"
#                     )
#                     with gr.Row():
#                         vc_stream = gr.Radio(
#                             choices=["Streaming", "Non-streaming"],
#                             value="Non-streaming",
#                             label="Mode"
#                         )
#                         vc_speed = gr.Slider(
#                             minimum=0.5,
#                             maximum=2.0,
#                             value=1.0,
#                             step=0.1,
#                             label="Speed"
#                         )
#                     vc_btn = gr.Button("Convert", variant="primary")
                
#                 with gr.Column():
#                     vc_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
#                     vc_audio_final = gr.Audio(label="Final Output")
            
#             vc_btn.click(
#                 fn=generate_vc,
#                 inputs=[vc_source, vc_prompt, vc_stream, vc_speed],
#                 outputs=[vc_audio_stream, vc_audio_final]
#             )
    
#     gr.Markdown("""
#     ---
#     ### 📖 Instructions:
    
#     - **SFT**: Select a pre-trained speaker and enter text
#     - **Zero-shot**: Upload a 3-10s audio sample with its transcript to clone the voice
#     - **Cross-lingual**: Clone voice characteristics across different languages
#     - **Instruct2**: Add style instructions (e.g., "happy", "slow", "whisper")
#     - **Voice Conversion**: Convert one voice to sound like another
    
#     **Streaming vs Non-streaming:**
#     - Streaming: Get audio chunks in real-time (faster feedback)
#     - Non-streaming: Process complete audio at once (better quality)
    
#     **Tips:**
#     - Prompt audio should be clear with minimal background noise
#     - Keep prompt audio between 3-10 seconds for best results
#     - Speed range: 0.5x (slower) to 2.0x (faster)
#     """)

# if __name__ == "__main__":
#     demo.queue()  # Enable queue for streaming
#     demo.launch()



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
MODEL_DIR = "./pretrained_models/CosyVoice2-0.5B"
cosyvoice = CosyVoice2(MODEL_DIR)
available_spks = cosyvoice.list_available_spks()

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

def run_asr(wav_path: str, language: str = "vi") -> str:
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

def calculate_metrics(generated_audio_path: str, reference_text: str, reference_audio_path: Optional[str], language: str = "vi") -> Dict:
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

def generate_sft(text, spk_id, stream_mode, speed, text_frontend, enable_metrics, ref_text, language):
    """SFT inference with metrics"""
    def _generate():
        stream = stream_mode == "Streaming"
        output_audios = []
        
        for i, output in enumerate(cosyvoice.inference_sft(
            tts_text=text,
            spk_id=spk_id,
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
    
    # Get reference audio (use spk2info if available)
    ref_audio = None
    # Note: For SFT, we don't have direct access to reference audio
    
    yield from generate_with_metrics(
        _generate,
        stream_mode,
        enable_metrics,
        ref_text if enable_metrics else "",
        ref_audio,
        language,
    )

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
    
    yield from generate_with_metrics(
        _generate,
        stream_mode,
        enable_metrics,
        ref_text if enable_metrics else "",
        prompt_audio if enable_metrics else None,
        language,
    )

def generate_cross_lingual(text, prompt_audio, stream_mode, speed, text_frontend, enable_metrics, ref_text, language):
    """Cross-lingual inference with metrics"""
    if prompt_audio is None:
        raise gr.Error("Please upload a prompt audio file")
    
    def _generate():
        prompt_speech = load_audio(prompt_audio)
        stream = stream_mode == "Streaming"
        output_audios = []
        
        for i, output in enumerate(cosyvoice.inference_cross_lingual(
            tts_text=text,
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
    
    yield from generate_with_metrics(
        _generate,
        stream_mode,
        enable_metrics,
        ref_text if enable_metrics else "",
        prompt_audio if enable_metrics else None,
        language,
    )

def generate_instruct2(text, instruct_text, prompt_audio, stream_mode, speed, text_frontend, enable_metrics, ref_text, language):
    """Instruction-guided inference with metrics and auto ASR for instruct text"""
    if prompt_audio is None:
        raise gr.Error("Please upload a prompt audio file")
    
    # Auto-generate instruct_text using ASR if not provided
    if not instruct_text or instruct_text.strip() == "":
        if not init_metrics_models():
            raise gr.Error("ASR model not available. Please provide instruction text manually.")
        
        gr.Info("🎤 Auto-generating instruction text using ASR...")
        instruct_text = run_asr(prompt_audio, language)
        
        if not instruct_text:
            raise gr.Error("Failed to transcribe prompt audio. Please provide instruction text manually.")
        
        gr.Info(f"✅ Generated instruction text: {instruct_text[:100]}...")
    
    def _generate():
        prompt_speech = load_audio(prompt_audio)
        stream = stream_mode == "Streaming"
        output_audios = []
        
        for i, output in enumerate(cosyvoice.inference_instruct2(
            tts_text=text,
            instruct_text=instruct_text,
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
    
    yield from generate_with_metrics(
        _generate,
        stream_mode,
        enable_metrics,
        ref_text if enable_metrics else "",
        prompt_audio if enable_metrics else None,
        language,
    )

def generate_vc(source_audio, prompt_audio, stream_mode, speed, enable_metrics, ref_text, language):
    """Voice conversion with metrics"""
    if source_audio is None or prompt_audio is None:
        raise gr.Error("Please upload both source and prompt audio files")
    
    def _generate():
        source_speech = load_audio(source_audio)
        prompt_speech = load_audio(prompt_audio)
        stream = stream_mode == "Streaming"
        output_audios = []
        
        for i, output in enumerate(cosyvoice.inference_vc(
            source_speech_16k=source_speech,
            prompt_speech_16k=prompt_speech,
            stream=stream,
            speed=speed
        )):
            audio = output['tts_speech'].numpy().flatten()
            output_audios.append(audio)
            
            if stream and i < 10:
                temp_audio = np.concatenate(output_audios)
                yield (cosyvoice.sample_rate, temp_audio), None
        
        final_audio = np.concatenate(output_audios)
        final_path = save_audio(torch.from_numpy(final_audio).unsqueeze(0), cosyvoice.sample_rate)
        yield (cosyvoice.sample_rate, final_audio), final_path
    
    yield from generate_with_metrics(
        _generate,
        stream_mode,
        enable_metrics,
        ref_text if enable_metrics else "",
        prompt_audio if enable_metrics else None,
        language,
    )

# Build Gradio Interface
with gr.Blocks(title="CosyVoice2 TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ CosyVoice2 Text-to-Speech System
    
    Multi-functional TTS system with real-time metrics and performance tracking.
    """)
    
    with gr.Tabs():
        # Tab 1: SFT
        with gr.Tab("🎤 SFT - Pre-defined Speaker"):
            with gr.Row():
                with gr.Column():
                    sft_text = gr.Textbox(label="Text to synthesize", placeholder="Enter text here...", lines=3)
                    sft_spk = gr.Dropdown(choices=available_spks, label="Speaker ID", value=available_spks[0] if available_spks else None)
                    
                    with gr.Row():
                        sft_stream = gr.Radio(choices=["Streaming", "Non-streaming"], value="Non-streaming", label="Mode")
                        sft_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")
                    
                    sft_frontend = gr.Checkbox(value=True, label="Enable text frontend")
                    
                    with gr.Accordion("📊 Metrics Settings (Optional)", open=False):
                        sft_enable_metrics = gr.Checkbox(value=False, label="Enable Metrics Calculation")
                        sft_ref_text = gr.Textbox(label="Reference Text (for CER)", placeholder="Ground truth text...")
                        sft_language = gr.Dropdown(choices=["vi", "en", "zh", "ja", "ko"], value="vi", label="Language for ASR")
                    
                    sft_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    sft_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
                    sft_audio_final = gr.Audio(label="Final Output")
                    sft_timing = gr.Markdown(label="Timing")
                    sft_metrics = gr.Markdown(label="Metrics")
            
            sft_btn.click(
                fn=generate_sft,
                inputs=[sft_text, sft_spk, sft_stream, sft_speed, sft_frontend, sft_enable_metrics, sft_ref_text, sft_language],
                outputs=[sft_audio_stream, sft_audio_final, sft_timing, sft_metrics]
            )
        
        # Tab 2: Zero-shot
        with gr.Tab("🎯 Zero-shot - Custom Voice"):
            with gr.Row():
                with gr.Column():
                    zs_text = gr.Textbox(label="Text to synthesize", placeholder="Enter text here...", lines=3)
                    zs_prompt_text = gr.Textbox(
                        label="Prompt text (Optional - Auto ASR if empty)", 
                        placeholder="Leave empty to auto-generate from audio using ASR...", 
                        lines=2
                    )
                    zs_prompt_audio = gr.Audio(label="Prompt audio (3-10s)", type="filepath")
                    
                    with gr.Row():
                        zs_stream = gr.Radio(choices=["Streaming", "Non-streaming"], value="Non-streaming", label="Mode")
                        zs_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")
                    
                    zs_frontend = gr.Checkbox(value=True, label="Enable text frontend")
                    
                    with gr.Accordion("📊 Metrics Settings (Optional)", open=False):
                        zs_enable_metrics = gr.Checkbox(value=False, label="Enable Metrics (CER + SIM)")
                        zs_ref_text = gr.Textbox(label="Reference Text (for CER)", placeholder="Ground truth text...")
                        zs_language = gr.Dropdown(
                            choices=["vi", "en", "zh", "ja", "ko"], 
                            value="vi", 
                            label="Language (for ASR & Metrics)",
                            info="Used for auto-generating prompt text and CER calculation"
                        )
                    
                    zs_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    zs_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
                    zs_audio_final = gr.Audio(label="Final Output")
                    zs_timing = gr.Markdown(label="Timing")
                    zs_metrics = gr.Markdown(label="Metrics")
            
            zs_btn.click(
                fn=generate_zero_shot,
                inputs=[zs_text, zs_prompt_text, zs_prompt_audio, zs_stream, zs_speed, zs_frontend, zs_enable_metrics, zs_ref_text, zs_language],
                outputs=[zs_audio_stream, zs_audio_final, zs_timing, zs_metrics]
            )
        
        # Tab 3: Cross-lingual
        with gr.Tab("🌍 Cross-lingual"):
            with gr.Row():
                with gr.Column():
                    cl_text = gr.Textbox(label="Text to synthesize", placeholder="Enter text in any language...", lines=3)
                    cl_prompt_audio = gr.Audio(label="Prompt audio", type="filepath")
                    
                    with gr.Row():
                        cl_stream = gr.Radio(choices=["Streaming", "Non-streaming"], value="Non-streaming", label="Mode")
                        cl_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")
                    
                    cl_frontend = gr.Checkbox(value=True, label="Enable text frontend")
                    
                    with gr.Accordion("📊 Metrics Settings (Optional)", open=False):
                        cl_enable_metrics = gr.Checkbox(value=False, label="Enable Metrics (CER + SIM)")
                        cl_ref_text = gr.Textbox(label="Reference Text (for CER)", placeholder="Ground truth text...")
                        cl_language = gr.Dropdown(choices=["vi", "en", "zh", "ja", "ko"], value="vi", label="Language")
                    
                    cl_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    cl_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
                    cl_audio_final = gr.Audio(label="Final Output")
                    cl_timing = gr.Markdown(label="Timing")
                    cl_metrics = gr.Markdown(label="Metrics")
            
            cl_btn.click(
                fn=generate_cross_lingual,
                inputs=[cl_text, cl_prompt_audio, cl_stream, cl_speed, cl_frontend, cl_enable_metrics, cl_ref_text, cl_language],
                outputs=[cl_audio_stream, cl_audio_final, cl_timing, cl_metrics]
            )
        
        # Tab 4: Instruct2
        with gr.Tab("📝 Instruct2 - Style Control"):
            with gr.Row():
                with gr.Column():
                    inst_text = gr.Textbox(label="Text to synthesize", placeholder="Enter text here...", lines=3)
                    inst_instruct = gr.Textbox(
                        label="Style instruction (Optional - Auto ASR if empty)", 
                        placeholder="E.g., 'speak cheerfully' or leave empty for auto-generation...", 
                        lines=2,
                        info="Provide style instruction OR leave empty to use ASR transcript"
                    )
                    inst_prompt_audio = gr.Audio(label="Prompt audio", type="filepath")
                    
                    with gr.Row():
                        inst_stream = gr.Radio(choices=["Streaming", "Non-streaming"], value="Non-streaming", label="Mode")
                        inst_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")
                    
                    inst_frontend = gr.Checkbox(value=True, label="Enable text frontend")
                    
                    with gr.Accordion("📊 Metrics Settings (Optional)", open=False):
                        inst_enable_metrics = gr.Checkbox(value=False, label="Enable Metrics (CER + SIM)")
                        inst_ref_text = gr.Textbox(label="Reference Text (for CER)", placeholder="Ground truth text...")
                        inst_language = gr.Dropdown(
                            choices=["vi", "en", "zh", "ja", "ko"], 
                            value="vi", 
                            label="Language (for ASR & Metrics)",
                            info="Used for auto-generating instruction and CER calculation"
                        )
                    
                    inst_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    inst_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
                    inst_audio_final = gr.Audio(label="Final Output")
                    inst_timing = gr.Markdown(label="Timing")
                    inst_metrics = gr.Markdown(label="Metrics")
            
            inst_btn.click(
                fn=generate_instruct2,
                inputs=[inst_text, inst_instruct, inst_prompt_audio, inst_stream, inst_speed, inst_frontend, inst_enable_metrics, inst_ref_text, inst_language],
                outputs=[inst_audio_stream, inst_audio_final, inst_timing, inst_metrics]
            )
        
        # Tab 5: Voice Conversion
        with gr.Tab("🔄 Voice Conversion"):
            with gr.Row():
                with gr.Column():
                    vc_source = gr.Audio(label="Source audio", type="filepath")
                    vc_prompt = gr.Audio(label="Target audio", type="filepath")
                    
                    with gr.Row():
                        vc_stream = gr.Radio(choices=["Streaming", "Non-streaming"], value="Non-streaming", label="Mode")
                        vc_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")
                    
                    with gr.Accordion("📊 Metrics Settings (Optional)", open=False):
                        vc_enable_metrics = gr.Checkbox(value=False, label="Enable Metrics (CER + SIM)")
                        vc_ref_text = gr.Textbox(label="Reference Text (for CER)", placeholder="Ground truth text...")
                        vc_language = gr.Dropdown(choices=["vi", "en", "zh", "ja", "ko"], value="vi", label="Language")
                    
                    vc_btn = gr.Button("Convert", variant="primary")
                
                with gr.Column():
                    vc_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
                    vc_audio_final = gr.Audio(label="Final Output")
                    vc_timing = gr.Markdown(label="Timing")
                    vc_metrics = gr.Markdown(label="Metrics")
            
            vc_btn.click(
                fn=generate_vc,
                inputs=[vc_source, vc_prompt, vc_stream, vc_speed, vc_enable_metrics, vc_ref_text, vc_language],
                outputs=[vc_audio_stream, vc_audio_final, vc_timing, vc_metrics]
            )
    
    gr.Markdown("""
    ---
    ### 📖 User Guide:
    
    **Features:**
    - ⏱️ **Timing**: Always tracked (Time to First Chunk + Total Time)
    - 📊 **Metrics**: Optional (Enable in accordion)
      - **CER**: Character Error Rate (lower is better, requires reference text)
      - **SIM**: Speaker Similarity (higher is better, 0-100%)
    - 🎤 **Auto ASR**: Zero-shot & Instruct2 modes auto-generate text from audio if not provided
    
    **Modes:**
    - **SFT**: Pre-defined speakers
    - **Zero-shot**: Clone voice with audio (+ optional transcript)
      - ✨ NEW: Leave prompt text empty to auto-generate using ASR
    - **Cross-lingual**: Voice cloning across languages
    - **Instruct2**: Style-controlled synthesis
      - ✨ NEW: Leave instruction empty to use ASR transcript as instruction
    - **Voice Conversion**: Convert voice A to voice B
    
    **Tips:**
    - Use 3-10s clear audio for best cloning results
    - Auto ASR works best with clear speech and correct language selection
    - Enable metrics only when needed (adds processing time)
    - Streaming mode shows faster feedback but final quality is same
    
    **Auto ASR Feature:**
    - Zero-shot: Automatically transcribes prompt audio if prompt text is empty
    - Instruct2: Uses ASR transcript as instruction if instruction text is empty
    - Requires correct language selection in dropdown
    - Works with: Vietnamese (vi), English (en), Chinese (zh), Japanese (ja), Korean (ko)
    """)

if __name__ == "__main__":
    demo.queue()
    demo.launch()