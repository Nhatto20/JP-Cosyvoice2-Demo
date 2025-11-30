
# import sys
# import torch
# sys.path.append('third_party/Matcha-TTS')

# import streamlit as st
# import torchaudio
# import tempfile
# import time
# import base64
# import numpy as np

# from cosyvoice.cli.cosyvoice import CosyVoice2
# from cosyvoice.utils.file_utils import load_wav

# # -------------------------
# # Streamlit UI
# # -------------------------
# st.set_page_config(page_title="CosyVoice2 Streaming Demo", layout="wide")

# st.title("🎤 CosyVoice2 Real-time Streaming Demo")
# st.markdown("### Test zero-shot + streaming inference with real-time playback")

# # -------------------------
# # Sidebar settings
# # -------------------------
# st.sidebar.header("⚙️ Settings")

# default_model_path = r"C:\Users\japan\datasets\pretrained_models\JP_CosyVoice2_finetune"

# model_path = st.sidebar.text_input(
#     "Model directory",
#     value=default_model_path,
#     help="Enter path to CosyVoice2 pretrained model"
# )

# uploaded_ref = st.sidebar.file_uploader(
#     "Reference Audio (16kHz WAV)",
#     type=["wav"],
#     help="Optional: Reference speaker audio for zero-shot"
# )

# ref_text = st.sidebar.text_area(
#     "Reference Text (optional)",
#     "",
#     help="If empty, model will use only the audio"
# )

# # -------------------------
# # Input text
# # -------------------------
# text = st.text_area(
#     "📌 Input Text for TTS",
#     "今日はとても良い天気ですね。\nゆっくり話しますので、最後まで聞いてください。",
#     height=160
# )

# btn_run = st.button("🚀 Start Streaming")

# # -------------------------
# # Load model (lazy)
# # -------------------------
# @st.cache_resource(show_spinner=True)
# def load_model(path):
#     return CosyVoice2(
#         path,
#         load_jit=False,
#         load_trt=False,
#         load_vllm=False,
#         fp16=False
#     )

# if btn_run:

#     if text.strip() == "":
#         st.error("❗ Please enter some text.")
#         st.stop()

#     cosyvoice = load_model(model_path)
#     st.success("✅ Model loaded!")

#     # ---------------------------------------
#     # Prepare reference audio
#     # ---------------------------------------
#     if uploaded_ref is not None:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#             f.write(uploaded_ref.read())
#             ref_path = f.name
#         prompt_speech_16k = load_wav(ref_path, 16000)
#     else:
#         prompt_speech_16k = None
#         st.warning("⚠️ No reference audio provided. Using model default voice.")

#     st.write("---")
#     st.subheader("🔊 Real-time Streaming Output")

#     audio_placeholder = st.empty()
#     log_placeholder = st.empty()

#     # ---------------------------------------
#     # Streaming inference
#     # ---------------------------------------
#     start_t = time.time()
#     first_packet_t = None

#     log_placeholder.write("⏳ Streaming started...")

#     # Buffer to accumulate audio for continuous playback
#     audio_buffer = np.array([], dtype=np.float32)

#     # Run streaming TTS
#     for idx, out in enumerate(
#         cosyvoice.inference_zero_shot(
#             text,
#             ref_text or "",
#             prompt_speech_16k,
#             stream=True
#         )
#     ):
#         t_now = time.time()

#         if first_packet_t is None:
#             first_packet_t = t_now
#             log_placeholder.write(
#                 f"🚀 **First packet latency:** {first_packet_t - start_t:.3f} sec"
#             )

#         chunk = out["tts_speech"].numpy().flatten()
#         audio_buffer = np.concatenate([audio_buffer, chunk])

#         # Convert to WAV binary buffer for playback
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#             torchaudio.save(f.name, out["tts_speech"], cosyvoice.sample_rate)
#             audio_bytes = open(f.name, "rb").read()

#         # 🔥 Real-time audio playback
#         audio_placeholder.audio(audio_bytes, format="audio/wav")

#     end_t = time.time()
#     log_placeholder.write(
#         f"✔️ **Total time:** {end_t - start_t:.3f} sec\n\n🎉 **Streaming completed!**"
#     )

#     # Save final combined audio
#     st.write("### 💾 Download Full Audio Output")
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#         torchaudio.save(f.name,
#                         torch.tensor(audio_buffer).unsqueeze(0),
#                         cosyvoice.sample_rate)
#         st.download_button(
#             "Download Full WAV",
#             data=open(f.name, "rb").read(),
#             file_name="cosyvoice_stream_output.wav",
#             mime="audio/wav"
#         )



import gradio as gr
import torch
import torchaudio
import numpy as np
from cosyvoice.cli.cosyvoice import CosyVoice2
import tempfile
import os
from pathlib import Path
import sys
sys.path.append('third_party/Matcha-TTS')


# Initialize CosyVoice2 model
#MODEL_DIR = r"C:\Users\japan\datasets\pretrained_models\JP_CosyVoice2_finetune"  # or your model path
MODEL_DIR = "/mnt/c/Users/japan/datasets/pretrained_models/JP_CosyVoice2_finetune"
cosyvoice = CosyVoice2(MODEL_DIR)

# Get available speakers
available_spks = cosyvoice.list_available_spks()

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

def generate_sft(text, spk_id, stream_mode, speed, text_frontend):
    """SFT inference with pre-defined speaker"""
    try:
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
            
            # Yield intermediate results for streaming
            if stream and i < 10:  # Limit streaming chunks displayed
                temp_audio = np.concatenate(output_audios)
                yield (cosyvoice.sample_rate, temp_audio), None
        
        # Final output
        final_audio = np.concatenate(output_audios)
        final_path = save_audio(torch.from_numpy(final_audio).unsqueeze(0), cosyvoice.sample_rate)
        yield (cosyvoice.sample_rate, final_audio), final_path
        
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")

def generate_zero_shot(text, prompt_text, prompt_audio, stream_mode, speed, text_frontend):
    """Zero-shot inference with custom voice"""
    try:
        if prompt_audio is None:
            raise gr.Error("Please upload a prompt audio file")
        
        # Load prompt audio
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
        
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")

def generate_cross_lingual(text, prompt_audio, stream_mode, speed, text_frontend):
    """Cross-lingual inference"""
    try:
        if prompt_audio is None:
            raise gr.Error("Please upload a prompt audio file")
        
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
        
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")

def generate_instruct2(text, instruct_text, prompt_audio, stream_mode, speed, text_frontend):
    """Instruction-guided inference"""
    try:
        if prompt_audio is None:
            raise gr.Error("Please upload a prompt audio file")
        
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
        
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")

def generate_vc(source_audio, prompt_audio, stream_mode, speed):
    """Voice conversion"""
    try:
        if source_audio is None or prompt_audio is None:
            raise gr.Error("Please upload both source and prompt audio files")
        
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
        
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")

# Build Gradio Interface
with gr.Blocks(title="CosyVoice2 TTS") as demo:
    gr.Markdown("""
    # 🎙️ CosyVoice2 Text-to-Speech System
    
    Multi-functional TTS system with support for SFT, Zero-shot, Cross-lingual, Instruct-guided, and Voice Conversion.
    """)
    
    with gr.Tabs():
        # Tab 1: SFT (Pre-defined Speaker)
        with gr.Tab("🎤 SFT - Pre-defined Speaker"):
            gr.Markdown("Use pre-trained speaker voices")
            with gr.Row():
                with gr.Column():
                    sft_text = gr.Textbox(
                        label="Text to synthesize",
                        placeholder="Enter text here...",
                        lines=3
                    )
                    sft_spk = gr.Dropdown(
                        choices=available_spks,
                        label="Speaker ID",
                        value=available_spks[0] if available_spks else None
                    )
                    with gr.Row():
                        sft_stream = gr.Radio(
                            choices=["Streaming", "Non-streaming"],
                            value="Non-streaming",
                            label="Mode"
                        )
                        sft_speed = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Speed"
                        )
                    sft_frontend = gr.Checkbox(
                        value=True,
                        label="Enable text frontend processing"
                    )
                    sft_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    sft_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
                    sft_audio_final = gr.Audio(label="Final Output")
            
            sft_btn.click(
                fn=generate_sft,
                inputs=[sft_text, sft_spk, sft_stream, sft_speed, sft_frontend],
                outputs=[sft_audio_stream, sft_audio_final]
            )
        
        # Tab 2: Zero-shot
        with gr.Tab("🎯 Zero-shot - Custom Voice"):
            gr.Markdown("Clone any voice with a short audio sample + text")
            with gr.Row():
                with gr.Column():
                    zs_text = gr.Textbox(
                        label="Text to synthesize",
                        placeholder="Enter text here...",
                        lines=3
                    )
                    zs_prompt_text = gr.Textbox(
                        label="Prompt text (transcript of prompt audio)",
                        placeholder="Enter the transcript of your audio sample...",
                        lines=2
                    )
                    zs_prompt_audio = gr.Audio(
                        label="Prompt audio (3-10 seconds recommended)",
                        type="filepath"
                    )
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
                        label="Enable text frontend processing"
                    )
                    zs_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    zs_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
                    zs_audio_final = gr.Audio(label="Final Output")
            
            zs_btn.click(
                fn=generate_zero_shot,
                inputs=[zs_text, zs_prompt_text, zs_prompt_audio, zs_stream, zs_speed, zs_frontend],
                outputs=[zs_audio_stream, zs_audio_final]
            )
        
        # Tab 3: Cross-lingual
        with gr.Tab("🌍 Cross-lingual"):
            gr.Markdown("Synthesize text in different language using voice characteristics")
            with gr.Row():
                with gr.Column():
                    cl_text = gr.Textbox(
                        label="Text to synthesize (any language)",
                        placeholder="Enter text in target language...",
                        lines=3
                    )
                    cl_prompt_audio = gr.Audio(
                        label="Prompt audio (voice sample)",
                        type="filepath"
                    )
                    with gr.Row():
                        cl_stream = gr.Radio(
                            choices=["Streaming", "Non-streaming"],
                            value="Non-streaming",
                            label="Mode"
                        )
                        cl_speed = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Speed"
                        )
                    cl_frontend = gr.Checkbox(
                        value=True,
                        label="Enable text frontend processing"
                    )
                    cl_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    cl_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
                    cl_audio_final = gr.Audio(label="Final Output")
            
            cl_btn.click(
                fn=generate_cross_lingual,
                inputs=[cl_text, cl_prompt_audio, cl_stream, cl_speed, cl_frontend],
                outputs=[cl_audio_stream, cl_audio_final]
            )
        
        # Tab 4: Instruct-guided
        with gr.Tab("📝 Instruct2 - Style Control"):
            gr.Markdown("Control speaking style with instructions (e.g., 'speak slowly and sadly')")
            with gr.Row():
                with gr.Column():
                    inst_text = gr.Textbox(
                        label="Text to synthesize",
                        placeholder="Enter text here...",
                        lines=3
                    )
                    inst_instruct = gr.Textbox(
                        label="Style instruction",
                        placeholder="E.g., 'speak in a cheerful and energetic way'",
                        lines=2
                    )
                    inst_prompt_audio = gr.Audio(
                        label="Prompt audio (voice sample)",
                        type="filepath"
                    )
                    with gr.Row():
                        inst_stream = gr.Radio(
                            choices=["Streaming", "Non-streaming"],
                            value="Non-streaming",
                            label="Mode"
                        )
                        inst_speed = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Speed"
                        )
                    inst_frontend = gr.Checkbox(
                        value=True,
                        label="Enable text frontend processing"
                    )
                    inst_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    inst_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
                    inst_audio_final = gr.Audio(label="Final Output")
            
            inst_btn.click(
                fn=generate_instruct2,
                inputs=[inst_text, inst_instruct, inst_prompt_audio, inst_stream, inst_speed, inst_frontend],
                outputs=[inst_audio_stream, inst_audio_final]
            )
        
        # Tab 5: Voice Conversion
        with gr.Tab("🔄 Voice Conversion"):
            gr.Markdown("Convert source voice to target voice")
            with gr.Row():
                with gr.Column():
                    vc_source = gr.Audio(
                        label="Source audio (voice to convert)",
                        type="filepath"
                    )
                    vc_prompt = gr.Audio(
                        label="Target audio (target voice)",
                        type="filepath"
                    )
                    with gr.Row():
                        vc_stream = gr.Radio(
                            choices=["Streaming", "Non-streaming"],
                            value="Non-streaming",
                            label="Mode"
                        )
                        vc_speed = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Speed"
                        )
                    vc_btn = gr.Button("Convert", variant="primary")
                
                with gr.Column():
                    vc_audio_stream = gr.Audio(label="Streaming Preview", autoplay=True)
                    vc_audio_final = gr.Audio(label="Final Output")
            
            vc_btn.click(
                fn=generate_vc,
                inputs=[vc_source, vc_prompt, vc_stream, vc_speed],
                outputs=[vc_audio_stream, vc_audio_final]
            )
    
    gr.Markdown("""
    ---
    ### 📖 Instructions:
    
    - **SFT**: Select a pre-trained speaker and enter text
    - **Zero-shot**: Upload a 3-10s audio sample with its transcript to clone the voice
    - **Cross-lingual**: Clone voice characteristics across different languages
    - **Instruct2**: Add style instructions (e.g., "happy", "slow", "whisper")
    - **Voice Conversion**: Convert one voice to sound like another
    
    **Streaming vs Non-streaming:**
    - Streaming: Get audio chunks in real-time (faster feedback)
    - Non-streaming: Process complete audio at once (better quality)
    
    **Tips:**
    - Prompt audio should be clear with minimal background noise
    - Keep prompt audio between 3-10 seconds for best results
    - Speed range: 0.5x (slower) to 2.0x (faster)
    """)

if __name__ == "__main__":
    demo.queue()  # Enable queue for streaming
    demo.launch()