# voltman_full.py
"""
VoltMan — Full-featured Circuit AI
- Multi-model selector: Gemini (default), Groq, HuggingFace, OpenRouter, Local Llama
- Strict JSON forcing for Gemini (response_mime_type)
- Dynamic theme engine
- Graphviz diagram generation
- Iron-Man greeting (TTS)
- Voice STT support (whisper.cpp fallback or speech_recognition)
- Save / download circuits (JSON, ZIP)

Notes:
- You still must provide API keys for cloud providers via Streamlit secrets or the input box.
- Local Llama/whisper features are optional and require separate local installs.

Deploy-ready: run with `streamlit run voltman_full.py`.
"""

import os
import io
import json
import base64
import zipfile
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

import streamlit as st
import pandas as pd
import requests
import graphviz

# Optional SDKs
try:
    import google.generativeai as genai
except Exception:
    genai = None

from gtts import gTTS

# Speech recognition fallback
try:
    import speech_recognition as sr
except Exception:
    sr = None

# -------------------- Theme Engine --------------------

def detect_category(text: Optional[str]) -> str:
    if not text:
        return "general"
    t = text.lower()
    power_keywords = ["rectifier", "regulator", "supply", "buck", "boost", "inverter", "charger", "psu"]
    amp_keywords = ["amplifier", "op-amp", "opamp", "audio", "gain", "bjt"]
    digital_keywords = ["and gate", "or gate", "microcontroller", "arduino", "mcu", "digital", "logic"]
    sensor_keywords = ["sensor", "adc", "thermistor", "ldr", "temperature"]
    if any(k in t for k in power_keywords):
        return "power"
    if any(k in t for k in amp_keywords):
        return "amplifier"
    if any(k in t for k in digital_keywords):
        return "digital"
    if any(k in t for k in sensor_keywords):
        return "sensor"
    return "general"


def get_theme_for_category(category: str) -> dict:
    themes = {
        "power": {"name": "Power Electronics Mode", "accent1": "#ff9800", "chip": "⚡ Power rail sequence complete."},
        "amplifier": {"name": "Analog Mode", "accent1": "#e040fb", "chip": "🎚️ Signal chain stabilized."},
        "digital": {"name": "Digital Mode", "accent1": "#00e5ff", "chip": "🧮 Logic matrix active."},
        "sensor": {"name": "Sensors Mode", "accent1": "#00e676", "chip": "📡 Sensing online."},
        "general": {"name": "General Mode", "accent1": "#00e5ff", "chip": "🔌 Lab bench ready."},
    }
    return themes.get(category, themes["general"])


# -------------------- Graphviz Builder --------------------

def build_graph(circuit: dict) -> str:
    components = circuit.get("components", []) or []
    connections = circuit.get("connections", []) or []
    lines = ["graph circuit {", "  rankdir=LR;", '  node [fontname="Helvetica"];']
    for c in components:
        cid = c.get("id") or c.get("name") or "?"
        t = c.get("type", "")
        v = c.get("value", "")
        label = f"{cid}\\n{t}\\n{v}" if (t or v) else cid
        lines.append(f'  "{cid}" [shape=box,label="{label}"];')
    nets = set()
    for conn in connections:
        if not conn:
            continue
        net = str(conn[0])
        if net not in nets:
            lines.append(f'  "{net}" [shape=ellipse,style=dashed,label="{net}"];')
            nets.add(net)
        for pin in conn[1:]:
            comp_id = str(pin).split(".")[0]
            lines.append(f'  "{net}" -- "{comp_id}";')
    lines.append("}")
    return "\n".join(lines)


# -------------------- Multi-Model Wrappers --------------------

# Gemini wrapper (requires google-generativeai)

def call_gemini(api_key: str, prompt: str, image_bytes: Optional[bytes] = None) -> str:
    if genai is None:
        raise RuntimeError("google.generativeai not installed")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config={"response_mime_type": "application/json"}
    )
    parts = [prompt]
    if image_bytes:
        parts.append({"mime_type": "image/png", "data": image_bytes})
    response = model.generate_content(parts)
    return response.text


# HuggingFace inference wrapper (uses inference API)

def call_hf(api_key: str, model_id: str, prompt: str) -> str:
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}" }
    payload = {"inputs": prompt}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    out = r.json()
    # Many HF models return text or array; handle simple cases
    if isinstance(out, dict) and "error" in out:
        raise RuntimeError(out["error"])
    if isinstance(out, list):
        # often returns [{'generated_text': '...'}]
        first = out[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
        return str(out)
    if isinstance(out, str):
        return out
    return json.dumps(out)


# OpenRouter / OpenAI-compatible wrapper

def call_openrouter(api_key: str, prompt: str, model: str = "gpt-oss-20b-free") -> str:
    url = "https://openrouter.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role":"user","content": prompt}],
        "max_tokens": 1200
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    # openrouter returns choices[].message.content
    return j.get("choices", [])[0].get("message", {}).get("content", "")


# Groq API wrapper (example) – adjust endpoint if needed

def call_groq(api_key: str, prompt: str, model: str = "llama-3.1-70b") -> str:
    url = "https://api.groq.com/v1/engines/" + model + "/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "max_tokens": 1024}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    return j.get("text", "")


# Local llama fallback (stub) — requires llama.cpp or similar installed

def call_local_llama(prompt: str) -> str:
    # This function tries common commands for llama.cpp / llamacpp
    # You must adjust the binary path and model path to your setup.
    binary_paths = ["./main", "/usr/local/bin/llama", "llama"]
    model_path = os.environ.get("LLAMA_MODEL_PATH", "./models/ggml-model.bin")
    for binp in binary_paths:
        if Path(binp).exists():
            cmd = [binp, "-m", model_path, "-p", prompt, "--tokens", "1024"]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=60)
                return out.decode("utf-8")
            except Exception as e:
                continue
    raise RuntimeError("Local Llama binary not found. Install llama.cpp or set LLAMA_MODEL_PATH and ensure binary is available.")


# Top-level function that chooses model and returns parsed JSON

def call_model(choice: str, keys: dict, prompt: str, image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """choice: one of 'gemini','openrouter','huggingface','groq','local'"""
    raw = ""
    try:
        if choice == "gemini":
            raw = call_gemini(keys.get("GEMINI_API_KEY", ""), prompt, image_bytes=image_bytes)
        elif choice == "openrouter":
            raw = call_openrouter(keys.get("OPENROUTER_API_KEY", ""), prompt)
        elif choice == "huggingface":
            raw = call_hf(keys.get("HUGGINGFACE_API_KEY", ""), keys.get("HF_MODEL", "phi-3-mini"), prompt)
        elif choice == "groq":
            raw = call_groq(keys.get("GROQ_API_KEY", ""), prompt)
        elif choice == "local":
            raw = call_local_llama(prompt)
        else:
            raise RuntimeError("Unknown model choice")
    except Exception as e:
        return {
            "title": "Error",
            "explanation": f"Model call failed: {e}",
            "components": [],
            "connections": [],
            "notes": ""
        }

    # Try parse JSON strictly
    try:
        data = json.loads(raw)
    except Exception:
        # Attempt to extract JSON substring (best-effort)
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                candidate = raw[start:end+1]
                data = json.loads(candidate)
            except Exception:
                data = {
                    "title": "Error",
                    "explanation": "Model did not return valid JSON.",
                    "components": [],
                    "connections": [],
                    "notes": raw
                }
        else:
            data = {
                "title": "Error",
                "explanation": "Model did not return valid JSON.",
                "components": [],
                "connections": [],
                "notes": raw
            }
    return data


# -------------------- TTS / STT Utilities --------------------

def text_to_speech_bytes(text: str) -> bytes:
    tts = gTTS(text=text, lang="en")
    temp = Path(tempfile.gettempdir()) / "voltman_tts.mp3"
    tts.save(temp)
    return temp.read_bytes()


def transcribe_with_speechrec(uploaded_file) -> str:
    if sr is None:
        return "SpeechRecognition not installed."
    recognizer = sr.Recognizer()
    tmp = Path(tempfile.gettempdir()) / uploaded_file.name
    with open(tmp, "wb") as f:
        f.write(uploaded_file.read())
    with sr.AudioFile(str(tmp)) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except Exception as e:
        return f"Transcription failed: {e}"


def transcribe_with_whispercpp(file_path: str) -> str:
    # This calls an installed whisper.cpp binary (if available). Adjust as needed.
    bins = ["./main", "./whisper.cpp/main", "/usr/local/bin/whisper.cpp"]
    for b in bins:
        if Path(b).exists():
            try:
                out = subprocess.check_output([b, file_path], stderr=subprocess.STDOUT, timeout=120)
                return out.decode("utf-8")
            except Exception as e:
                continue
    return "whisper.cpp binary not found"


# -------------------- Streamlit UI --------------------

def show_output(circuit: dict, theme: dict, tts_key: str):
    st.subheader(f"📘 {circuit.get('title', 'Circuit Result')}")
    st.caption(f"Mode: {theme['name']} — {theme.get('chip','')}")

    st.markdown("### 📖 Detailed Walkthrough")
    st.write(circuit.get("explanation", "No explanation provided."))

    st.markdown("### 🧩 Component-Level View")
    comps = circuit.get("components", []) or []
    if comps:
        df = pd.DataFrame(comps)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No components listed.")

    st.markdown("### 🔗 Connectivity / Netlist View")
    conns = circuit.get("connections", []) or []
    if conns:
        for c in conns:
            st.code("  - " + "  ---  ".join(str(x) for x in c), language="text")
    else:
        st.info("No connections listed.")

    st.markdown("### 🗺️ Conceptual Connectivity Diagram")
    try:
        dot = build_graph(circuit)
        st.graphviz_chart(dot)
    except Exception as e:
        st.error(f"Could not render graph: {e}")

    st.markdown("### 📝 Design Notes & Tips")
    st.write(circuit.get("notes", "None"))

    with st.expander("🔊 Hear the Explanation (Suit AI Voice)"):
        if st.button("Play Voice Explanation", key=tts_key):
            audio_bytes = text_to_speech_bytes(circuit.get("explanation", ""))
            st.audio(audio_bytes, format="audio/mp3")

    with st.expander("📦 Raw JSON Output"):
        st.code(json.dumps(circuit, indent=2), language="json")

    # Save / Download
    st.markdown("---")
    if st.button("Download JSON"):
        b = json.dumps(circuit, indent=2).encode("utf-8")
        st.download_button("Click to download JSON", data=b, file_name="circuit.json", mime="application/json")

    if st.button("Download as ZIP"):
        tmpzip = Path(tempfile.gettempdir()) / "circuit.zip"
        with zipfile.ZipFile(tmpzip, "w") as z:
            z.writestr("circuit.json", json.dumps(circuit, indent=2))
        with open(tmpzip, "rb") as f:
            st.download_button("Download ZIP", data=f, file_name="circuit_bundle.zip", mime="application/zip")


def main():
    st.set_page_config(page_title="VoltMan — Circuit AI", layout="wide")

    st.title("⚡ VoltMan — Multimodal Circuit AI (Multi-Model)")

    # LEFT PANEL: API keys & settings
    st.sidebar.header("API & Settings")

    model_choice = st.sidebar.selectbox("Model / Provider", ["gemini", "openrouter", "huggingface", "groq", "local"], index=0)

    st.sidebar.markdown("**API Keys (store in session for demo)**")
    keys = {}
    keys["GEMINI_API_KEY"] = st.sidebar.text_input("Gemini API Key", type="password")
    keys["OPENROUTER_API_KEY"] = st.sidebar.text_input("OpenRouter API Key", type="password")
    keys["HUGGINGFACE_API_KEY"] = st.sidebar.text_input("HuggingFace API Key", type="password")
    keys["GROQ_API_KEY"] = st.sidebar.text_input("Groq API Key", type="password")
    keys["HF_MODEL"] = st.sidebar.text_input("HF Model ID", value="phi-3-mini-4k-instruct")

    st.sidebar.markdown("---")
    use_iron_greeting = st.sidebar.checkbox("Enable Iron‑Man Greeting (TTS)", value=True)
    use_whispercpp = st.sidebar.checkbox("Use whisper.cpp for STT (if available)", value=False)
    enable_local_llama = st.sidebar.checkbox("Enable local Llama fallback", value=False)

    # MAIN UI
    tabs = st.tabs(["✍️ Text", "🖼️ Image", "🎤 Voice"])

    # Keep circuits in state
    if "last_circuit" not in st.session_state:
        st.session_state["last_circuit"] = None

    # Iron-Man Greeting once
    if use_iron_greeting and st.session_state.get("greeted") is not True:
        if st.button("Play Suit Greeting"):
            txt = "System online. Power levels stable. Circuit assistant ready."
            st.audio(text_to_speech_bytes(txt), format="audio/mp3")
            st.session_state["greeted"] = True

    # TEXT TAB
    with tabs[0]:
        st.subheader("Text-based Circuit Design & Explanation")
        text_prompt = st.text_area("Describe what you want to build or understand:", height=160,
                                   placeholder="Example: Design a 5V regulated DC supply from 230V AC...")
        generate_text = st.button("Generate from Text")

        if generate_text and text_prompt.strip():
            with st.spinner("Calling model..."):
                payload_prompt = "\n".join([
                    "You are an expert electronics circuit design assistant. Reply ONLY in JSON as specified.",
                    text_prompt
                ])
                raw = call_model(model_choice, keys, payload_prompt)
                st.session_state["last_circuit"] = raw
                cat = detect_category((raw.get("title",""))+" "+(raw.get("explanation","")))
                st.session_state["theme_category"] = cat

        if st.session_state["last_circuit"]:
            theme = get_theme_for_category(st.session_state.get("theme_category","general"))
            show_output(st.session_state["last_circuit"], theme, tts_key="tts_text")
        else:
            st.info("Enter a prompt and click Generate from Text.")

    # IMAGE TAB
    with tabs[1]:
        st.subheader("Image-based Circuit Understanding")
        img_file = st.file_uploader("Upload a circuit image (PNG/JPG)", type=["png","jpg","jpeg"])
        img_prompt = st.text_area("Optional: what to do with the image:", value="Explain this circuit in detail and output JSON netlist.", height=120)
        analyze_image = st.button("Analyze Image")
        if analyze_image and img_file:
            bytes_img = img_file.read()
            with st.spinner("Calling model with image..."):
                raw = call_model(model_choice, keys, img_prompt, image_bytes=bytes_img)
                st.session_state["last_circuit"] = raw
                cat = detect_category((raw.get("title",""))+" "+(raw.get("explanation","")))
                st.session_state["theme_category"] = cat
        if st.session_state["last_circuit"]:
            theme = get_theme_for_category(st.session_state.get("theme_category","general"))
            show_output(st.session_state["last_circuit"], theme, tts_key="tts_img")

    # VOICE TAB
    with tabs[2]:
        st.subheader("Voice-based Circuit Request")
        audio_file = st.file_uploader("Upload audio file (wav/mp3/m4a)", type=["wav","mp3","m4a"])
        generate_voice = st.button("Generate from Voice")
        if generate_voice and audio_file:
            st.audio(audio_file)
            # save temp
            tmp = Path(tempfile.gettempdir()) / audio_file.name
            with open(tmp, "wb") as f:
                f.write(audio_file.read())
            if use_whispercpp:
                with st.spinner("Transcribing with whisper.cpp..."):
                    transcript = transcribe_with_whispercpp(str(tmp))
            else:
                with st.spinner("Transcribing..."):
                    transcript = transcribe_with_speechrec(audio_file)
            st.write("**Transcription:**")
            st.write(transcript)
            with st.spinner("Generating..."):
                raw = call_model(model_choice, keys, transcript)
                st.session_state["last_circuit"] = raw
                cat = detect_category((raw.get("title",""))+" "+(raw.get("explanation","")))
                st.session_state["theme_category"] = cat
        if st.session_state["last_circuit"]:
            theme = get_theme_for_category(st.session_state.get("theme_category","general"))
            show_output(st.session_state["last_circuit"], theme, tts_key="tts_voice")

    # Footer instructions
    st.markdown("---")
    st.caption("Tips: store keys in Streamlit Secrets for production. Local fallback requires local model + binaries installed.")


if __name__ == '__main__':
    main()
