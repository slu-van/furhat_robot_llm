import re
import numpy as np
import sounddevice as sd
import mlx_whisper
from furhat_remote_api import FurhatRemoteAPI
from openai import OpenAI

ROBOT_IP = "192.168.1.10"
LLM_URL = "http://127.0.0.1:8080/v1"
MODEL = "lmstudio-community/Qwen3.8-27B-MLX-6bit"
WHISPER = "mlx-community/whisper-large-v3-turbo"

SAMPLE_RATE = 16000
FRAME_MS = 30
START_TIMEOUT = 5.0
MAX_LISTEN = 8.0
SILENCE_TO_STOP = 0.8
CALIBRATE_MS = 500
WARMUP_MS = 200

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

furhat = FurhatRemoteAPI(ROBOT_IP)
llm = OpenAI(base_url=LLM_URL, api_key="local")
history = [{
    "role": "system",
    "content": "You are Furhat. Reply in one or two short spoken sentences. Do not think aloud."
}]

def rms(frame: np.ndarray) -> float:
    x = np.asarray(frame, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(np.square(x))))

def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else as_text(part)
            for part in value
        )
    return str(value)

def extract_reply(resp) -> str:
    choice = resp.choices[0]
    msg = choice.message
    dump = {}
    if hasattr(msg, "model_dump"):
        dump = msg.model_dump()
        print("RAW DUMP:", dump)
    else:
        print("RAW:", msg)

    content = as_text(dump.get("content") or getattr(msg, "content", None))
    reasoning = as_text(
        dump.get("reasoning_content")
        or dump.get("reasoning")
        or getattr(msg, "reasoning_content", None)
        or getattr(msg, "reasoning", None)
    )
    text = THINK_RE.sub("", content).strip()
    if not text:
        text = THINK_RE.sub("", reasoning).strip()
    return text

def hear() -> str:
    print("Listening on Mac mic...")
    frame_n = int(SAMPLE_RATE * FRAME_MS / 1000)
    chunks = []
    heard_speech = False
    silence_ms = 0
    total_ms = 0
    peak = 0.0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=frame_n,
    ) as stream:
        for _ in range(max(1, int(WARMUP_MS / FRAME_MS))):
            stream.read(frame_n)

        noise_vals = []
        for _ in range(max(1, int(CALIBRATE_MS / FRAME_MS))):
            frame, _ = stream.read(frame_n)
            noise_vals.append(rms(frame))
        noise_floor = float(np.median(noise_vals))
        speech_rms = max(noise_floor * 8.0, 0.01)
        print(f"noise={noise_floor:.4f}  speech_threshold={speech_rms:.4f}")
        print("Speak now.")

        while True:
            frame, _ = stream.read(frame_n)
            frame = np.squeeze(np.asarray(frame, dtype=np.float32))
            chunks.append(frame)
            total_ms += FRAME_MS
            level = rms(frame)
            peak = max(peak, level)

            if level >= speech_rms:
                heard_speech = True
                silence_ms = 0
                mark = "SPEECH"
            else:
                silence_ms += FRAME_MS
                mark = "quiet"

            if total_ms % 300 == 0:
                print(
                    f"{mark} rms={level:.4f} peak={peak:.4f} "
                    f"silence={silence_ms}ms heard={heard_speech}"
                )

            if not heard_speech and total_ms >= START_TIMEOUT * 1000:
                print(f"stop: no speech (peak={peak:.4f})")
                return ""
            if heard_speech and silence_ms >= SILENCE_TO_STOP * 1000:
                print("stop: pause")
                break
            if total_ms >= MAX_LISTEN * 1000:
                print("stop: max listen")
                break

    wav = np.concatenate(chunks).astype(np.float32)
    print("Sending to Whisper...")
    out = mlx_whisper.transcribe(wav, path_or_hf_repo=WHISPER, language="en")
    return (out.get("text") or "").strip()

furhat.say(text="Hello. I am ready to talk.", blocking=True)

while True:
    text = hear()
    print("TEXT:", repr(text))
    if not text:
        continue

    history.append({"role": "user", "content": text})
    resp = llm.chat.completions.create(
        model=MODEL,
        messages=history,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.8,
        extra_body={
            "think": False,
            "enableThinking": False,
            "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    reply = extract_reply(resp)
    if not reply:
        print("EMPTY MODEL REPLY")
        furhat.say(text="Please say that again.", blocking=True)
        history.pop()
        continue

    history.append({"role": "assistant", "content": reply})
    if len(history) > 17:
        history = history[:1] + history[-16:]
    print("ROBOT:", reply)
    furhat.say(text=reply, blocking=True)
