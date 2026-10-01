# furhat_robot_llm
Connecting a furhat robot to a local llm

# Furhat voice loop

Local voice chat with a Furhat robot. A Mac listens on its microphone, detects a pause, transcribes with Whisper, and sends the text to a local Qwen model. Furhat only speaks.

## Layout

Mac mic (sounddevice) pause detection mlx-whisper LM Studio Qwen FurhatRemoteAPI.say --> Furhat robot

Whisper and Qwen stay on the Mac. Furhat is not asked to recognize speech.

## Requirements

- Furhat on the same network as the Mac
- Furhat Remote API, default robot IP used here: `192.168.1.10`
- Python packages: `furhat-remote-api`, `mlx-whisper`, `sounddevice`, `numpy`, `openai`
- `ffmpeg`
- LM Studio OpenAI-compatible server at `http://127.0.0.1:8080/v1`

```bash
python3 -m pip install furhat-remote-api mlx-whisper sounddevice numpy openai
brew install ffmpeg
```
The package name is mlx-whisper. The import is mlx_whisper.

## Script

furhat_whisper.py on the Mac.

| Setting | Value |
| :--- | :--- |
| Robot | `FurhatRemoteAPI("192.168.1.10")` |
| Model server | `http://127.0.0.1:8080/v1` |
| Model | `lmstudio-community/Qwen3.8-27B-MLX-6bit` |
| Whisper | `mlx-community/whisper-large-v3-turbo` |
| Sample rate | `16000` |
| Silence to stop | `0.8 s after speech` |
| Max listen | 8s |
| Start timeout | 5 s if nobody speaks |

Listening uses one sounddevice.InputStream, not repeated sd.rec calls. It calibrates the noise floor for 500 ms, then treats a level above max(noise * 8, 0.01) as speech. A pause ends the turn. The clip is sent to Whisper only after that pause.

Speech:
```python
furhat.say(text=reply, blocking=True)
```
Replies are kept to one or two short spoken sentences.

## Qwen

Thinking mode makes message.content empty after a few turns and leaves the sentence in reasoning_content. The script:

sends enable_thinking: false and reasoning_effort: "low"
reads content, then reasoning_content if content is empty
strips <think> blocks
keeps only the system prompt plus the last 16 turns
In LM Studio, turn Enable Thinking off for this model. If the raw message is still an empty content plus a long think block, the server setting is overriding the request.

## Run
Start the LM Studio local server and load the Qwen model.
Confirm Furhat's IP. The chest or web interface shows it.
Run python3 furhat_whisper.py.
Speak after Speak now.

A fixed 4-second record was the first version. Pause detection replaced it. If the mic never stops, the speech threshold is above your voice: lower the multiplier from 8 toward 4, or lower the floor from 0.01.

## Notes

- The Mac microphone is next to the robot. Furhat's own mic is not used.
- blocking=True waits until Furhat finishes, so the next listen does not start during playback.
- Empty transcripts are skipped. An empty model reply says "Please say that again" and does not append that turn to history.
- Username for ssh is furnix.
