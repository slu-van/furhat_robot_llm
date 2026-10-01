from furhat_remote_api import FurhatRemoteAPI
from openai import OpenAI

ROBOT_IP = "192.168.1.10"
LLM_URL = "http://127.0.0.1:8080/v1"
MODEL = "lmstudio-community/Qwen3.8-27B-MLX-6bit"

furhat = FurhatRemoteAPI(ROBOT_IP)
llm = OpenAI(base_url=LLM_URL, api_key="local")

history = [{
    "role": "system",
    "content": "You are Furhat, a social robot. Reply in one or two short spoken sentences. No markdown, no lists./no_think"
}]

print("Connected to robot", ROBOT_IP)
furhat.say(text="Hello. I am ready to talk.", blocking=True)

while True:
    print("\nListening... speak after this line")
    try:
        result = furhat.listen(language="en-US")
    except TypeError:
        result = furhat.listen()

    print("RAW:", result)

    if result is None:
        text = ""
    elif isinstance(result, str):
        text = result
    elif isinstance(result, dict):
        text = result.get("message") or result.get("text") or ""
    else:
        text = getattr(result, "message", None) or getattr(result, "text", None) or ""

    text = str(text).strip()
    print("TEXT:", repr(text))

    if not text or text.upper() in {"SILENCE", "INTERRUPTED", "FAILED"}:
        print("No speech recognized. Try again.")
        continue

    print("USER:", text)
    history.append({"role": "user", "content": text + "no_think"})

    try:
        resp = llm.chat.completions.create(
            model=MODEL,
            messages=history,
            max_tokens=256,
            temperature=0.3,
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        print("LLM RAW:", resp)
        choice = resp.choices[0].message
        print("content:", repr(choice.content))
        print("reasoning:", repr(getattr(choice, "reasoning_content", None)))
        print("finish:", resp.choices[0].finish_reason)
        reply = choice.content or getattr(choice, "reasoning_content", None) or ""
        reply = str(reply).strip()
    except Exception as e:
        print("LLM error:", e)
        furhat.say(text="I could not reach the language model.", blocking=True)
        continue

    if not reply:
        print("Empty model reply")
        furhat.say(text="I did not get a reply. Please say that again.", blocking=True)
        continue

    history.append({"role": "assistant", "content": reply})
    if len(history) > 13:
        history = history[:1] + history[-12:]

    print("ROBOT:", reply)
    furhat.say(text=reply, blocking=True)
