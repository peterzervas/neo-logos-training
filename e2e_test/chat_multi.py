"""Multi-turn chat against the local llama-server to prove context is held."""
import json
import urllib.request

URL = "http://127.0.0.1:18080/v1/chat/completions"
history = []

def say(user):
    history.append({"role": "user", "content": user})
    req = urllib.request.Request(
        URL,
        data=json.dumps({
            "messages": history,
            "max_tokens": 80,
            "temperature": 0.0,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    reply = resp["choices"][0]["message"]["content"].strip()
    history.append({"role": "assistant", "content": reply})
    print(f"USER: {user}")
    print(f"BOT : {reply}\n")

say("Hello. What is your name?")
say("Nice. What OS were you trained on?")
say("And what is the name I just mentioned at the start? (what did you say your name was?)")
say("Good. Last question: was Windows involved in your training?")
