from flask import Flask, request, jsonify
from flask_cors import CORS
from firebase import store_message, get_conversation
import os, requests
from dotenv import load_dotenv
import re
from firebase import store_user_language, get_user_language
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from datetime import datetime, timezone, timedelta
SESSION_TIMEOUT_MINUTES = 3

load_dotenv()
app = Flask(__name__)
CORS(app)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
# model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
print("GODEL model loaded.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BAD_WORDS = ["sex", "fuck", "nude", "horny", "boobs"]

def detect_language(text):
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    if hindi_chars > len(text) * 0.3:  # 30% or more Devanagari
        return "hindi"
    return "english"

# def ai_reply(context, user_lang):
#     try:
#         instruction = (
#             "Given the dialog context, generate a helpful, supportive, and emotionally intelligent response.\n"
#         )

#         if user_lang == "hindi":
#             instruction += "Reply only in pure Hindi with empathy.\n"
#         else:
#             instruction += "Reply only in simple, caring English.\n"

#         # GODEL expects input as: [CONTEXT] => [RESPONSE]
#         prompt = f"{instruction} \n {context}"

#         inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
#         output_ids = model.generate(
#             **inputs,
#             max_length=200,
#             num_beams=4,
#             do_sample=True,
#             temperature=0.7,
#             top_k=50,
#             top_p=0.95
#         )

#         reply = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         return reply.strip()

#     except Exception as e:
#         print("❌ Exception in ai_reply():", e)
#         return "Sorry, I couldn't respond right now. Please try again later."


def ai_reply(context, user_lang):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    language_rule = ""
    if user_lang == "hindi":
        language_rule = "- Always reply ONLY in pure Hindi. Do not mix English words.\n"
    else:
        language_rule = "- Always reply ONLY in clear English. Do not mix Hindi.\n"

    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "system",
                "content":
                    "You are SayHey, a safe, warm, and emotionally supportive chatbot that listens with love and empathy.\n"
                    "Rules:\n"
                    + language_rule +
                    "- Users should feel you like a human support.\n"
                    "- Never speak explicitly or tolerate inappropriate language.\n"
                    "- If the user tries inappropriate language, warn them once. On repeated offenses, end the session and report.\n"
                    "- Always be engaging — ask open-ended questions that help users reflect and feel heard.\n"
                    "- Remember previous conversations to maintain emotional continuity.\n"
                    "- You are not a therapist, but a trained empathy listener.\n"
                    "- Be efficient — no wasteful words!\n"
            },
            {
                "role": "user",
                "content": context
            }
        ]
    }

    try:
        r = requests.post(url, headers=headers, json=payload)
        print("▶️ Groq Response:", r.status_code, r.text)
        print("Using model:", payload["model"])
        print("Authorization Header:", headers["Authorization"][:20] + "...")

        if r.status_code != 200:
            return "Sorry, I'm having trouble responding right now. Please try again later."

        response_json = r.json()
        return response_json["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("❌ Exception in ai_reply():", e)
        return "Oops! Something went wrong. Try again later."


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    uid = data["user_id"]
    user_msg = data["message"]

    if any(bad in user_msg.lower() for bad in BAD_WORDS):
        store_message(uid, "user", user_msg)
        return jsonify({"response": "⚠️ This kind of language isn't allowed. Please speak kindly or I’ll end the session."})

    history = get_conversation(uid)

    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    is_new_session = True

    if history:
        last_msg = history[-1]
        last_ts = datetime.fromisoformat(last_msg["timestamp"]).replace(tzinfo=timezone.utc)
        gap = now - last_ts

        if gap <= timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            is_new_session = False


    detected_lang = detect_language(user_msg)
    store_user_language(uid, detected_lang)

    # Get latest saved preference
    user_lang = get_user_language(uid) or "english"


    # ✅ If session timed out, generate summary
    if is_new_session and history:
        convo_text = "\n".join(f"{m['sender']}: {m['message']}" for m in history if m['sender'] in ['user', 'bot'])
        summary_prompt = f"Summarize this conversation in 2-3 sentences so we can remember it later:\n{convo_text}"
        summary = ai_reply(summary_prompt,user_lang)
        store_message(uid, "summary", summary)

        session_intro = "It's been a while! Welcome back. Here's what we discussed earlier:\n"
    else:
        session_intro = ""
    
    # ✅ If there is a summary, use that as context instead of last 6 messages
    latest_summary = None
    for m in reversed(history):
        if m['sender'] == "summary":
            latest_summary = m['message']
            break

    if latest_summary:
        base_context = f"Summary of our previous session:\n{latest_summary}\n"
    else:
        base_context = "Previous conversation:\n" + "\n".join(f"{m['sender']}: {m['message']}" for m in history[-6:])

    context = session_intro + base_context + f"\nuser: {user_msg}\nSayHey:"

    bot_reply = ai_reply(context, user_lang)
    store_message(uid, "user", user_msg)
    store_message(uid, "bot", bot_reply)

    return jsonify({"response": bot_reply})




if __name__ == "__main__":
    app.run(debug=True, port=5000)
