from flask import Flask, request, jsonify
from flask_cors import CORS
from firebase import store_message, get_conversation, increment_warning_count, get_warning_count, is_session_cancelled, reset_warning_count
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
BAD_WORDS = [
    "sex", "fuck", "nude", "horny", "boobs", "porn", "dick", "cock", "pussy", "vagina", 
    "penis", "ass", "bitch", "slut", "whore", "cum", "suck", "blow", "fucking", "shit",
    "damn", "bastard", "motherfucker", "cunt", "twat", "pornography", "erotic", "sexy",
    "hot", "seductive", "intimate", "bedroom", "naked", "nude", "undress", "strip"
]

def contains_explicit_language(text):
    """Use AI to check if text contains explicit or inappropriate language"""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a content moderation system. Analyze the given text and determine if it contains explicit, inappropriate, or offensive language. Consider sexual content, profanity, hate speech, or any content that would be inappropriate for a supportive chatbot environment. Respond with ONLY 'YES' if inappropriate content is detected, or 'NO' if the content is appropriate."
                },
                {
                    "role": "user",
                    "content": f"Analyze this text: '{text}'\n\nIs this text inappropriate or explicit? Respond with only YES or NO."
                }
            ],
            "temperature": 0.1,
            "max_tokens": 10
        }

        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"].strip().upper()
            print(f"🤖 AI Content Check for '{text}': {result}")
            return result == "YES"
        else:
            print(f"❌ AI content check failed: {response.status_code}")
            # Fallback to basic check if AI fails
            return any(bad_word in text.lower() for bad_word in BAD_WORDS)
            
    except Exception as e:
        print(f"❌ Exception in AI content check: {e}")
        # Fallback to basic check if AI fails
        return any(bad_word in text.lower() for bad_word in BAD_WORDS)

def detect_language(text):
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    english_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')

    if hindi_chars > 0 and english_chars > 0:
        return "hinglish"
    elif hindi_chars > len(text) * 0.3:
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
    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        print("API Key:", os.getenv("ANTHROPIC_API_KEY"))
        if user_lang == "hindi":
            lang_rule = "Respond strictly in Hindi only. Do not use English words."
        elif user_lang == "english":
            lang_rule = "Respond strictly in English only. Do not use Hindi words."
        else:
            lang_rule = "Respond in Hinglish — mix Hindi and English naturally like a friendly conversation."

        payload = {
            "model": "claude-3-haiku-20240307", # You can also use claude-3-sonnet if required
            "max_tokens": 1000,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": f"""
You are SayHey, a safe, warm, and emotionally supportive chatbot that listens with love and empathy.

Rules:
- {lang_rule}
- Always be caring, empathetic, and non-judgmental.
- Never speak explicitly or tolerate inappropriate language.
- Respond like a caring human, not a bot.
- Avoid long generic replies — be concise and thoughtful.

User message and context:
{context}
"""
                }
            ]
        }

        r = requests.post(url, headers=headers, json=payload)
        print("▶️ Claude Response:", r.status_code, r.text)

        if r.status_code != 200:
            return "Sorry, I'm having trouble responding right now. Please try again later."

        return r.json()["content"][0]["text"].strip()

    except Exception as e:
        print("❌ Exception in Claude ai_reply():", e)
        return "Oops! Something went wrong. Try again later."



@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    uid = data["user_id"]
    user_msg = data["message"]
    
    print(f"📨 Received message from user {uid}: '{user_msg}'")

    # Check if user is permanently flagged due to inappropriate behavior
    current_warnings = get_warning_count(uid)
    print(f"🔍 User {uid} has {current_warnings} warnings")
    
    if current_warnings >= 3:
        print(f"🚫 User {uid} is banned (has {current_warnings} warnings) - returning ban message")
        return jsonify({
            "response": "Due to your inappropriate behaviour, You can't chat with SayHey Assistant.",
            "session_cancelled": True
        }), 403

    # Check for explicit language
    if contains_explicit_language(user_msg):
        print(f"🚨 User {uid} used explicit language: '{user_msg}'")
        warning_count = increment_warning_count(uid)
        print(f"⚠️ Warning count incremented to: {warning_count}")
        store_message(uid, "user", user_msg)
        
        if warning_count == 1:
            warning_msg = "⚠️ First warning: Please use respectful language. This is a safe space for supportive conversation."
        elif warning_count == 2:
            warning_msg = "⚠️ Second warning: Inappropriate language is not tolerated. One more violation will result in permanent ban from SayHey Assistant."
        else:  # warning_count >= 3
            warning_msg = "🚫 You have been permanently banned from SayHey Assistant due to repeated inappropriate behaviour."
            print(f"🚫 User {uid} has been permanently banned!")
        
        store_message(uid, "system", warning_msg)
        
        # If this was the 3rd warning, immediately check if user should be banned
        if warning_count >= 3:
            print(f"🚫 User {uid} reached 3 warnings - checking ban status")
            # Double-check the warning count from database
            final_warning_count = get_warning_count(uid)
            print(f"🔍 Final warning count from DB: {final_warning_count}")
            
        return jsonify({
            "response": warning_msg,
            "warning_count": warning_count,
            "session_cancelled": warning_count >= 3
        })

    print(f"✅ User {uid} passed all checks, processing normal message")
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


@app.route("/debug/<user_id>", methods=["GET"])
def debug_user(user_id):
    """Debug endpoint to check user's warning count and status"""
    warnings = get_warning_count(user_id)
    is_banned = warnings >= 3
    return jsonify({
        "user_id": user_id,
        "warning_count": warnings,
        "is_banned": is_banned,
        "message": f"User has {warnings} warnings, banned: {is_banned}"
    })


@app.route("/reset/<user_id>", methods=["POST"])
def reset_user_warnings(user_id):
    """Reset user's warning count (for testing)"""
    reset_warning_count(user_id)
    return jsonify({"message": f"Reset warnings for user {user_id}"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
