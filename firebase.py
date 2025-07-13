# Render deploy cache bust
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os
import json

# Load Firebase credentials from environment variable
firebase_key = os.environ.get("FIREBASE_KEY")
if not firebase_key:
    raise ValueError("Missing FIREBASE_KEY environment variable")

# Parse the JSON string into a Python dictionary
cred_dict = json.loads(firebase_key)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)

db = firestore.client()

def store_message(user_id, sender, message):
    doc_ref = db.collection("conversations").document(user_id)
    doc = doc_ref.get()
    data = doc.to_dict() if doc.exists else {"messages": []}
    data["messages"].append({
        "sender": sender,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    })
    doc_ref.set(data)

def get_conversation(user_id):
    doc_ref = db.collection("conversations").document(user_id)
    doc = doc_ref.get()
    return doc.to_dict().get("messages", []) if doc.exists else []

def store_user_name(user_id, name):
    doc_ref = db.collection("users").document(user_id)
    doc_ref.set({"name": name}, merge=True)

def get_user_name(user_id):
    doc_ref = db.collection("users").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("name", None)
    return None

def store_user_language(user_id, language):
    doc_ref = db.collection("users").document(user_id)
    doc_ref.set({"language": language}, merge=True)

def get_user_language(user_id):
    doc_ref = db.collection("users").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("language", None)
    return None
