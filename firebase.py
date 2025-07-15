import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os


cred = credentials.Certificate(os.environ["FIREBASE_KEY"])
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


def increment_warning_count(user_id):
    """Increment warning count for explicit language violations"""
    doc_ref = db.collection("users").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        current_warnings = doc.to_dict().get("warning_count", 0)
        new_warnings = current_warnings + 1
        doc_ref.set({"warning_count": new_warnings, "last_warning": datetime.utcnow().isoformat()}, merge=True)
        return new_warnings
    else:
        doc_ref.set({"warning_count": 1, "last_warning": datetime.utcnow().isoformat()})
        return 1

def get_warning_count(user_id):
    """Get current warning count for a user"""
    doc_ref = db.collection("users").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("warning_count", 0)
    return 0

def reset_warning_count(user_id):
    """Reset warning count (for new sessions)"""
    doc_ref = db.collection("users").document(user_id)
    doc_ref.set({"warning_count": 0}, merge=True)

def is_session_cancelled(user_id):
    """Check if user's session is cancelled due to too many warnings"""
    warning_count = get_warning_count(user_id)
    return warning_count >= 3

