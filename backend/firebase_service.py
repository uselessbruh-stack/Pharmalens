"""Firebase token verification and server-owned per-user Firestore records."""

import os
import threading
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException, Request

_lock = threading.Lock()
_app = None


def auth_mode():
    mode = os.getenv("PHARMALENS_AUTH_MODE", "local")
    if mode not in {"local", "firebase"}:
        raise RuntimeError("PHARMALENS_AUTH_MODE must be local or firebase")
    return mode


def firebase_app():
    global _app
    with _lock:
        if _app is None:
            import firebase_admin

            project = os.getenv("FIREBASE_PROJECT_ID")
            if not project:
                raise RuntimeError("FIREBASE_PROJECT_ID is required")
            _app = firebase_admin.initialize_app(options={"projectId": project}, name="pharmalens")
    return _app


def verify_token(token):
    from firebase_admin import auth

    return auth.verify_id_token(token, app=firebase_app(), check_revoked=True)


def authenticate(authorization):
    if auth_mode() == "local":
        return {"uid": "local", "local": True}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Sign in to access PharmaLens", headers={"WWW-Authenticate": "Bearer"})
    token = authorization[7:]
    if not token or len(token) > 8192:
        raise HTTPException(401, "Invalid authentication token")
    try:
        claims = verify_token(token)
    except Exception as exc:
        # Configuration/service failures must never fall back to anonymous access.
        if isinstance(exc, (RuntimeError, OSError)):
            raise HTTPException(503, "Authentication service is not configured or unavailable") from exc
        raise HTTPException(401, "Session expired or invalid; sign in again") from exc
    if not claims.get("uid") or claims["uid"] == "local":
        raise HTTPException(401, "Invalid user identity")
    if not claims.get("email_verified"):
        raise HTTPException(403, "Verify your email address before using the workspace")
    return {"uid": claims["uid"], "email": claims.get("email", ""), "local": False}


def current_user(request: Request):
    return request.state.user


def database():
    from firebase_admin import firestore

    return firestore.client(app=firebase_app())


def user_document(user):
    return database().collection("users").document(user["uid"])


def save_profile(user):
    if not user.get("local"):
        user_document(user).set({"email": user.get("email", ""), "lastSeenAt": datetime.now(timezone.utc)}, merge=True)


def save_prediction(user, result, smiles, sequence):
    if user.get("local"):
        return None
    record_id = uuid.uuid4().hex
    user_document(user).collection("predictions").document(record_id).set({
        "createdAt": datetime.now(timezone.utc), "kind": "single",
        "smiles": smiles, "sequence": sequence, **result,
    })
    return record_id


def prediction_history(user, limit=30):
    if user.get("local"):
        return []
    from google.cloud.firestore_v1 import Query

    return [{"id": item.id, **item.to_dict()} for item in
            user_document(user).collection("predictions").order_by("createdAt", direction=Query.DESCENDING).limit(limit).stream()]


def save_batch(user, record_id, metadata, result=None):
    if user.get("local"):
        return
    document = user_document(user).collection("batches").document(record_id)
    # Each result row is a separate document to stay below Firestore's document limit.
    # Publishing status last prevents clients from treating partial writes as complete.
    document.set({"updatedAt": datetime.now(timezone.utc), **metadata}, merge=True)
    if result is not None:
        rows, inputs = result["results"], result["inputs"]
        client = database()
        for start in range(0, len(rows), 200):
            batch = client.batch()
            for index in range(start, min(start + 200, len(rows))):
                batch.set(document.collection("rows").document(str(index)), {"input": inputs[index], **rows[index]})
            batch.commit()
        document.set({"status": "completed", "total": len(rows), "model_id": result["model_id"],
                      "artifact_sha256": result["artifact_sha256"], "updatedAt": datetime.now(timezone.utc)}, merge=True)
