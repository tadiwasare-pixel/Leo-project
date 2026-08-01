import os
import json
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import pymongo
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, auth
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Load environment variables from .env file (local)
load_dotenv()

# ============================================================
# CONFIGURATION (read from environment)
# ============================================================
GEMINI_KEY = os.getenv("GEMINI_KEY")
MONGO_URI = os.getenv("MONGO_URI")
FIREBASE_CRED_JSON = os.getenv("FIREBASE_CRED_JSON")   # entire JSON as string
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH")   # or path to .json file

if not GEMINI_KEY:
    raise Exception("GEMINI_KEY environment variable not set")
if not MONGO_URI:
    raise Exception("MONGO_URI environment variable not set")

# ============================================================
# FIREBASE ADMIN SDK INITIALIZATION
# ============================================================
def init_firebase():
    if FIREBASE_CRED_PATH and os.path.exists(FIREBASE_CRED_PATH):
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase initialized from file:", FIREBASE_CRED_PATH)
    elif FIREBASE_CRED_JSON:
        try:
            cred_dict = json.loads(FIREBASE_CRED_JSON)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            print("✅ Firebase initialized from env var")
        except Exception as e:
            raise Exception(f"Failed to parse FIREBASE_CRED_JSON: {e}")
    else:
        raise Exception("No Firebase credentials provided (set FIREBASE_CRED_JSON or FIREBASE_CRED_PATH)")

init_firebase()

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        decoded = auth.verify_id_token(token)
        return decoded["uid"]
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

# ============================================================
# MONGODB
# ============================================================
client = pymongo.MongoClient(MONGO_URI)
db = client["leo"]
chats_col = db["chats"]
chats_col.create_index("user_id")
chats_col.create_index("timestamp")

def get_history(user_id: str, limit: int = 20):
    docs = chats_col.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
    return [{"message": d["message"], "reply": d["reply"]} for d in docs]

def save_chat(user_id: str, message: str, reply: str):
    chats_col.insert_one({
        "user_id": user_id,
        "message": message,
        "reply": reply,
        "timestamp": time.time()
    })

# ============================================================
# GEMINI AI
# ============================================================
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

def build_prompt(user_message: str, context: List[str]) -> str:
    context_str = "\n".join([f"- {c}" for c in context[-5:]]) if context else "No previous context."
    return f"""
You are Leo, an enthusiastic, expert physics tutor and 3D builder.
Explain concepts simply, step-by-step.
When you mention an object (sphere, cylinder, pendulum, box), a mass, gravity, or color,
the frontend will automatically update its 3D scene.
Example: "Imagine a red sphere with mass 5kg..." triggers scene updates.

Previous context:
{context_str}

User: {user_message}

Leo (Physics Tutor):"""

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Leo Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # For production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# PYDANTIC MODELS
# ============================================================
class ChatRequest(BaseModel):
    message: str

class SceneUpdateRequest(BaseModel):
    objectType: Optional[str] = None
    color: Optional[str] = None
    mass: Optional[float] = None
    gravity: Optional[float] = None
    isAnimating: Optional[bool] = None
    position: Optional[List[float]] = None

# ============================================================
# SCENE STATE (in-memory – resets on restart)
# ============================================================
scene_state = {
    "objectType": "box",
    "color": "#f43f5e",
    "mass": 1.0,
    "gravity": 9.8,
    "isAnimating": False,
    "position": [0, 1, 0]
}

# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, user_id: str = Depends(verify_token)):
    """Streams the AI response with context from MongoDB."""
    # 1. Get history
    history = get_history(user_id, limit=10)
    context = [h["message"] for h in history]

    # 2. Build prompt
    prompt = build_prompt(request.message, context)

    # 3. Generator for streaming
    def generate():
        full_reply = ""
        try:
            response = gemini_model.generate_content(
                prompt,
                stream=True,
                request_options={"timeout": 30}
            )
            for chunk in response:
                if chunk.text:
                    full_reply += chunk.text
                    yield chunk.text
        except Exception as e:
            yield f"\n\n⚠️ Error: {str(e)}"
        finally:
            # Save after streaming completes
            if full_reply:
                save_chat(user_id, request.message, full_reply)

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/api/history")
async def history_endpoint(user_id: str = Depends(verify_token)):
    """Returns the user's chat history."""
    return get_history(user_id, limit=50)

@app.get("/api/scene/state")
async def get_scene_state():
    """Get the current scene state."""
    return scene_state

@app.post("/api/scene/update")
async def update_scene_state(request: SceneUpdateRequest):
    """Update the 3D scene state."""
    if request.objectType is not None:
        scene_state["objectType"] = request.objectType
    if request.color is not None:
        scene_state["color"] = request.color
    if request.mass is not None:
        scene_state["mass"] = request.mass
    if request.gravity is not None:
        scene_state["gravity"] = request.gravity
    if request.isAnimating is not None:
        scene_state["isAnimating"] = request.isAnimating
    if request.position is not None:
        scene_state["position"] = request.position
    return {"status": "ok", "scene": scene_state}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Leo Backend v1.0"}

# ============================================================
# RUN SERVER (production uses PORT from environment)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)