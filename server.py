import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from livekit import api
from livekit.api import LiveKitAPI, ListRoomsRequest
import redis.asyncio as redis

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

async def get_rooms():
    api = LiveKitAPI()
    rooms = await api.room.list_rooms(ListRoomsRequest())
    await api.aclose()
    return [room.name for room in rooms]

async def get_dynamic_room_name(phone_number=None):
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        if phone_number:
            room = await redis_client.get(f"room:{phone_number}")
        else:
            room = await redis_client.get("active_room")
        if room:
            return room.decode('utf-8')
        raise ValueError("No active room found in Redis")
    except Exception as e:
        print(f"Error fetching room name from Redis: {e}")
        room = os.getenv("LIVEKIT_ROOM_NAME")
        if room:
            return room
        raise ValueError("No room name available and LIVEKIT_ROOM_NAME not set")

@app.route("/api/getActiveRoom", methods=["GET"])
async def get_active_room():
    phone_number = request.args.get("phone_number", None)
    try:
        room = await get_dynamic_room_name(phone_number)
        return jsonify({"room": room})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/getToken", methods=["POST"])
async def get_token_post():
    data = request.get_json()
    name = data.get("name", "my name")
    room = data.get("room", None)
    phone_number = data.get("phone_number", None)

    if not room:
        room = await get_dynamic_room_name(phone_number)

    if not room.startswith("call-"):
        return jsonify({"error": "Invalid room name format"}), 400

    token = api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")) \
        .with_identity(name) \
        .with_name(name) \
        .with_grants(api.VideoGrants(
            room_join=True,
            room=room
        ))

    return jsonify({"token": token.to_jwt(), "room": room})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)