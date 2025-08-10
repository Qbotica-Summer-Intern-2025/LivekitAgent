import { useState, useCallback } from "react";
import { LiveKitRoom, RoomAudioRenderer } from "@livekit/components-react";
import "@livekit/components-styles";
import SimpleVoiceAssistant from "./SimpleVoiceAssistant";

const LiveKitModal = ({ setShowSupport }) => {
  const [isSubmittingName, setIsSubmittingName] = useState(true);
  const [name, setName] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [token, setToken] = useState(null);
  const [roomName, setRoomName] = useState(null);
  const [error, setError] = useState(null);

  const flaskServerUrl = import.meta.env.VITE_FLASK_SERVER_URL || "http://localhost:5001";

  const getToken = useCallback(async (userName, phone) => {
    try {
      setError(null);
      const cleanedPhone = phone.replace(/[^0-9+]/g, "");
      const roomResponse = await fetch(`${flaskServerUrl}/api/getActiveRoom?phone_number=${encodeURIComponent(cleanedPhone)}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });
      if (!roomResponse.ok) {
        throw new Error(`Failed to fetch room name: ${roomResponse.statusText}`);
      }
      const { room } = await roomResponse.json();
      if (!room) {
        throw new Error("No room name returned");
      }

      const response = await fetch(`${flaskServerUrl}/api/getToken`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",

        },
        body: JSON.stringify({ name: userName, room, phone_number: cleanedPhone }),
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch token: ${response.statusText}`);
      }
      const data = await response.json();
      setToken(data.token);
      setRoomName(data.room);
      console.log("Connected to room:", data.room);
      setIsSubmittingName(false);
    } catch (error) {
      console.error("Error fetching token:", error);
      setError(error.message);
    }
  }, [flaskServerUrl]);

  const handleNameSubmit = (e) => {
    e.preventDefault();
    if (name.trim() && phoneNumber.trim()) {
      getToken(name, phoneNumber);
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <div className="support-room">
          {isSubmittingName ? (
            <form onSubmit={handleNameSubmit} className="name-form">
              {error && <p className="error">{error}</p>}
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Your name"
                required
              />
              <input
                type="tel"
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
                placeholder="Your phone number (e.g., +1234567890)"
                required
              />
              <button type="submit">Connect</button>
              <button
                type="button"
                className="cancel-button"
                onClick={() => setShowSupport(false)}
              >
                Cancel
              </button>
            </form>
          ) : token ? (
            <LiveKitRoom
              serverUrl={import.meta.env.VITE_LIVEKIT_URL}
              token={token}
              connect={true}
              video={false}
              audio={true}
              onDisconnected={() => {
                setShowSupport(false);
                setIsSubmittingName(true);
                setToken(null);
                setRoomName(null);
                setError(null);
              }}
            >
              <RoomAudioRenderer />
              <SimpleVoiceAssistant name={name} />
            </LiveKitRoom>
          ) : (
            <p>Loading...</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default LiveKitModal;