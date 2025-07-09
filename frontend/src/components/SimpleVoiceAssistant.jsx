import {
  useVoiceAssistant,
  BarVisualizer,
  VoiceAssistantControlBar,
  useTrackTranscription,
  useLocalParticipant,
  useParticipants,
} from "@livekit/components-react";
import { Track } from "livekit-client";
import { useEffect, useState } from "react";
import "./SimpleVoiceAssistant.css";

const Message = ({ type, text }) => {
  return (
    <div className="message">
      <strong className={`message-${type}`}>
        {type === "agent" ? "Qubi: " : "You: "}
      </strong>
      <span className="message-text">{text}</span>
    </div>
  );
};

const ParticipantCircle = ({ name, isActive, isLocal = false }) => {
  return (
    <div className="participant-circle">
      <div className={`circle ${isActive ? 'active' : ''}`}>
        <div className="participant-icon">
          {isLocal ? (
            <svg className="icon" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
            </svg>
          ) : (
            <svg className="icon" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>
          )}
        </div>
        {isActive && (
          <div className="mic-indicator">
            <svg className="mic-icon" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
              <path d="M17 11H19c0 3.53-2.61 6.43-6 6.92V21h-2v-3.08c-3.39-.49-6-3.39-6-6.92h2c0 2.76 2.24 5 5 5s5-2.24 5-5z"/>
            </svg>
          </div>
        )}
      </div>
      <div className="participant-info">
        <div className="participant-name">{name}</div>
        <div className="participant-status">Connected</div>
      </div>
    </div>
  );
};

const SimpleVoiceAssistant = () => {
  const { state, audioTrack, agentTranscriptions } = useVoiceAssistant();
  const localParticipant = useLocalParticipant();
  const participants = useParticipants();
  const { segments: userTranscriptions } = useTrackTranscription({
    publication: localParticipant.microphoneTrack,
    source: Track.Source.Microphone,
    participant: localParticipant.localParticipant,
  });

  const [messages, setMessages] = useState([]);

  useEffect(() => {
    const allMessages = [
      ...(agentTranscriptions?.map((t) => ({ ...t, type: "agent" })) ?? []),
      ...(userTranscriptions?.map((t) => ({ ...t, type: "user" })) ?? []),
    ].sort((a, b) => a.firstReceivedTime - b.firstReceivedTime);
    setMessages(allMessages);
  }, [agentTranscriptions, userTranscriptions]);

  const isLocalActive = state === "listening" || state === "thinking";
  const isAiActive = state === "speaking";

  return (
    <div className="voice-assistant-container">
      <div className="main-content">
        <div className="title-section">
          <h2 className="main-title">Live Polaris Logistics </h2>
        </div>

        <div className="participants-section">
          <ParticipantCircle
            name="Qubi"
            isActive={isAiActive}
          />
          <ParticipantCircle
            name="You"
            isActive={isLocalActive}
            isLocal={true}
          />
        </div>

        <div className="visualizer-container">
          <BarVisualizer state={state} barCount={7} trackRef={audioTrack} />
        </div>

        <div className="control-section">
          <VoiceAssistantControlBar />
        </div>
      </div>

      <div className="conversation-panel">
        <div className="conversation">
          {messages.length === 0 ? (
            <div className="no-messages">
              Start speaking to begin the conversation...
            </div>
          ) : (
            messages.map((msg, index) => (
              <Message key={msg.id || index} type={msg.type} text={msg.text} />
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default SimpleVoiceAssistant;