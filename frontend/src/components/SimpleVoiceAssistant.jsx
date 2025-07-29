import {
  useVoiceAssistant,
  BarVisualizer,
  VoiceAssistantControlBar,
  useTrackTranscription,
  useLocalParticipant,
  useParticipants,
} from "@livekit/components-react";
import { Track } from "livekit-client";
import { useEffect, useState, useRef } from "react";
import "./SimpleVoiceAssistant.css";

const Message = ({ type, text, name }) => {
  return (
    <div className="message">
      <strong className={`message-${type}`}>
        {type === "agent" ? "Qubi: " : `${name || "You"}: `}
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

// Separate component for handling SIP transcription
const SipTranscriptionHandler = ({ sipParticipant, onTranscriptions }) => {
  const sipAudioTrack = sipParticipant?.audioTrackPublications?.values().next().value;

  const { segments: userTranscriptions } = useTrackTranscription({
    publication: sipAudioTrack || null,
    source: Track.Source.Microphone,
    participant: sipParticipant || null, // Fixed syntax
  });

  useEffect(() => {
    if (onTranscriptions) {
      onTranscriptions(userTranscriptions || []);
    }
  }, [userTranscriptions, onTranscriptions]);

  return null; // This component doesn't render anything
};

const SimpleVoiceAssistant = ({ name }) => {
  const { state, audioTrack, agentTranscriptions } = useVoiceAssistant();
  const { localParticipant } = useLocalParticipant();
  const participants = useParticipants();

  // Find the SIP participant (Twilio phone caller)
  const sipParticipant = participants.find(p => p?.identity?.startsWith('sip_'));

  const [userTranscriptions, setUserTranscriptions] = useState([]);
  const [messages, setMessages] = useState([]);

  // Create a ref for the conversation container
  const conversationRef = useRef(null);

  // Auto-scroll function
  const scrollToBottom = () => {
    if (conversationRef.current) {
      conversationRef.current.scrollTop = conversationRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    console.log('=== DEBUG INFO ===');
    console.log('SIP participant:', sipParticipant);
    console.log('Agent transcriptions:', agentTranscriptions);
    console.log('User transcriptions:', userTranscriptions);
    console.log('=== END DEBUG ===');

    const allMessages = [
      ...(agentTranscriptions?.map((t) => ({ ...t, type: "agent" })) ?? []),
      ...(userTranscriptions?.map((t) => ({ ...t, type: "user" })) ?? []),
    ].sort((a, b) => a.firstReceivedTime - b.firstReceivedTime);
    setMessages(allMessages);
  }, [agentTranscriptions, userTranscriptions, sipParticipant]);

  // Auto-scroll whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const isLocalActive = state === "listening" || state === "thinking";
  const isAiActive = state === "speaking";

  return (
    <div className="voice-assistant-container">
      {/* Only render the SIP transcription handler if we have a SIP participant */}
      {sipParticipant && (
        <SipTranscriptionHandler
          sipParticipant={sipParticipant}
          onTranscriptions={setUserTranscriptions}
        />
      )}

      <div className="main-content">
        <div className="participants-section">
          <ParticipantCircle
            name="Qubi"
            isActive={isAiActive}
          />
          <ParticipantCircle
            name={name || "You"}
            isActive={isLocalActive}
            isLocal={true}
          />
        </div>

        <div className="visualizer-container">
          <BarVisualizer state={state} barCount={7} trackRef={audioTrack} />
        </div>
      </div>

      <div className="conversation-panel">
        <div className="conversation" ref={conversationRef}>
          {messages.length === 0 ? (
            <div className="no-messages">
              Start speaking to begin the conversation...
            </div>
          ) : (
            messages.map((msg, index) => (
              <Message
                key={msg.id || index}
                type={msg.type}
                text={msg.text}
                name={name}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default SimpleVoiceAssistant;