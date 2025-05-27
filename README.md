# 🚚 Voice Agent for Logistics

This project uses **LiveKit**, **OpenAI**, **Deepgram**, and **Cartesia TTS** to create an autonomous voice agent that can:

- 📞 Make outbound voice calls
- 🧠 Talk using LLM (GPT-4o)
- 🗣️ Transcribe and speak in real time
- ✅ Collect freight quotes from logistics companies

---

## 🧠 Agent Description

### `logistics_agent.py` ✅
An AI agent that **calls logistics companies** to gather freight quotes.

- Collects:
  - Origin & destination
  - Truck type (reefer, dry van, flatbed, etc.)
  - Price
  - Pickup availability
- Saves the quote as a `.json` file in the `quotes/` folder.

---

## 🔧 Environment Setup

Create a `.env.local` file in your root directory:

```env
LIVEKIT_URL=https://your.livekit.server
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret

SIP_OUTBOUND_TRUNK_ID=TRXXXXXXXXXXXXXXXXXXXXXXXX
OPENAI_API_KEY=your_openai_key
DEEPGRAM_API_KEY=your_deepgram_key
```

---

## 🐍 Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Agent

```bash
python logistics_agent.py
```

---

## 🗃️ Data Output

Quotes are saved under:

```
quotes/quote_<phone_number>_<timestamp>.json
```

---

## 📦 Dependencies

See `requirements.txt` for packages used:  
LiveKit Agents, Deepgram, OpenAI, Cartesia TTS, dotenv, etc.

---

## 📜 License

MIT License
