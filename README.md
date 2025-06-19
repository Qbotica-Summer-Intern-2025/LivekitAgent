# Qbotica Logistics Agent

The **Qbotica Logistics Agent** is a voice-enabled AI system that streamlines logistics operations using LiveKit and AI plugins like Deepgram, OpenAI, and Silero. It consists of two agents:

- **Outbound Agent** – Initiates calls to negotiate profitable shipment prices and confirm dispatch dates with logistics partners.
- **Inbound Agent** – Receives calls from customers, provides shipping quotes, and engages with natural language responses.

---

## 🚀 Features

### Outbound Agent
- Initiates outbound calls to logistics companies.
- Negotiates shipping rates with a 25% initial markup, adjustable to 15%.
- Confirms sending dates with persuasive natural-sounding responses.

### Inbound Agent
- Handles incoming SIP calls via LiveKit.
- Responds to quote requests based on origin/destination or shipment ID.
- Offers friendly engagement, including jokes.
- Auto-terminates after 5 minutes of inactivity.

---

## 📦 Prerequisites

- Python 3.8+
- `.env` file with required API keys and config
- `logistics_dataset.csv` with route data

### Required Python Libraries
Install them via:
```bash
pip install -r requirements.txt
```

Minimal `requirements.txt`:
```
livekit
pandas
aiohttp
python-dotenv
```

---

## 🔐 Environment Variables (`.env`)

```env
LIVEKIT_URL=your_livekit_url
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret
DEEPGRAM_API_KEY=your_deepgram_key
OPENAI_API_KEY=your_openai_key
LOGISTICS_PHONE_NUMBER=+1xxxxxxxxxx
SIP_INBOUND_TRUNK_ID=your_trunk_id
```

---

## 📊 Dataset Structure (`logistics_dataset.csv`)

### Required columns:
- `origin_city`, `origin_state`
- `destination_city`, `destination_state`
- `fuel_cost_usd`, `toll_cost_usd`
- `weight_lbs`, `distance_miles`
- `shipment_id` (optional for inbound)

### Example row:
```
Chicago,IL,Miami,FL,100.00,20.00,500.0,1200.0,SHP00001
```

---

## 🛠 Usage

### ▶️ Running the Outbound Agent
```bash
python outbound.py
```

- Uses `LOGISTICS_PHONE_NUMBER` from `.env`
- Handles negotiation and date confirmation

### 📞 Running the Inbound Agent
```bash
python inbound.py
```

- Waits for incoming SIP calls
- Responds with quotes or jokes

---

## 🧩 Troubleshooting

- **Calls drop instantly**: Confirm correct SIP trunk ID and active call route.
- **Quote errors**: Validate your dataset and markup logic in `get_quote_from_dataset`.
- **Logs**: Enable `logging.DEBUG` to get detailed output.

---

## 🤝 Contributing

Pull requests are welcome! Please:

- Stick to the existing code style
- Update docs/tests as needed
- Open an issue before major changes

---

## 📄 License

MIT License 
