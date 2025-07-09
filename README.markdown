# Polaris Logistics Agent Deployment

This project consists of a Python backend (Flask server and LiveKit agent) and a React frontend for the Polaris Logistics Agent application. Follow these steps to deploy it on a Windows server.

## Prerequisites
- **Operating System**: Windows Server (or Windows 10/11 for testing)
- **Tools**:
  - Python 3.10+ (download from https://www.python.org)
  - Node.js 18+ (download from https://nodejs.org)
  - Redis server (running on localhost:6379 or a cloud-hosted instance, download from https://redis.io/download)
- **Ports**: Ensure ports 5001 (backend) and 3000 (frontend) are open in the firewall
- **External Services**:
  - LiveKit server (URL, API key, and secret)
  - API keys for Deepgram, OpenAI, Google Maps
  - SMTP credentials for email notifications (optional, for orders with >7 skids)
  - SIP trunk ID for LiveKit voice functionality

## Project Structure
```
polaris-logistics/
├── server.py
├── inbound.py
├── order_management.py
├── faq_dataset.py
├── requirements.txt
├── .env
├── start-backend.bat
├── App.jsx
├── main.jsx
├── SimpleVoiceAssistant.jsx
├── index.css
├── App.css
├── SimpleVoiceAssistant.css
├── package.json
├── public/
│   ├── index.html
│   ├── test.png
├── start-frontend.bat
├── README.md
```

## Setup Instructions

1. **Copy the Project Files**:
   - Place all files in a directory on the server (e.g., `C:\polaris-logistics`).

2. **Configure Environment Variables**:
   - Open `.env` and fill in the required values:
     ```
     LIVEKIT_URL=<your-livekit-url>
     LIVEKIT_API_KEY=<your-livekit-api-key>
     LIVEKIT_API_SECRET=<your-livekit-api-secret>
     DEEPGRAM_API_KEY=<your-deepgram-api-key>
     OPENAI_API_KEY=<your-openai-api-key>
     GOOGLE_MAPS_API_KEY=<your-google-maps-api-key>
     SIP_INBOUND_TRUNK_ID=<your-sip-trunk-id>
     SMTP_SERVER=smtp.gmail.com
     SMTP_PORT=587
     SMTP_USERNAME=<your-smtp-username>
     SMTP_PASSWORD=<your-smtp-password>
     RATES_TEAM_EMAIL=rates@polarislogistics.com
     TRANSFER_PHONE_NUMBER=+14807172012
     ```
   - Note: SMTP settings are required for email notifications (e.g., for orders with >7 skids). If not provided, email functionality will be disabled.

3. **Install Redis**:
   - Download and install Redis for Windows from https://redis.io/download.
   - Start Redis with:
     ```
     redis-server
     ```
   - Ensure Redis is running on `localhost:6379` (default) or update the Redis host in `server.py` and `inbound.py` if using a cloud-hosted Redis.

4. **Run the Backend**:
   - Open Command Prompt and navigate to the project directory:
     ```
     cd C:\polaris-logistics
     ```
   - Run the backend script:
     ```
     start-backend.bat
     ```
   - This sets up a Python virtual environment, installs dependencies, and starts:
     - Flask server on `http://0.0.0.0:5001`
     - LiveKit agent for voice functionality

5. **Run the Frontend**:
   - Open a new Command Prompt and navigate to the project directory:
     ```
     cd C:\polaris-logistics
     ```
   - Run the frontend script:
     ```
     start-frontend.bat
     ```
   - This installs Node.js dependencies, builds the React app, and serves it on `http://0.0.0.0:3000`.

6. **Access the Application**:
   - Backend API: `http://<server-ip>:5001` (e.g., `http://localhost:5001` for testing)
   - Frontend: `http://<server-ip>:3000` (e.g., `http://localhost:3000` for testing)

## Notes
- Ensure Redis is running before starting the backend.
- If using a cloud-hosted Redis, update `server.py` and `inbound.py` with the correct Redis host and port.
- For production, consider:
  - Using a reverse proxy like Nginx for the frontend (instead of `npx serve`) with HTTPS.
  - Running the backend with a process manager like Gunicorn (e.g., `gunicorn -w 4 server:app`).
- The backend requires valid API keys in `.env` for LiveKit, Deepgram, OpenAI, and Google Maps to function fully.

## Troubleshooting
- **Backend Errors**:
  - Check logs in Command Prompt for `start-backend.bat`.
  - Ensure `.env` has valid API keys and SIP trunk ID.
  - Verify Redis is running (`redis-cli ping` should return `PONG`).
- **Frontend Errors**:
  - Check logs in Command Prompt for `start-frontend.bat`.
  - Ensure Node.js is installed and `package.json` includes all dependencies.
- **Port Issues**:
  - Ensure ports 5001 and 3000 are open in the server’s firewall.
- **SMTP Warning**:
  - If you see `SMTP configuration incomplete`, provide SMTP credentials in `.env` or ignore if email functionality isn’t needed.

## Optional: Production Deployment
- For better performance, use Nginx to serve the frontend `build/` folder with HTTPS.
- Use a process manager like PM2 for the frontend or Gunicorn for the backend.
- Consider Docker for easier deployment (contact the developer for Docker files if needed).