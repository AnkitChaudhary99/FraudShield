# FraudShield — Dual-Stage Fraud Call Detection

FraudShield is a machine-learning based audio analysis system designed to identify suspicious phone calls using two complementary signals:

1. **Audio authenticity** — detects whether the voice is likely to be synthetic/spoofed.
2. **Conversational intent** — transcribes the call and analyzes whether the conversation contains fraud-related intent.

The two signals are then combined into a single **risk score**.

> FraudShield performs analysis on uploaded call recordings. It is not a real-time call monitoring system.

---

## Architecture

```text
                    Audio Recording
                           │
                           ▼
                ┌─────────────────────┐
                │     Stage 1         │
                │   Anti-Spoofing     │
                │                     │
                │ Wav2Vec2 Audio      │
                │ Classification      │
                └──────────┬──────────┘
                           │
                    Spoofing Score
                           │
                           │
                           ▼
                ┌─────────────────────┐
                │     Stage 2         │
                │ Intent Recognition  │
                │                     │
                │ Whisper             │
                │      ↓              │
                │ Transcript          │
                │      ↓              │
                │ DistilBERT / MNLI   │
                └──────────┬──────────┘
                           │
                     Intent Score
                           │
                           ▼
                ┌─────────────────────┐
                │     Stage 3         │
                │   Score Fusion      │
                │                     │
                │ Weighted Average    │
                └──────────┬──────────┘
                           │
                           ▼
                    Final Risk Score
                           │
                           ▼
                 Low / Medium / High
```

---

## How the System Works

### Stage 1 — Audio Anti-Spoofing

The uploaded audio is analyzed for characteristics associated with synthetic or spoofed speech.

FraudShield uses a pretrained Wav2Vec2-based audio classification model:

```text
Vansh180/deepfake-audio-wav2vec2
```

The audio is:

- loaded at a 16 kHz sampling rate
- converted to mono audio
- limited to a fixed analysis window
- processed by the Wav2Vec2 feature extractor
- passed through the pretrained audio classification model

The output is converted into a **spoofing score between 0 and 1**.

A score closer to `1` indicates a higher likelihood of the audio being classified as spoofed/synthetic by the model.

---

### Stage 2 — Speech Transcription and Intent Recognition

The same audio recording is passed through OpenAI Whisper.

```text
Audio
  ↓
Whisper
  ↓
Transcript
```

The transcript is then analyzed using a zero-shot Natural Language Inference (NLI) classifier.

The project uses:

```text
typeform/distilbert-base-uncased-mnli
```

The transcript is compared against candidate intent labels:

- fraudulent request
- scam attempt
- asking for sensitive information
- urgent money transfer
- genuine conversation
- normal customer service
- casual talk

The fraud-related labels are aggregated into an **intent fraud score between 0 and 1**.

This allows the system to detect suspicious conversational behavior even when the voice itself does not appear synthetic.

---

## Stage 3 — Risk Score Fusion

The system combines the two independent signals:

```text
Spoofing Score
       +
Intent Score
       ↓
Weighted Fusion
       ↓
Final Risk Score
```

The default weights are:

```text
Spoofing weight = 0.5
Intent weight   = 0.5
```

Therefore:

```text
Final Score =
    0.5 × Spoofing Score
  + 0.5 × Intent Score
```

The final score is converted into a risk category:

| Final Score | Verdict |
|---|---|
| `< 0.40` | Genuine Call — Low Risk |
| `0.40 – < 0.65` | Suspicious Call — Medium Risk |
| `≥ 0.65` | Fraudulent Call Detected — High Risk |

The weights can be configured through environment variables.

---

## Web Application

FraudShield provides a FastAPI web application with a simple analysis dashboard.

The user can:

1. Upload an audio recording.
2. Start the analysis.
3. View the final risk score.
4. View the spoofing score.
5. View the fraud-intent score.
6. Read the generated transcript.
7. Inspect the intent classification breakdown.

The uploaded file is stored temporarily during processing and removed after the analysis is complete.

---

## Project Structure

```text
fraud_call_detection/
│
├── app.py
│       FastAPI web application and API endpoints
│
├── main.py
│       Command-line entry point for running the pipeline
│
├── anti_spoofing.py
│       Stage 1 — Wav2Vec2-based audio anti-spoofing
│
├── anti_spoofing_legacy.py
│       Original Mel-spectrogram CNN prototype
│
├── intent_recognition.py
│       Stage 2 — Whisper transcription and intent classification
│
├── fusion.py
│       Stage 3 — weighted risk-score fusion
│
├── test_pipeline.py
│       Automated tests for core pipeline logic
│
├── requirements.txt
│       Python dependencies
│
├── .env.example
│       Example configuration values
│
├── .gitignore
│       Files excluded from Git
│
├── Dockerfile
│       Container configuration
│
├── render.yaml
│       Render deployment configuration
│
├── README.md
│       Project documentation
│
└── static/
        └── index.html
                Web dashboard
```

---

## Technologies Used

### Machine Learning / AI

- PyTorch
- Wav2Vec2
- Whisper
- DistilBERT
- Natural Language Inference (MNLI)
- Hugging Face Transformers

### Audio Processing

- Librosa
- SoundFile
- FFmpeg

### Backend

- Python
- FastAPI
- Uvicorn
- Multipart file uploads

### Frontend

- HTML
- CSS
- JavaScript

### Development

- Git
- GitHub
- VS Code
- Docker

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd fraud_call_detection
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install FFmpeg

FFmpeg is required for processing certain audio formats used by Whisper and the audio-processing pipeline.

Verify the installation:

```powershell
ffmpeg -version
```

---

## Running the Web Application

Start the FastAPI server:

```powershell
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The first analysis may take longer because the required machine-learning models may need to be downloaded and loaded.

---

## Running from the Command Line

The pipeline can also be executed without the web interface:

```powershell
python main.py path/to/audio_file.wav
```

The command-line pipeline performs:

```text
Audio
 ↓
Anti-Spoofing
 ↓
Whisper Transcription
 ↓
Intent Recognition
 ↓
Score Fusion
 ↓
Final Verdict
```

---

## Testing

Run the automated tests with:

```powershell
python -m unittest -v test_pipeline.py
```

The tests verify important parts of the pipeline such as:

- score ranges
- score fusion
- low-risk classification
- medium-risk classification
- high-risk classification
- invalid score handling

---

## Configuration

The project supports configuration through environment variables.

Example:

```text
ANTI_SPOOF_MODEL=Vansh180/deepfake-audio-wav2vec2
WHISPER_MODEL_SIZE=base
INTENT_MODEL=typeform/distilbert-base-uncased-mnli
SPOOFING_WEIGHT=0.5
INTENT_WEIGHT=0.5
ANTI_SPOOF_MAX_SECONDS=5
MAX_UPLOAD_MB=20
ALLOWED_ORIGINS=*
```

An example configuration is provided in:

```text
.env.example
```

No API key is required for the core local pipeline.

---

## Important Model Note

The project originally contained a custom CNN that generated Mel-spectrogram features and classified them using a PyTorch CNN.

During development, it was identified that the original repository did not contain trained CNN weights. Running that CNN with randomly initialized weights would not provide a meaningful anti-spoofing detector.

The original implementation has therefore been preserved separately as:

```text
anti_spoofing_legacy.py
```

The active Stage 1 implementation uses a pretrained Wav2Vec2-based anti-spoofing classifier instead.

This makes the inference pipeline meaningful while preserving the original CNN implementation for reference.

---

## Limitations

FraudShield is a project/research prototype and should not be treated as a production fraud-detection system.

Important limitations include:

- The anti-spoofing model is pretrained rather than trained specifically for this application.
- Model performance can vary across accents, recording devices, compression formats, languages, and unseen spoofing techniques.
- Whisper transcription errors can affect intent classification.
- Zero-shot intent classification does not replace a task-specific fraud classifier trained on a representative dataset.
- The fusion weights and thresholds are configurable heuristics rather than statistically calibrated production thresholds.
- The system analyzes uploaded recordings rather than monitoring live phone calls.

A production system would require a larger representative dataset, task-specific training, calibration, extensive evaluation, monitoring, and security/privacy controls.

---

## Future Improvements

Possible improvements include:

- fine-tuning the anti-spoofing model on a larger ASVspoof-based dataset
- training a dedicated fraud-intent classifier
- adding confidence calibration
- evaluating precision, recall, F1-score and ROC-AUC
- supporting longer recordings through chunk-based analysis
- adding speaker diarization for multi-speaker calls
- adding multilingual intent recognition
- improving model serving and inference latency
- adding authentication and secure file handling for production deployment
- adding persistent analysis history and monitoring

---

## Disclaimer

FraudShield is intended for educational and research purposes.

The risk score should be treated as an automated signal for further investigation, not as definitive proof that a call is fraudulent.