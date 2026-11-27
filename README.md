# SceneSolver: Automated Crime Video Analysis and Forensic Intelligence Platform

SceneSolver is an end-to-end multi-modal artificial intelligence system designed for automated surveillance footage analysis, forensic evidence tagging, and incident narrative synthesis. By orchestrating zero-shot vision-language models with transformer-based sequence-to-sequence summarization, SceneSolver ingests raw CCTV video or imagery, detects violent anomalies, extracts forensic cues, and compiles contextual incident reports for investigative workflows.

---

## Architecture Overview

SceneSolver decouples user identity management from compute-intensive computer vision and natural language processing tasks. The system comprises three primary tiers:

1. **Client Interface (React / Tailwind CSS)**: Provides an interactive dashboard for media upload, frame-by-frame visual verification, authentication, and forensic report review.
2. **AI Inference Engine (Flask / PyTorch / Hugging Face Transformers)**: Executes computer vision processing, OpenCV frame subsampling, CLIP zero-shot classification, forensic vocabulary extraction, and BART abstractive summarization.
3. **Authentication and Application Service (Node.js / Express / MongoDB)**: Manages secure user registration, credential hashing with bcrypt, session tokens via JSON Web Tokens (JWT), and request validation via Joi.

```mermaid
flowchart TD
    User([Investigator / User]) -->|Uploads Video / Image| Client[React Frontend]
    User -->|Sign Up / Login| Client
    Client -->|JWT Auth Requests| NodeAuth[Node.js / Express Auth Server]
    NodeAuth <-->|User Credentials & Profiles| MongoDB[(MongoDB Database)]
    Client -->|Multipart Media Upload| FlaskAI[Flask AI Inference Engine]
    
    subgraph AI Pipeline
        FlaskAI --> Subsample[OpenCV Frame Subsampling]
        Subsample --> CLIP[OpenAI CLIP Vision-Language Model]
        CLIP -->|Logits / Cosine Similarity| ZeroShot[Crime Category Classification]
        CLIP -->|Multi-Label Ranking| Evidence[Forensic Evidence Extractor]
        CLIP -->|Corpus Semantic Alignment| CaptionMatch[Generalized Caption Selector]
        CaptionMatch --> BART[BART Large CNN Summarizer]
        BART --> Story[Synthesized Incident Narrative]
    end

    ZeroShot --> ResponseAggregator[Response Formatter]
    Evidence --> ResponseAggregator
    Story --> ResponseAggregator
    ResponseAggregator -->|JSON Forensic Report| Client
```

---

## Machine Learning Pipeline

### 1. Frame Subsampling and Temporal Extraction
Raw video inputs are processed using OpenCV (`cv2.VideoCapture`). The engine dynamically samples frames at a calibrated frequency (default: 1 frame per second up to 30 keyframes), discarding redundant background frames to optimize throughput while preserving temporal fidelity across critical action intervals.

### 2. Multi-Class Crime Classification
Images and extracted video keyframes are evaluated against a high-level incident taxonomy using OpenAI's `clip-vit-base-patch32` zero-shot classification head:
- Robbery
- Explosion
- Shoplifting
- Shooting
- Fight
- Normal Action

Logits across candidate text prompts are normalized through a softmax function. For video media, a majority-voting consensus algorithm aggregates frame-level predictions to identify the dominant crime classification.

### 3. Forensic Evidence Extraction
Each frame is cross-referenced against a forensic label dictionary encompassing over 150 visual descriptors across:
- **Combustion & Explosions**: Flames, Smoke, Blast Plume, Shockwave, Scorch marks, Crater, Debris.
- **Physical Altercations & Violence**: Punch, Grapple, Choke, Headlock, Tackled, Bleeding, Melee, Kick.
- **Weaponry & Armed Threat**: Handgun, Rifle, Shotgun, Knife, Shell casing, Bullet holes, Armed assailant.
- **Tactical & Behavioral Patterns**: Masked, Forced, Looting, Fleeing, Intruder, Defensive posture.

Predictions are scored and ranked per frame; the highest-frequency evidence indicators across the entire temporal sequence are compiled into the top 10 forensic findings.

### 4. Corpus-Aligned Caption Matching
The system loads a curated repository of forensic narrative descriptions (`generalized_captions_generalized.txt`). Keyframes are matched to the most semantically aligned caption via cross-modal cosine similarity over CLIP image-text embeddings.

### 5. Abstractive Incident Report Generation
Aggregated unique captions are passed to Facebook's `bart-large-cnn` abstractive sequence-to-sequence model. The summarizer parses temporal cues, action descriptions, and contextual observations, outputting a structured, human-readable forensic summary detailing the sequence of events.

---

## Technology Stack

### Artificial Intelligence and Machine Learning
- **PyTorch**: Deep learning backend and tensor computation.
- **Hugging Face Transformers**: Pretrained model execution for `openai/clip-vit-base-patch32` and `facebook/bart-large-cnn`.
- **OpenCV (cv2)**: Video stream decoding, frame rate parsing, and image transformations.
- **PIL (Pillow)**: Image loading and preprocessing pipelines.

### AI Inference Microservice
- **Python 3.10+**: Runtime environment.
- **Flask**: HTTP microservice exposing REST inference endpoints.
- **Flask-CORS**: Cross-Origin Resource Sharing middleware enabling secure client-server communication.

### Authentication Backend
- **Node.js & Express.js**: Asynchronous REST API framework.
- **MongoDB & Mongoose**: Document storage for user authentication data.
- **JSON Web Token (jsonwebtoken)**: Stateless Bearer token generation and authorization verification.
- **bcryptjs**: Cryptographic salted password hashing.
- **Joi**: Strict schema validation for incoming request bodies.

### User Interface
- **React 18**: Component-based single-page application (SPA).
- **React Router DOM**: Client-side declarative routing (`/login`, `/signup`, `/dashboard`, `/tool`).
- **Tailwind CSS**: Utility-first CSS framework for responsive layout styling.
- **Axios**: Asynchronous HTTP client for multipart form and JSON payloads.
- **Lucide React**: Vector iconography for interface controls.

---

## Repository Structure

```text
CrimeSceneSolver/
|-- backend/
|   |-- Controller/
|   |   `-- AuthController.js        # User signup and login handlers
|   |-- Middlewares/
|   |   |-- Auth.js                  # JWT verification middleware
|   |   `-- AuthValidation.js        # Joi schema definitions for payloads
|   |-- Models_db/
|   |   |-- User.js                  # Mongoose User schema definition
|   |   `-- db.js                    # MongoDB connection lifecycle management
|   |-- Routes/
|   |   |-- AuthRouter.js            # Auth routing (/auth/signup, /auth/login)
|   |   `-- ProductRouter.js         # Protected resource demonstration route
|   |-- uploads/                     # Temporary media upload spool directory
|   |-- app.py                       # Flask AI inference application server
|   |-- index.js                     # Node.js authentication application server
|   `-- pipeline.py                  # PyTorch/CLIP/BART video and image analysis logic
|-- frontend/
|   |-- public/                      # Static assets and HTML entry point
|   |-- src/
|   |   |-- components/              # UI components (Navbar, Footer, Features, Team)
|   |   |-- pages/                   # Application views
|   |   |   |-- Dashboard.jsx        # User overview and quick actions
|   |   |   |-- Login.jsx            # Account authentication form
|   |   |   |-- SignUp.jsx           # User registration form
|   |   |   `-- ToolPage.jsx         # Video upload and forensic report view
|   |   |-- App.js                   # Application route hierarchy
|   |   |-- index.js                 # React root DOM hydration
|   |   `-- utils.js                 # Shared front-end utility helpers
|   `-- tailwind.config.js           # Tailwind theme configuration
|-- generalized_captions_generalized.txt # Curated semantic caption corpus
|-- train_model.ipynb                # CLIP fine-tuning and evaluation notebook
|-- .gitignore                       # Git exclusion rules
`-- README.md                        # Project documentation
```

---

## API Specifications

### 1. AI Inference Microservice (Flask, Port `5000`)

#### Analyze Media File
```http
POST /analyze
Content-Type: multipart/form-data
```

**Parameters:**
- `media`: The video (`.mp4`, `.avi`, `.mov`) or image (`.jpg`, `.png`, `.jpeg`) binary payload.

**Response (`200 OK`):**
```json
{
  "crime": "robbery",
  "evidence": [
    "Handgun",
    "Cash register",
    "Masked",
    "Covering face",
    "Threatening"
  ],
  "final_summary": "Surveillance footage confirms a forced armed intrusion where a masked suspect approached the counter brandishing a handgun. Demands were made for cash assets before the subject fled the scene."
}
```

**Error Response (`400 / 500`):**
```json
{
  "error": "No media uploaded"
}
```

---

### 2. Authentication API (Node.js / Express, Port `8080`)

#### User Registration
```http
POST /auth/signup
Content-Type: application/json
```
```json
{
  "name": "Investigator Name",
  "email": "investigator@agency.gov",
  "password": "StrongPassword123"
}
```

#### User Login
```http
POST /auth/login
Content-Type: application/json
```
```json
{
  "email": "investigator@agency.gov",
  "password": "StrongPassword123"
}
```
**Response (`200 OK`):**
```json
{
  "message": "Login Success",
  "success": true,
  "jwtToken": "<JSON_WEB_TOKEN>",
  "name": "Investigator Name"
}
```

---

## Installation and Setup

### Prerequisites
- Python 3.10 or higher
- Node.js 18.x or higher and npm
- MongoDB running locally or accessible via MongoDB Atlas URI
- CUDA-compatible GPU (recommended for real-time inference; CPU supported as fallback)

---

### 1. Flask AI Inference Engine Setup

1. Navigate to the backend directory:
   ```bash
   cd CrimeSceneSolver/backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required Python packages:
   ```bash
   pip install flask flask-cors torch torchvision transformers opencv-python pillow tqdm
   ```
4. Verify configuration paths:
   In `backend/pipeline.py`, ensure the caption corpus path points to your local repository directory:
   ```python
   # Default expectation: root project folder
   corpus_path = os.path.abspath("../generalized_captions_generalized.txt")
   ```
5. Launch the Flask inference server:
   ```bash
   python app.py
   ```
   The service will listen on `http://localhost:5000`.

---

### 2. Node.js Authentication Service Setup

1. Open a new terminal session in `CrimeSceneSolver/backend`:
   ```bash
   cd CrimeSceneSolver/backend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file in `backend/`:
   ```env
   PORT=8080
   MONGO_CONN=mongodb://localhost:27017/scenesolver
   JWT_SECRET=your_secure_jwt_secret_key_here
   ```
4. Start the Node.js server:
   ```bash
   npm start
   # Or for development with auto-reload:
   npx nodemon index.js
   ```
   The service will listen on `http://localhost:8080`.

---

### 3. React Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd CrimeSceneSolver/frontend
   ```
2. Install node dependencies:
   ```bash
   npm install
   ```
3. Start the Vite/Create-React-App development server:
   ```bash
   npm start
   ```
4. Access the user interface at `http://localhost:3000`.

---

## Model Weights & Checkpoints (Hugging Face)

Due to file size constraints on git version control (849 MB), the fine-tuned model checkpoint and categorical label encoders are hosted on Hugging Face:

- **Hugging Face Model Repository**: [https://huggingface.co/preethamaleti/crimescenesolver](https://huggingface.co/preethamaleti/crimescenesolver)
- **Hosted Artifacts**:
  - `best_model.pth` (849 MB): PyTorch state dictionary of the fine-tuned multi-modal CLIP model.
  - `label_encoder.pt` (1.8 KB): Serialized Scikit-learn LabelEncoder mapping output class logits to forensic crime categories.

### Automated Weight Retrieval
The backend inference engine (`backend/pipeline.py`) automatically checks for `best_model.pth` in the project root. If absent, it fetches the checkpoint directly from Hugging Face if `huggingface_hub` is installed:

```python
from huggingface_hub import hf_hub_download

# Download and cache fine-tuned weights
model_weights_path = hf_hub_download(
    repo_id="preethamaleti/crimescenesolver",
    filename="best_model.pth"
)
```

Alternatively, download the file manually and place it in the project root:
```bash
# Direct download from Hugging Face
curl -L -o best_model.pth https://huggingface.co/preethamaleti/crimescenesolver/resolve/main/best_model.pth
```

---

## Datasets and Benchmark Reference

The crime classification model was trained and validated on surveillance benchmarks curated from real-world CCTV feeds:

### 1. UCF-Crime Anomaly Detection Dataset
- **Academic Citation**: Sultani, W., Chen, C., & Shah, M. *"Real-world Anomaly Detection in Surveillance Videos"*, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
- **Dataset Resource**: [UCF CRCV Project Page](https://www.crcv.ucf.edu/research/projects/real-world-anomaly-detection-in-surveillance-videos/)
- **Kaggle Mirror**: [Kaggle Crime Dataset](https://www.kaggle.com/datasets/mission-ai/crime-dataset)
- **Data Configuration**: 128 hours of untrimmed surveillance videos across real-world CCTV cameras. For SceneSolver, incident videos are processed into 64 temporal keyframes per event, annotated with dense frame-level forensic action captions (`captions_*.json`).
- **Target Classification Categories**: Robbery, Explosion, Shoplifting, Shooting, Fighting, and Normal Action.

### 2. Base Foundation Models
- **Vision-Language Backbone**: OpenAI's [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) (224x224 input resolution, 512-dimensional shared multi-modal embedding space).
- **Abstractive Text Summarizer**: Facebook's [BART Large CNN](https://huggingface.co/facebook/bart-large-cnn) (Bidirectional and Auto-Regressive Transformer fine-tuned on CNN/DailyMail).

---

## Model Training and Customization

For complete training pipelines, refer to `train_model.ipynb`. The notebook details:
1. Ingesting UCF-Crime video directories and parsing frame-level JSON caption annotations.
2. Generating normalized visual tensors using `CLIPProcessor`.
3. Constructing a custom `CLIPClassifier` projection head (`nn.Linear(512, num_classes)`).
4. Optimization with AdamW ($lr = 5\times 10^{-6}$) and Cross-Entropy loss over train/validation splits.
5. Exporting fine-tuned weights (`best_model.pth`) and class encoders (`label_encoder.pt`).

---

## Security Considerations

- **File Upload Limits**: Ensure reverse proxies (such as Nginx) enforce maximum client body limits to prevent resource exhaustion from large uncompressed media streams.
- **Sanitized Filenames**: Media files are processed within dedicated spool directories; paths should be sanitized to avoid directory traversal.
- **Token Invalidation**: Authentication tokens must be transmitted securely over HTTPS in production deployments.
