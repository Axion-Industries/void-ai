# Core imports
import hashlib
import logging
import os
import pickle
import signal
import time
from collections import defaultdict
from datetime import datetime
from functools import wraps

# Model imports
import torch

# Flask imports
from flask import Flask, jsonify, request, send_from_directory

from model import GPT, GPTConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("void-z1")


# --- Security settings ---
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "1000"))

# Rate limiting
request_counts = defaultdict(lambda: {"count": 0, "window_start": time.time()})


def get_client_identifier():
    """Get a unique identifier for the client."""
    identifier = str(request.headers.get("X-Forwarded-For", request.remote_addr))
    return hashlib.sha256(identifier.encode()).hexdigest()


def rate_limit(f):
    """Rate limiting decorator."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_id = get_client_identifier()
        current_time = time.time()
        client_data = request_counts[client_id]

        # Reset window if expired
        if current_time - client_data["window_start"] > RATE_LIMIT_WINDOW:
            client_data["count"] = 0
            client_data["window_start"] = current_time

        # Check rate limit
        if client_data["count"] >= RATE_LIMIT_REQUESTS:
            response = jsonify(
                {
                    "error": "Rate limit exceeded",
                    "reset_time": datetime.fromtimestamp(
                        client_data["window_start"] + RATE_LIMIT_WINDOW
                    ).isoformat(),
                }
            )
            response.status_code = 429
            return response

        client_data["count"] += 1
        return f(*args, **kwargs)

    return decorated_function


def add_security_headers(response):
    """Add security headers to response."""
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Content-Security-Policy": "default-src 'self' 'unsafe-inline'",
        "Strict-Transport-Security": ("max-age=31536000; includeSubDomains"),
    }
    for key, value in headers.items():
        response.headers[key] = value
    return response


class TimeoutException(Exception):
    """Exception raised when operation times out."""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeouts."""
    raise TimeoutException()


class time_limit:
    """Context manager for time limits."""

    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "out", "model.pt")
VOCAB_PATH = os.path.join(BASE_DIR, "data", "void", "vocab.pkl")
META_PATH = os.path.join(BASE_DIR, "data", "void", "meta.pkl")

# --- Check for required files ---
required_files = [MODEL_PATH, VOCAB_PATH, META_PATH]
missing_files = [f for f in required_files if not os.path.exists(f)]

app = Flask(__name__, static_folder=".", static_url_path="")
app.after_request(add_security_headers)


@app.route("/health")
def health_check():
    """Health check endpoint for Void Z1."""
    return jsonify({"status": "ok"})


@app.route("/")
def serve_index():
    """Serve the main frontend for Void Z1."""
    return send_from_directory(".", "index.html", cache_timeout=300)


# --- Model and vocab loading ---
if missing_files:
    logger.error(f"Missing required model files: {', '.join(missing_files)}")
    model = None
    stoi = None
    itos = None
else:
    try:
        logger.info("Loading model and vocabulary...")
        with open(VOCAB_PATH, "rb") as f:
            chars, stoi = pickle.load(f)
        itos = {i: ch for ch, i in stoi.items()}
        vocab_size = len(chars)
        logger.info(f"Loaded vocabulary with size {vocab_size}")
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        logger.info("Loaded meta configuration")
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=meta.get("block_size", 64),
            n_layer=meta.get("n_layer", 4),
            n_head=meta.get("n_head", 4),
            n_embd=meta.get("n_embd", 128),
            dropout=0.0,
            bias=meta.get("bias", True),
        )
        logger.info("Loading model weights...")
        model = GPT(config)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        model = None
        stoi = None
        itos = None


# --- Rate limiting ---
# REMOVE the following redundant code:
# RATE_LIMIT = 5
# RATE_WINDOW = 1
# request_counts = defaultdict(list)
# def is_rate_limited(ip): ...
# @app.before_request
# def check_rate_limit(): ...


# --- Chat endpoint ---
@app.route("/chat", methods=["POST"])
@rate_limit
def chat():
    if missing_files or not model or not stoi or not itos:
        msg = (
            "Missing required model files: "
            f"{', '.join(missing_files)}. "
            "Please train your model and add these files to your repo."
        )
        return jsonify({"error": msg}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        if len(prompt) > MAX_PROMPT_LENGTH:
            error_msg = (
                "Prompt too long. Maximum length is " f"{MAX_PROMPT_LENGTH} characters."
            )
            return jsonify({"error": error_msg}), 400
        max_new_tokens = min(int(data.get("max_new_tokens", 100)), 500)
        temperature = max(0.1, min(float(data.get("temperature", 0.8)), 2.0))
        logger.info(
            "Chat request - tokens: %d, temp: %.2f", max_new_tokens, temperature
        )
        with time_limit(30):
            encoded = torch.tensor([stoi[c] for c in prompt], dtype=torch.long)
            torch.cuda.empty_cache()
            with torch.no_grad():
                output = model.generate(
                    encoded.unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            completion = "".join([itos[int(i)] for i in output[0].tolist()])
            return jsonify({"text": completion[len(prompt) :]})
    except TimeoutException:
        logger.error("Request timed out")
        return jsonify({"error": "Request timed out"}), 408
    except Exception as e:
        logger.error("Error processing request: %s", str(e))
        return jsonify({"error": str(e)}), 500


def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()


@app.after_request
def after_request(response):
    cleanup_memory()
    return response


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error. Please try again later."}), 500


# --- Static file serving ---
@app.route("/<path:path>")
def serve_static(path):
    """Serve static files with cache control for Void Z1."""
    response = send_from_directory(".", path)
    if path.endswith((".js", ".css", ".html")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
