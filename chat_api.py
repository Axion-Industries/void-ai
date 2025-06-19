# Core imports
import hashlib
import logging
<<<<<<< Updated upstream
=======
from logging.handlers import RotatingFileHandler
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
from werkzeug.utils import safe_join
>>>>>>> Stashed changes

from model import GPT, GPTConfig

# Configure logging
<<<<<<< Updated upstream
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


=======
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        RotatingFileHandler(
            os.path.join(log_dir, "app.log"),
            maxBytes=10485760,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("void-z1")

# --- Security settings ---
# ... (rest of your code follows)

# --- Timeout helpers ---
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
=======
@app.route("/<path:path>")
def serve_static(path):
    safe_path = safe_join(".", path)
    if not safe_path or not os.path.isfile(safe_path):
        return jsonify({"error": "File not found"}), 404
    response = send_from_directory(".", path)
    if path.endswith((".js", ".css", ".html")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
=======
# --- Model and vocab loading ---
if missing_files:
    logger.error(f"Missing required model files: {', '.join(missing_files)}")
    logger.error("Please ensure the following files exist:")
    logger.error(f"- Model: {MODEL_PATH}")
    logger.error(f"- Vocabulary: {VOCAB_PATH}")
    logger.error(f"- Meta: {META_PATH}")
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
        logger.error("Stack trace:", exc_info=True)
        model = None
        stoi = None
        itos = None
>>>>>>> Stashed changes

# --- Rate limiting ---
# REMOVE the following redundant code:
# RATE_LIMIT = 5
# RATE_WINDOW = 1
# request_counts = defaultdict(list)
# def is_rate_limited(ip): ...
# @app.before_request
# def check_rate_limit(): ...


<<<<<<< Updated upstream
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
=======
@app.before_request
def check_rate_limit():
    if request.endpoint == 'chat':
        ip = request.remote_addr
        if is_rate_limited(ip):
            return jsonify({'error': 'Too many requests. Please wait a moment.'}), 429

@app.route("/chat", methods=["POST"])
def chat():
    if missing_files or not model or not stoi or not itos:
        msg = (
            "Server is not properly initialized. "
            "Required model files are missing. Please try again later."
        )
        logger.error(msg)
        return jsonify({"error": msg}), 503  # Service Unavailable
        
>>>>>>> Stashed changes
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
<<<<<<< Updated upstream
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
=======
            
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
            
        if len(prompt) > 5000:
            error_msg = f"Prompt too long. Maximum length is 5000 characters."
            return jsonify({"error": error_msg}), 400
            
        max_new_tokens = min(int(data.get("max_new_tokens", 100)), 500)
        temperature = max(0.1, min(float(data.get("temperature", 0.8)), 2.0))
        
        logger.info(
            "Chat request - tokens: %d, temp: %.2f, prompt_len: %d",
            max_new_tokens, temperature, len(prompt)
        )
        
        with time_limit(30):
            # Clean up any GPU memory before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            encoded = torch.tensor([stoi[c] for c in prompt], dtype=torch.long)
            
>>>>>>> Stashed changes
            with torch.no_grad():
                output = model.generate(
                    encoded.unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
<<<<<<< Updated upstream
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
=======
                
            completion = "".join([itos[int(i)] for i in output[0].tolist()])
            response_text = completion[len(prompt):]
            
            # Log successful response
            logger.info(
                "Generated response - length: %d chars",
                len(response_text)
            )
            
            return jsonify({"text": response_text})
            
    except TimeoutException:
        logger.error("Request timed out")
        return jsonify({"error": "Request timed out. Please try again."}), 408
        
    except ValueError as ve:
        logger.error("Value error: %s", str(ve))
        return jsonify({"error": "Invalid input parameters"}), 400
        
    except Exception as e:
        logger.error("Error processing request: %s", str(e), exc_info=True)
        return jsonify({
            "error": "An unexpected error occurred. Please try again later."
        }), 500

def cleanup_memory():
    """Clean up memory after each request."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        import gc
        gc.collect()
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
>>>>>>> Stashed changes


@app.after_request
def after_request(response):
    """After request handler to clean up resources."""
    cleanup_memory()
    return response

<<<<<<< Updated upstream

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
=======
@app.errorhandler(500)
def server_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(e)}", exc_info=True)
    return jsonify({
        "error": "An unexpected error occurred. Please try again later."
    }), 500

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'"
    return response

@app.route("/health")
def health_check():
    """Health check endpoint for Void Z1."""
    health_status = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "model_files": {
                "status": "ok" if not missing_files else "error",
                "details": {
                    "model": os.path.exists(MODEL_PATH),
                    "vocab": os.path.exists(VOCAB_PATH),
                    "meta": os.path.exists(META_PATH)
                }
            },
            "model_loaded": {
                "status": "ok" if model and stoi and itos else "error"
            },
            "gpu": {
                "available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
            }
        }
    }
    
    # Overall status is ok only if all components are ok
    if any(
        component["status"] == "error" 
        for component in health_status["components"].values()
    ):
        health_status["status"] = "error"
        return jsonify(health_status), 503
        
    return jsonify(health_status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
>>>>>>> Stashed changes
