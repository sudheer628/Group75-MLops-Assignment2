"""
FastAPI application for Cats vs Dogs classification.
With Prometheus metrics and structured logging for monitoring.
"""

import io
import json
import logging
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from app.config import settings
from app.schemas import ErrorResponse, HealthResponse, PredictionResponse

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import CatsDogsPredictor


# =============================================================================
# Structured JSON Logger
# =============================================================================
class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "endpoint"):
            log_data["endpoint"] = record.endpoint
        if hasattr(record, "method"):
            log_data["method"] = record.method
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code
        if hasattr(record, "prediction"):
            log_data["prediction"] = record.prediction
        if hasattr(record, "confidence"):
            log_data["confidence"] = record.confidence
        if hasattr(record, "client_ip"):
            log_data["client_ip"] = record.client_ip
        if hasattr(record, "error_type"):
            log_data["error_type"] = record.error_type
        if hasattr(record, "stack_trace"):
            log_data["stack_trace"] = record.stack_trace
            
        return json.dumps(log_data)


# Configure structured logging
def setup_logging():
    """Setup JSON structured logging."""
    logger = logging.getLogger("cats_dogs_api")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with JSON format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


# =============================================================================
# Prometheus Metrics
# =============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Prediction metrics
PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total number of predictions",
    ["predicted_class"]
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Prediction confidence distribution",
    ["predicted_class"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Model inference latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# Error metrics
ERROR_COUNT = Counter(
    "api_errors_total",
    "Total number of API errors",
    ["error_type", "endpoint"]
)

# Model metrics
MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the model is loaded (1) or not (0)"
)

# Active requests
ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of requests currently being processed"
)


# =============================================================================
# Global State
# =============================================================================
predictor: CatsDogsPredictor = None
request_counter = 0


def get_request_id():
    """Generate unique request ID."""
    global request_counter
    request_counter += 1
    return f"req-{int(time.time())}-{request_counter}"


# =============================================================================
# Lifespan Management
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global predictor
    
    logger.info("Starting application", extra={"endpoint": "startup"})
    
    # Load model at startup
    try:
        predictor = CatsDogsPredictor(
            model_path=settings.MODEL_PATH,
            class_mapping_path=settings.CLASS_MAPPING_PATH,
            model_config_path=settings.MODEL_CONFIG_PATH,
            model_architecture=settings.MODEL_ARCH,
            image_size=settings.IMAGE_SIZE,
        )
        predictor.load()
        MODEL_LOADED.set(1)
        logger.info("Model loaded successfully", extra={"endpoint": "startup"})
    except Exception as e:
        MODEL_LOADED.set(0)
        logger.error(
            f"Failed to load model: {e}",
            extra={
                "endpoint": "startup",
                "error_type": type(e).__name__,
                "stack_trace": traceback.format_exc()
            }
        )
        predictor = None
    
    yield
    
    logger.info("Shutting down application", extra={"endpoint": "shutdown"})


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Binary image classification API for cats vs dogs",
    lifespan=lifespan,
)


# =============================================================================
# Middleware for Request Logging
# =============================================================================
@app.middleware("http")
async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware to log all requests with metrics."""
    request_id = get_request_id()
    start_time = time.time()
    
    # Track active requests
    ACTIVE_REQUESTS.inc()
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Process request
    try:
        response = await call_next(request)
        latency = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(latency)
        
        # Log request (skip static files and metrics endpoint)
        if not request.url.path.startswith("/static") and request.url.path != "/metrics":
            logger.info(
                f"{request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round(latency * 1000, 2),
                    "client_ip": client_ip,
                }
            )
        
        return response
        
    except Exception as e:
        latency = time.time() - start_time
        
        # Record error metrics
        ERROR_COUNT.labels(
            error_type=type(e).__name__,
            endpoint=request.url.path
        ).inc()
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=500
        ).inc()
        
        # Log error with stack trace
        logger.error(
            f"Request failed: {str(e)}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "endpoint": request.url.path,
                "latency_ms": round(latency * 1000, 2),
                "client_ip": client_ip,
                "error_type": type(e).__name__,
                "stack_trace": traceback.format_exc(),
            }
        )
        raise
        
    finally:
        ACTIVE_REQUESTS.dec()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/", response_class=FileResponse)
async def root():
    """Serve the web portal."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor and predictor.is_loaded else "unhealthy",
        version=settings.APP_VERSION,
        model_loaded=predictor.is_loaded if predictor else False,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an image contains a cat or dog.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        
    Returns:
        Prediction with confidence score
    """
    request_id = get_request_id()
    inference_start = time.time()
    
    # Check if model is loaded
    if not predictor or not predictor.is_loaded:
        ERROR_COUNT.labels(error_type="ModelNotLoaded", endpoint="/predict").inc()
        logger.error(
            "Model not loaded",
            extra={
                "request_id": request_id,
                "endpoint": "/predict",
                "error_type": "ModelNotLoaded"
            }
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service unavailable.",
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        ERROR_COUNT.labels(error_type="InvalidFileType", endpoint="/predict").inc()
        logger.warning(
            f"Invalid file type: {file.content_type}",
            extra={
                "request_id": request_id,
                "endpoint": "/predict",
                "error_type": "InvalidFileType"
            }
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Expected image.",
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Make prediction
        result = predictor.predict(image)
        
        inference_latency = time.time() - inference_start
        
        # Record prediction metrics
        PREDICTION_COUNT.labels(predicted_class=result["prediction"]).inc()
        PREDICTION_CONFIDENCE.labels(predicted_class=result["prediction"]).observe(result["confidence"])
        PREDICTION_LATENCY.observe(inference_latency)
        
        # Log prediction result
        logger.info(
            f"Prediction completed: {result['prediction']}",
            extra={
                "request_id": request_id,
                "endpoint": "/predict",
                "prediction": result["prediction"],
                "confidence": round(result["confidence"], 4),
                "latency_ms": round(inference_latency * 1000, 2),
            }
        )
        
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__, endpoint="/predict").inc()
        logger.error(
            f"Prediction failed: {str(e)}",
            extra={
                "request_id": request_id,
                "endpoint": "/predict",
                "error_type": type(e).__name__,
                "stack_trace": traceback.format_exc(),
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
