import os
import uuid
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis

app = FastAPI(title="ELF Malware Detector API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QUEUE_KEY = "elf_queue"

redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


@app.get("/")
def root():
    return {
        "service": "ELF Malware Detector",
        "status": "running",
        "endpoints": {
            "POST /analyze": "Upload ELF file for analysis",
            "GET /result/{job_id}": "Get analysis result"
        }
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Upload an ELF file for malware analysis."""
    job_id = str(uuid.uuid4())
    
    # Read file content
    content = await file.read()
    
    # Store file in Redis (base64 encoded)
    b64_content = base64.b64encode(content).decode("utf-8")
    file_key = f"elf:file:{job_id}"
    redis_client.set(file_key, b64_content)
    
    # Add job to queue
    redis_client.lpush(QUEUE_KEY, job_id)
    
    return {"job_id": job_id}


@app.get("/result/{job_id}")
def get_result(job_id: str):
    """Get the result of an analysis job."""
    result_key = f"elf:result:{job_id}"
    
    result = redis_client.hgetall(result_key)
    
    if not result:
        return {"status": "pending"}
    
    return {"status": "done", "result": result}


@app.get("/health")
def health():
    """Health check endpoint."""
    try:
        redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except:
        return {"status": "unhealthy", "redis": "disconnected"}
