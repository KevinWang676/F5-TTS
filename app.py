import os
import tempfile
import requests
import uuid
import shutil
import logging
import io
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import soundfile as sf
from urllib.parse import urlparse
import torch
import tqdm
from cached_path import cached_path

# Import F5TTS directly from the code instead of trying to import it as a module
# Define the F5TTS class here, copied from the original code
import soundfile as sf
import torch
import tqdm
from cached_path import cached_path

# You'll need to ensure these modules are available
from model import DiT, UNetT
from model.utils import save_spectrogram
from model.utils_infer import load_vocoder, load_model, infer_process, remove_silence_for_generated_wav
from model.utils import seed_everything
import random
import sys


class F5TTS:
    def __init__(
        self,
        model_type="F5-TTS",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        local_path=None,
        device=None,
    ):
        # Initialize parameters
        self.final_wave = None
        self.target_sample_rate = 24000
        self.n_mel_channels = 100
        self.hop_length = 256
        self.target_rms = 0.1
        self.seed = -1

        # Set device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Load models
        self.load_vocoder_model(local_path)
        self.load_ema_model(model_type, ckpt_file, vocab_file, ode_method, use_ema)

    def load_vocoder_model(self, local_path):
        self.vocos = load_vocoder(local_path is not None, local_path, self.device)

    def load_ema_model(self, model_type, ckpt_file, vocab_file, ode_method, use_ema):
        if model_type == "F5-TTS":
            if not ckpt_file:
                ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT
        elif model_type == "E2-TTS":
            if not ckpt_file:
                ckpt_file = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.ema_model = load_model(model_cls, model_cfg, ckpt_file, vocab_file, ode_method, use_ema, self.device)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spect, file_spect):
        save_spectrogram(spect, file_spect)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spect=None,
        seed=-1,
    ):
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed
        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spect is not None:
            self.export_spectrogram(spect, file_spect)

        return wav, sr, spect

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("f5tts-api")

app = FastAPI(
    title="F5TTS API", 
    description="API for text-to-speech generation using F5TTS",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the F5TTS model
tts_model = None

# Create a temporary directory for storing files
TEMP_DIR = tempfile.mkdtemp()
logger.info(f"Created temporary directory: {TEMP_DIR}")

class TTSRequest(BaseModel):
    tts_text: str = Field(..., description="Text to be converted to speech")
    ref_file: str = Field(..., description="URL to reference audio file")
    ref_text: str = Field(..., description="Transcription of the reference audio")
    remove_silence: bool = Field(False, description="Whether to remove silence from generated audio")
    target_rms: float = Field(0.1, description="Target RMS value for output audio", ge=0.0, le=1.0)
    cross_fade_duration: float = Field(0.15, description="Duration of cross fade in seconds", ge=0.0, le=1.0)
    sway_sampling_coef: float = Field(-1, description="Sway sampling coefficient")
    cfg_strength: float = Field(2.0, description="Classifier-free guidance strength", ge=0.0, le=10.0)
    nfe_step: int = Field(32, description="Number of function evaluations for sampling", ge=1, le=100)
    speed: float = Field(1.0, description="Speed factor for speech generation", ge=0.5, le=2.0)
    fix_duration: Optional[float] = Field(None, description="Fixed duration in seconds (optional)")
    seed: int = Field(-1, description="Random seed (-1 for random)")

    @validator('ref_file')
    def validate_ref_file_url(cls, v):
        try:
            result = urlparse(v)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL format")
            return v
        except Exception:
            raise ValueError("Invalid URL format")

def download_file(url: str, dest_path: str):
    """Download a file from a URL to a destination path."""
    try:
        logger.info(f"Downloading file from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"File downloaded to: {dest_path}")
        
        # Verify the downloaded file is a valid audio file
        try:
            data, samplerate = sf.read(dest_path)
            logger.info(f"Verified audio file: {samplerate}Hz, shape: {data.shape}")
        except Exception as e:
            logger.error(f"Invalid audio file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Downloaded file is not a valid audio file: {str(e)}")
        
        return dest_path
    
    except requests.RequestException as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")

def cleanup_files(file_paths):
    """Clean up temporary files."""
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove file {file_path}: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize the TTS model on startup."""
    global tts_model
    logger.info("Initializing F5TTS model...")
    try:
        # We'll delay actual model initialization until the first request
        # to prevent startup failures due to missing dependencies
        # tts_model will be initialized in the endpoints as needed
        tts_model = None
        logger.info("F5TTS model will be initialized on first request")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        # Just log the error rather than raising an exception to allow server to start
        logger.error(f"Server will attempt to initialize model on first request")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    try:
        logger.info(f"Removing temporary directory: {TEMP_DIR}")
        shutil.rmtree(TEMP_DIR)
    except Exception as e:
        logger.warning(f"Failed to remove temporary directory: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Always return healthy even if model isn't initialized yet
    # This allows health checks to succeed during startup
    return {"status": "healthy", "model": "F5TTS", "initialized": tts_model is not None}

@app.get("/model_info")
async def model_info():
    """Get model information."""
    global tts_model
    
    # Initialize model if not already done
    if tts_model is None:
        try:
            logger.info("Initializing F5TTS model on model_info request...")
            tts_model = F5TTS()
            logger.info("F5TTS model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            return {
                "model_type": "F5-TTS",
                "status": "not_initialized",
                "error": str(e)
            }
    
    return {
        "model_type": "F5-TTS",
        "status": "initialized",
        "device": tts_model.device,
        "target_sample_rate": tts_model.target_sample_rate,
        "n_mel_channels": tts_model.n_mel_channels,
        "hop_length": tts_model.hop_length,
    }

@app.post("/generate_speech")
async def generate_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Generate speech based on input text and reference audio.
    
    Parameters:
    - tts_text: Text to be converted to speech
    - ref_file: URL to reference audio file
    - ref_text: Transcription of the reference audio
    - remove_silence: Whether to remove silence from generated audio
    - target_rms: Target RMS value for output audio
    - cross_fade_duration: Duration of cross fade in seconds
    - sway_sampling_coef: Sway sampling coefficient
    - cfg_strength: Classifier-free guidance strength
    - nfe_step: Number of function evaluations for sampling
    - speed: Speed factor for speech generation
    - fix_duration: Fixed duration in seconds (optional)
    - seed: Random seed (-1 for random)
    
    Returns:
        Generated audio file as streaming response
    """
    global tts_model
    
    # Initialize model if not already done
    if tts_model is None:
        try:
            logger.info("Initializing F5TTS model on first request...")
            tts_model = F5TTS()
            logger.info("F5TTS model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Failed to initialize TTS model: {str(e)}")
    
    # Create unique filenames for temporary files
    unique_id = str(uuid.uuid4())
    ref_audio_path = os.path.join(TEMP_DIR, f"ref_{unique_id}.wav")
    output_audio_path = os.path.join(TEMP_DIR, f"out_{unique_id}.wav")
    
    files_to_cleanup = [ref_audio_path, output_audio_path]
    
    try:
        # Download reference audio from URL
        download_file(request.ref_file, ref_audio_path)
        
        logger.info(f"Generating speech for text: '{request.tts_text[:50]}...'")
        
        # Generate speech using F5TTS
        wav, sr, _ = tts_model.infer(
            ref_file=ref_audio_path,
            ref_text=request.ref_text,
            gen_text=request.tts_text,
            file_wave=output_audio_path,
            file_spect=None,
            remove_silence=request.remove_silence,
            target_rms=request.target_rms,
            cross_fade_duration=request.cross_fade_duration,
            sway_sampling_coef=request.sway_sampling_coef,
            cfg_strength=request.cfg_strength,
            nfe_step=request.nfe_step,
            speed=request.speed,
            fix_duration=request.fix_duration,
            seed=request.seed
        )
        
        logger.info(f"Speech generated successfully, seed: {tts_model.seed}")
        
        # Read the audio file into memory
        with open(output_audio_path, "rb") as f:
            audio_data = f.read()
        
        # Schedule cleanup after response is sent
        background_tasks.add_task(cleanup_files, files_to_cleanup)
        
        # Return the audio data as a streaming response
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"X-Seed": str(tts_model.seed)}
        )
    
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        # Clean up files if an error occurs
        cleanup_files(files_to_cleanup)
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

# Alternative endpoint that accepts form data
@app.post("/generate_speech_form")
async def generate_speech_form(
    background_tasks: BackgroundTasks,
    tts_text: str = Form(...),
    ref_file: str = Form(...),
    ref_text: str = Form(...),
    remove_silence: bool = Form(False),
    target_rms: float = Form(0.1),
    cross_fade_duration: float = Form(0.15),
    sway_sampling_coef: float = Form(-1),
    cfg_strength: float = Form(2.0),
    nfe_step: int = Form(32),
    speed: float = Form(1.0),
    fix_duration: Optional[float] = Form(None),
    seed: int = Form(-1)
):
    """Form-based version of the generate_speech endpoint"""
    request = TTSRequest(
        tts_text=tts_text,
        ref_file=ref_file,
        ref_text=ref_text,
        remove_silence=remove_silence,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        sway_sampling_coef=sway_sampling_coef,
        cfg_strength=cfg_strength,
        nfe_step=nfe_step,
        speed=speed,
        fix_duration=fix_duration,
        seed=seed
    )
    return await generate_speech(request, background_tasks)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7990)
