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

# Import needed for F5TTS
import random
import sys
from importlib.resources import files

import soundfile as sf
import tqdm
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    transcribe,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from f5_tts.model.utils import seed_everything


class F5TTS:
    def __init__(
        self,
        model="F5TTS_v1_Base",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_local_path=None,
        device=None,
        hf_cache_dir=None,
    ):
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.target_sample_rate = model_cfg.model.mel_spec.target_sample_rate

        self.ode_method = ode_method
        self.use_ema = use_ema

        if device is not None:
            self.device = device
        else:
            import torch

            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if torch.xpu.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        # Load models
        self.vocoder = load_vocoder(
            self.mel_spec_type, vocoder_local_path is not None, vocoder_local_path, self.device, hf_cache_dir
        )

        repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

        # override for previous models
        if model == "F5TTS_Base":
            if self.mel_spec_type == "vocos":
                ckpt_step = 1200000
            elif self.mel_spec_type == "bigvgan":
                model = "F5TTS_Base_bigvgan"
                ckpt_type = "pt"
        elif model == "E2TTS_Base":
            repo_name = "E2-TTS"
            ckpt_step = 1200000

        if not ckpt_file:
            ckpt_file = str(
                cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}", cache_dir=hf_cache_dir)
            )
        self.ema_model = load_model(
            model_cls, model_arc, ckpt_file, self.mel_spec_type, vocab_file, self.ode_method, self.use_ema, self.device
        )

    def transcribe(self, ref_audio, language=None):
        return transcribe(ref_audio, language)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spec, file_spec):
        save_spectrogram(spec, file_spec)

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
        file_spec=None,
        seed=None,
    ):
        if seed is None:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)

        wav, sr, spec = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
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

        if file_spec is not None:
            self.export_spectrogram(spec, file_spec)

        return wav, sr, spec

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
    seed: Optional[int] = Field(None, description="Random seed (None for random)")
    model: str = Field("F5TTS_v1_Base", description="Model name to use")
    auto_transcribe: bool = Field(False, description="Automatically transcribe reference audio if ref_text is empty")
    language: Optional[str] = Field(None, description="Language for auto transcription (if auto_transcribe is enabled)")

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
                "model_type": "F5TTS_v1_Base",
                "status": "not_initialized",
                "error": str(e)
            }
    
    return {
        "model_type": "F5TTS_v1_Base",
        "status": "initialized",
        "device": tts_model.device,
        "target_sample_rate": tts_model.target_sample_rate,
        "mel_spec_type": tts_model.mel_spec_type,
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
    - seed: Random seed (None for random)
    - model: Model name to use
    - auto_transcribe: Automatically transcribe reference audio if ref_text is empty
    - language: Language for auto transcription
    
    Returns:
        Generated audio file as streaming response
    """
    global tts_model
    
    # Initialize model if not already done
    if tts_model is None or (hasattr(tts_model, 'model') and tts_model.model != request.model):
        try:
            logger.info(f"Initializing F5TTS model '{request.model}' on first request...")
            tts_model = F5TTS(model=request.model)
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
        
        # Check if transcription is needed
        ref_text = request.ref_text
        if request.auto_transcribe and (not ref_text or ref_text.strip() == ""):
            logger.info("Auto-transcribing reference audio...")
            ref_text = tts_model.transcribe(ref_audio_path, request.language)
            logger.info(f"Auto-transcription result: {ref_text}")
        
        logger.info(f"Generating speech for text: '{request.tts_text[:50]}...'")
        
        # Generate speech using F5TTS
        wav, sr, spec = tts_model.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=request.tts_text,
            file_wave=output_audio_path,
            file_spec=None,
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
        response_headers = {
            "X-Seed": str(tts_model.seed),
            "X-Model": request.model
        }
        
        # Add transcription to headers if auto-transcribed
        if request.auto_transcribe and (not request.ref_text or request.ref_text.strip() == ""):
            response_headers["X-Transcription"] = ref_text
            
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers=response_headers
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
    seed: Optional[int] = Form(None),
    model: str = Form("F5TTS_v1_Base"),
    auto_transcribe: bool = Form(False),
    language: Optional[str] = Form(None)
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
        seed=seed,
        model=model,
        auto_transcribe=auto_transcribe,
        language=language
    )
    return await generate_speech(request, background_tasks)

@app.post("/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    ref_file: str = Form(...),
    language: Optional[str] = Form(None)
):
    """
    Transcribe reference audio.
    
    Parameters:
    - ref_file: URL to reference audio file
    - language: Optional language hint for transcription
    
    Returns:
        Transcription text
    """
    global tts_model
    
    # Initialize model if not already done
    if tts_model is None:
        try:
            logger.info("Initializing F5TTS model for transcription...")
            tts_model = F5TTS()
            logger.info("F5TTS model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Failed to initialize TTS model: {str(e)}")
    
    # Create unique filenames for temporary files
    unique_id = str(uuid.uuid4())
    ref_audio_path = os.path.join(TEMP_DIR, f"ref_{unique_id}.wav")
    
    files_to_cleanup = [ref_audio_path]
    
    try:
        # Download reference audio from URL
        download_file(ref_file, ref_audio_path)
        
        # Transcribe the audio
        logger.info("Transcribing reference audio...")
        transcription = tts_model.transcribe(ref_audio_path, language)
        logger.info(f"Transcription result: {transcription}")
        
        # Schedule cleanup after response is sent
        background_tasks.add_task(cleanup_files, files_to_cleanup)
        
        return {"transcription": transcription}
    
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        # Clean up files if an error occurs
        cleanup_files(files_to_cleanup)
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7990)
