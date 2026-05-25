"""FastAPI backend for SceneMotion-LLM inference."""

import os
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from pathlib import Path

from scene_motion_llm.models.sentiment_classifier import SceneMotionLLMModel
from scene_motion_llm.inference import SceneMotionInferencer
from scene_motion_llm.utils.config import CHECKPOINT_PATH, FRAME_SIZE, FPS, MAX_FRAMES, DEVICE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title='SceneMotion-LLM API',
    description='REST API for video sentiment analysis powered by motion understanding and LLM explanations.',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

DEVICE_SETUP = DEVICE
if DEVICE_SETUP == 'cuda' and not torch.cuda.is_available():
    DEVICE_SETUP = 'cpu'

MODEL = SceneMotionLLMModel(
    spatial_feature_dim=512,
    motion_feature_dim=2,
    temporal_hidden_dim=256,
    attention_dim=128,
    fusion_dim=512,
    num_classes=3,
    max_frames=MAX_FRAMES,
    dropout=0.3,
    pretrained=False
)

INFERENCER = SceneMotionInferencer(
    model=MODEL,
    checkpoint_path=os.path.join(CHECKPOINT_PATH, 'best_model.pt'),
    device=DEVICE_SETUP,
    max_frames=MAX_FRAMES,
    frame_size=FRAME_SIZE,
    fps=FPS
)


@app.get('/health')
async def health_check():
    return {
        'status': 'ok',
        'device': DEVICE_SETUP,
        'cuda_available': torch.cuda.is_available(),
        'model_loaded': INFERENCER.model is not None
    }


@app.post('/predict')
async def predict_sentiment(file: UploadFile = File(...)):
    if file.content_type not in ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/mkv']:
        raise HTTPException(status_code=400, detail='Unsupported file type')

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, 'wb') as fh:
            fh.write(await file.read())

        result = INFERENCER.analyze_video(
            temp_path,
            output_dir=temp_dir,
            save_frames=False
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f'Inference error: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
