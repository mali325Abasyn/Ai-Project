from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import numpy as np
import cv2
import onnxruntime as ort

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AnimeGAN Section ---
class AnimeGAN:
    def __init__(self, model_path: str, downsize_ratio: float = 1.0) -> None:
        if not os.path.exists(model_path):
            raise Exception(f"Model doesn't exist at {model_path}")
        
        self.downsize_ratio = downsize_ratio
        providers = ['CUDAExecutionProvider'] if ort.get_device() == "GPU" else ['CPUExecutionProvider']
        self.ort_sess = ort.InferenceSession(model_path, providers=providers)

    def to_32s(self, x):
        return 256 if x < 256 else x - x % 32

    def process_frame(self, frame: np.ndarray, x32: bool = True) -> np.ndarray:
        h, w = frame.shape[:2]
        if x32:
            frame = cv2.resize(frame, (self.to_32s(int(w * self.downsize_ratio)), self.to_32s(int(h * self.downsize_ratio))))
        frame = frame.astype(np.float32) / 127.5 - 1.0
        return frame

    def post_process(self, frame: np.ndarray, wh: tuple) -> np.ndarray:
        frame = (frame.squeeze() + 1.) / 2 * 255
        frame = frame.astype(np.uint8)
        frame = cv2.resize(frame, (wh[0], wh[1]))
        return frame

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        image = self.process_frame(frame)
        outputs = self.ort_sess.run(None, {self.ort_sess.get_inputs()[0].name: np.expand_dims(image, axis=0)})
        frame = self.post_process(outputs[0], frame.shape[:2][::-1])
        return frame

# Initialize AnimeGAN object
MODEL_PATH_ANIMEGAN = "Hayao_64.onnx"
animegan = AnimeGAN(MODEL_PATH_ANIMEGAN)

@app.post("/generate-anime/")
async def generate_anime(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        transformed_image = animegan(image)
        _, encoded_image = cv2.imencode('.jpg', transformed_image)
        image_bytes = encoded_image.tobytes()

        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating anime: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the AnimeGAN API! Use the /generate-anime endpoint to upload an image for AnimeGAN transformation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8085)
