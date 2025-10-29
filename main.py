# # from fastapi import FastAPI, File, UploadFile
# # from fastapi.responses import StreamingResponse
# # import io
# # # from pydantic import BaseModel

# # app = FastAPI()

# # @app.get("/{name}")
# # def read_root(name:str):
# #     return {"message": f"Hello from {name}"}

# # @app.post("/predict")
# # async def predicted(file:UploadFile=File(...)):
# #     image_bytes=await file.read()
# # # Convert bytes to a buffer
# #     buf = io.BytesIO(image_bytes)
# #     buf.seek(0)
# #   # Return the image directly
# #     return StreamingResponse(buf, media_type="image/jpeg")

# # # class User(BaseModel):
# # #     name:str
# # #     age:int
# # # @app.post("/")
# # # def read_req(user:User):
# # #     return{"massage":f"Hello from {user.name} and age is {user.age}"}
import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi.responses import Response

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import cv2
import json
import base64
import tensorflow as tf

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://brainsightai.vercel.app/"],  # Next.js URL
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------
# Load model & class indices
# -----------------------

@app.get("/")
def root():
    return {"message": "API is running"}

MODEL_PATH = "best_resnet_model.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# model = load_model(MODEL_PATH)

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
# Convert to list where index = class index
CLASS_NAMES = [None] * len(class_indices)
for k, v in class_indices.items():
    CLASS_NAMES[v] = k

# -----------------------
# Helper functions
# -----------------------
def preprocess_image_from_bytes(file_bytes):
    """Convert uploaded bytes to model-ready array"""
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, np.array(img)
# -----------------------
# API Route
# -----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()
    img_array, original_img = preprocess_image_from_bytes(file_bytes)

    # Predict class
    preds = model.predict(img_array)
    class_idx = int(np.argmax(preds[0]))  # <--  np.argmax 
    label = CLASS_NAMES[class_idx]
    confidence = float(preds[0][class_idx])

    # Generate Grad-CAM
    heatmap_base64, _ = generate_gradcam(img_array, original_img, model, class_idx)

    return JSONResponse({
        "predicted_class": label,
        "confidence": confidence,
        "heatmap": heatmap_base64
    })
    # headers = {
    #     "X-Predicted-Class": label,
    #     "X-Confidence": str(confidence)
    # }

    # return Response(content=heatmap_base64, media_type="image/jpeg", headers=headers)
# -----------------------
# Updated generate_gradcam
# -----------------------
def generate_gradcam(img_array, original_img, model, class_idx, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array])  # wrap in list
        predictions = predictions[0]  # get tensor
        class_idx = min(class_idx, predictions.shape[0] - 1)  # ensure in bounds
        loss = predictions[class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", superimposed)
    heatmap_base64 = base64.b64encode(buffer).decode()

    return heatmap_base64, class_idx
