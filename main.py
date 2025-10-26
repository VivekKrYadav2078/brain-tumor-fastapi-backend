from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io
# from pydantic import BaseModel

app = FastAPI()

@app.get("/{name}")
def read_root(name:str):
    return {"message": f"Hello from {name}"}

@app.post("/predict")
async def predicted(file:UploadFile=File(...)):
    image_bytes=await file.read()
# Convert bytes to a buffer
    buf = io.BytesIO(image_bytes)
    buf.seek(0)
  # Return the image directly
    return StreamingResponse(buf, media_type="image/jpeg")

# class User(BaseModel):
#     name:str
#     age:int
# @app.post("/")
# def read_req(user:User):
#     return{"massage":f"Hello from {user.name} and age is {user.age}"}
