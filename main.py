import asyncio
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import angles
import modes
import numpy as np
import cv2 as cv
import os
import uuid
import aiofiles
from imgproc import resize

app = FastAPI()

UPLOAD_DIRECTORY = "upload"
OUTPUT_DIRECTORY = "output"
PROCESSING_DIRECTORY = "processing"
TEMPLATES_DIRECTORY = "templates"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSING_DIRECTORY, exist_ok=True)

async def timeout_delete(file_id: str, timeout: int = 60):
    # cleans up the directories 

    await asyncio.sleep(timeout)
    upload_location = os.path.join(UPLOAD_DIRECTORY, f"{file_id}_uploaded.png")
    output_location = os.path.join(OUTPUT_DIRECTORY, f"{file_id}_output.png")

    sobel_location = os.path.join(PROCESSING_DIRECTORY, f"{file_id}_sobel.png")
    threshed_location = os.path.join(PROCESSING_DIRECTORY, f"{file_id}_threshed.png")
    blobs_location = os.path.join(PROCESSING_DIRECTORY, f"{file_id}_blobs.png")
    contour_location = os.path.join(PROCESSING_DIRECTORY, f"{file_id}_contour.png")
    filtered_location = os.path.join(PROCESSING_DIRECTORY, f"{file_id}_filtered.png")
    
    if os.path.exists(upload_location):
        os.remove(upload_location)
    if os.path.exists(output_location):
        os.remove(output_location)
    if os.path.exists(sobel_location):
        os.remove(sobel_location)
    if os.path.exists(threshed_location):
        os.remove(threshed_location)
    if os.path.exists(blobs_location):
        os.remove(blobs_location)
    if os.path.exists(contour_location):
        os.remove(contour_location)
    if os.path.exists(filtered_location):
        os.remove(filtered_location)

@app.post("/uploadimg")
async def upload_img(file: UploadFile = File(...),
                    constellation: str = Form(...),
                    identification_mode: str = Form(...),
                    threshold: int = Form(None),
                    filter_mode: str = Form(...),
                    ksize: int = Form(None)):
    file_content = await file.read()
    content_array = np.frombuffer(file_content, np.uint8)
    image = cv.imdecode(content_array, cv.IMREAD_GRAYSCALE)
    if image is None:
        print("could not decode image")
        return {"status": "error", "message": "Could not decode image"}
    image = resize(image, max_dimension=700)


    _, buffer = cv.imencode('.png', image)
    image = buffer.tobytes()

    unique_id = str(uuid.uuid4())
    upload_location = os.path.join(UPLOAD_DIRECTORY, f"{unique_id}_uploaded.png")
    async with aiofiles.open(upload_location, "wb") as upload_file:
        await upload_file.write(image)

    if threshold is not None:
        if threshold < 0 or threshold > 255:
            print("invalid threshold value")
            return {"status": "error", "message": "invalid threshold value"}
    if ksize is not None:
        if ksize < 2 or ksize % 2 == 0:
            print("invalid kernel size")
            return {"status": "error", "message": "invalid kernel size"}

    try:
        identification_mode = modes.Identification_mode(identification_mode)
        filter_mode = modes.Filter_mode(filter_mode)
    except:
        print("invalid identification or filter modes")
        return {"status": "error", "message": "invalid identification or filter modes"}

    closest_constellation = angles.process(upload_location, constellation, identification_mode, threshold, filter_mode, ksize, unique_id)

    asyncio.create_task(timeout_delete(unique_id))

    return {"filename": file.filename, "file_id": unique_id, "constellation": closest_constellation}


@app.get("/")
async def get_homepage():
    return HTMLResponse(open("home.html").read())


@app.get("/output")
async def get_output(file_id: str):
    file_location = os.path.join(OUTPUT_DIRECTORY, f"{file_id}_output.png")
    if os.path.exists(file_location):
        return FileResponse(path=file_location, filename="output.png")
    else:
        return {"error": "File not found"}
    
@app.get("/filtered")
async def get_filtered(file_id: str):
    file_location = os.path.join(PROCESSING_DIRECTORY, f"{file_id}_filtered.png")
    if os.path.exists(file_location):
        return FileResponse(path=file_location, filename="filtered.png")
    else:
        return {"error": "File not found"}
    
@app.get("/blobs")
async def get_blobs(file_id: str):
    file_location = os.path.join(PROCESSING_DIRECTORY, f"{file_id}_blobs.png")
    if os.path.exists(file_location):
        return FileResponse(path=file_location, filename="blobs.png")
    else:
        return {"error": "File not found"}

@app.get("/contour")
async def get_contoured(file_id: str):
    file_location = os.path.join(PROCESSING_DIRECTORY, f"{file_id}_contour.png")
    if os.path.exists(file_location):
        return FileResponse(path=file_location, filename="contour.png")
    else:
        return {"error": "File not found"}

@app.get("/threshed")
async def get_threshed(file_id: str):
    file_location = os.path.join(PROCESSING_DIRECTORY, f"{file_id}_threshed.png")
    if os.path.exists(file_location):
        return FileResponse(path=file_location, filename="threshed.png")
    else:
        return {"error": "File not found"}

@app.get("/identifications")
async def get_identifications():
    return modes.get_identification_modes()

@app.get("/filters")
async def get_identifications():
    return modes.get_filter_modes()

@app.get("/constellations")
async def get_constellations():
    constellList = ["none"]
    for filename in os.listdir(TEMPLATES_DIRECTORY):
        constellation = filename[:-4]
        constellList.append(constellation)

    return constellList