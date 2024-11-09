import os
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from inference import predict_image

app = FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the class of an uploaded image.

        Args:
            file (UploadFile): The image file to be classified.

        Returns:
            JSONResponse: A JSON object containing the prediction result or an error message.

        Raises:
            HTTPException: If there's an error processing the file or making the prediction.
    """
#    print("Server Started at PORT:8000")
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file to the temporary directory
            temp_file_path = os.path.join(temp_dir, file.filename)
            with open(temp_file_path, "wb") as temp_file:
                contents = await file.read()
                temp_file.write(contents)

            # Make prediction using the temporary file path
            predicted_class, prob = predict_image(temp_file_path)

        # Return prediction as JSON
        return JSONResponse(content={"prediction": predicted_class, "probability": prob})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)