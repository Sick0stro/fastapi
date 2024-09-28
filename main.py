from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
import os
from PIL import Image
from deepface import DeepFace
import pandas as pd
import logging
import re
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI application
app = FastAPI()

# Define the origins that are allowed to make requests to this API
origins = [
    "3ain-sigma.vercel.app", "http://localhost:3000",  # Allow your frontend's origin
    # Add other origins if needed
]

# Apply the CORS middleware to your FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Allow specific origins
    allow_credentials=True,           # Allow cookies or credentials
    allow_methods=["*"],              # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],              # Allow all headers
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ImageData(BaseModel):
    image: str

class ImageAnalyzer:
    def __init__(self, image_data):
        self.image_data = image_data

    def clean_base64(self):
        # Remove the Data URI scheme part if present
        if re.match(r'^data:image/.+;base64,', self.image_data):
            return re.sub(r'^data:image/.+;base64,', '', self.image_data)
        return self.image_data

    def analyze_photo(self):
        temp_image_path = 'temp_image.jpg'
        try:
            # Clean the base64 image string
            image_data_cleaned = self.clean_base64()

            # Convert base64 image to PIL Image
            try:
                image_bytes = base64.b64decode(image_data_cleaned)
                image = Image.open(io.BytesIO(image_bytes))
                image.save(temp_image_path)
                logger.debug(f"Saved temporary image at: {temp_image_path}")
            except Exception as e:
                logger.error(f"Error decoding or saving image: {e}")
                return {"error": "Invalid image data."}

            # Analyze the photo using DeepFace
            try:
                objs = DeepFace.analyze(img_path=temp_image_path, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
                logger.debug(f"DeepFace analysis result: {objs}")
            except Exception as e:
                logger.error(f"Error during DeepFace analysis: {e}")
                return {"error": "Error analyzing image with DeepFace."}

            if objs and isinstance(objs, list) and len(objs) > 0:
                analysis = objs[0]
                dominant_emotion = analysis.get('dominant_emotion', 'N/A')
                dominant_race = analysis.get('dominant_race', 'N/A')
                dominant_gender = analysis.get('dominant_gender', 'N/A')
                age = analysis.get('age', 'N/A')
                
                # Perform face recognition
                try:
                    dfs = DeepFace.find(img_path=temp_image_path, db_path="./user/database", model_name="ArcFace", enforce_detection=False)
                    logger.debug(f"Face recognition result: {dfs}")

                    if isinstance(dfs, list) and len(dfs) > 0:
                        df_list = [pd.DataFrame(d) for d in dfs if isinstance(d, pd.DataFrame)]
                        
                        if df_list:
                            df = pd.concat(df_list, ignore_index=True)
                            
                            if not df.empty and 'distance' in df.columns:
                                min_distance_row = df.loc[df['distance'].idxmin()]
                                identity_value = min_distance_row['identity']
                                folder_name = os.path.basename(os.path.dirname(identity_value))
                                
                                with open(identity_value, "rb") as image_file:
                                    base64_encoded_data = base64.b64encode(image_file.read())
                                    base64_string = base64_encoded_data.decode('utf-8')
                                    
                                return {
                                    "dominant_emotion": dominant_emotion,
                                    "dominant_race": dominant_race,
                                    "dominant_gender": dominant_gender,
                                    "predicted_age": age,
                                    "identity_image": base64_string,
                                    "folder_name": folder_name,
                                    "scanned_image": image_data_cleaned
                                }
                            else:
                                return {"error": "No data available in the DataFrame or 'distance' column missing."}
                        else:
                            return {"error": "No valid DataFrames found in the list."}
                    else:
                        return {"error": "No matches found or data is not in list format."}
                except Exception as e:
                    logger.error(f"Error during face recognition: {e}")
                    return {"error": "Error performing face recognition."}
            else:
                return {"error": "No analysis results found."}

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"error": str(e)}
        
        finally:
            # Clean up temporary image file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

@app.post("/analyze")
async def analyze_image(image_data: ImageData):
    try:
        analyzer = ImageAnalyzer(image_data.image)
        result = analyzer.analyze_photo()
        
        # Return the analysis result
        return result
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Use this if running locally or in Railway
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="debug")
