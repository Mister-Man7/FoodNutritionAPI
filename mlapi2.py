from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import PIL.Image
import io
import pandas as pd
import os
import gdown

url = "https://drive.google.com/drive/folders/1nXvQvjzrVDBO8-1GYQPVM5F0poIn_Dth?usp=sharing"
gdown.download_folder(url, output=".", quiet=True)

# Matiin log kalo run di Windows doang
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI()

model = load_model("foodnutritio_baru.h5")

CLASS_NAMES = ['Ayam Bakar', 'Ayam Geprek', 'Ayam Goreng', 'Ayam Tepung', 'Bakso', 
               'Chicken Katsu', 'Donat', 'Gado-Gado', 'Kopi', 'Mie Ayam', 'Mie Instan', 
               'Nasi Goreng', 'Pecel Lele', 'Random Image', 'Rendang', 'Sate Ayam', 
               'Sop', 'Soto', 'Telur Balado']

# Muat file CSV
nutrisi_csv = pd.read_csv('nutrisi_origin.csv')

KOLOM_NAMA = nutrisi_csv.columns[1]

class FoodRequest(BaseModel):
    nama_makanan: str

def get_nutrition_data(food_name: str):
    pencarian = nutrisi_csv[nutrisi_csv[KOLOM_NAMA].str.lower() == food_name.lower()]
    
    kalori, protein, lemak, karbo = 0.0, 0.0, 0.0, 0.0
    kategori = "Tidak Diketahui"

    if not pencarian.empty:
        detail_nutrisi = pencarian.iloc[0]
        
        kal_teks = str(detail_nutrisi.get('Kalori', '0')).lower().replace(',', '.').replace('g', '').strip()
        pro_teks = str(detail_nutrisi.get('Protein', '0')).lower().replace(',', '.').replace('g', '').strip()
        lem_teks = str(detail_nutrisi.get('Lemak', '0')).lower().replace(',', '.').replace('g', '').strip()
        karbo_teks = str(detail_nutrisi.get('Karbohidrat', '0')).lower().replace(',', '.').replace('g', '').strip()
        
        if 'Kategori' in detail_nutrisi:
            kategori = str(detail_nutrisi['Kategori'])

        try: kalori = float(kal_teks)
        except: kalori = 0.0
        
        try: protein = float(pro_teks)
        except: protein = 0.0
        
        try: lemak = float(lem_teks)
        except: lemak = 0.0
        
        try: karbo = float(karbo_teks)
        except: karbo = 0.0

    return {
        "kalori": kalori,
        "protein": protein,
        "lemak": lemak,
        "karbohidrat": karbo,
        "kategori": kategori
    }

@app.get('/')
async def hello():
    return {"message": "Food Nutrition API Ready"}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = PIL.Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((128, 128))

    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    index = np.argmax(preds[0])
    predicted_label = CLASS_NAMES[index]
    confidence = float(np.max(preds[0]))

    data_nutrisi = get_nutrition_data(predicted_label)

    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "nutrisi": data_nutrisi
    }

@app.post('/predict_text')
async def predict_text(nama_makanan: str = Form(...)):
    
    data_nutrisi = get_nutrition_data(nama_makanan)

    return {
        "prediction": nama_makanan,
        "confidence": 1.0,
        "nutrisi": data_nutrisi

    }
