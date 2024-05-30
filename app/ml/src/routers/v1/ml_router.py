import io
from typing import List

from fastapi import APIRouter, Depends, File
from starlette.responses import Response
from ml.src.services.ml_service import ChurnMLService
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import pandas as pd


ml_router = APIRouter(
    tags=['ML'],
    prefix='/ml'
)

ml_service = ChurnMLService()

@ml_router.post("/predict")
def get_predictions(file: bytes = File(...)):
    df = pd.read_csv(io.BytesIO(file))
    df = ml_service.preprocess(df)
    client_ids, preds = ml_service.predict(df)


    content = pd.DataFrame(data={'client_id': client_ids, 'pred': preds}).to_csv()
    
    return Response(content=content, media_type="text/csv")


# @ml_router.post("/download")
# def get_segmentation_map():
    

#     return Response(content=final_data.to_csv(), media_type="text/csv")



