from typing import Union

from fastapi import FastAPI
from .utils.logger import logger
app = FastAPI()


@app.get("/")
def read_root():
    print("Hello World bonito")
    return {"Hello": "World bonito"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    logger.info(f"Endpoint /items/{item_id} was called")
    return {"item_id": item_id, "q": q}