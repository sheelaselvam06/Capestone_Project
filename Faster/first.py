from fastapi import FastAPI

appn =FastAPI()
@appn.get("/ABCD")
def read_root():
    return {"first": "one", "second": "two"}

@appn.post("/status")
def status():
   return {"status": "running"}

@appn.get("/items/{item_id}")
def read_item(item_id):
    return {"item_id":item_id}

class item(BaseModel):
    name: str 
    email:str 
@appn.post("/create-items/") 
def create_item(item: Item):
    return {"name": item.name, "email": item.email}