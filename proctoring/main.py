### main.py
from fastapi import FastAPI, Request
from .database import engine, Base
from .routers import auth, users, proctor
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import models so they get registered
from . import models

# Create tables
Base.metadata.create_all(bind=engine)

# Ensure directories are correct
STATIC_DIR = os.path.join("proctoring", "static")
TEMPLATES_DIR = os.path.join("proctoring", "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
def show_login(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/quiz", response_class=HTMLResponse)
async def read_quiz(request: Request):
    return templates.TemplateResponse("quiz.html", {"request": request})

@app.get("/users/login")
def login_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(proctor.router)