from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import os

load_dotenv()

def db_connect():
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine