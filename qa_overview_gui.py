import os
import time
import json
import streamlit as st
import requests as req
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

load_dotenv()
DETA_URL = os.getenv('DETA_URL')
BONSAI_URL = os.getenv('BONSAI_URL')

response = req.request('GET', DETA_URL)
resp_json = response.content.decode("utf-8")

st.title('DevJam QA overview')

st.write(
    f'{resp_json}'
)