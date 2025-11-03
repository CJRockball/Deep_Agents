# MongoDB client - to be implemented
# services/agent-core/src/db/mongo_client.py

import logging
from pymongo import MongoClient
from config import MONGO_URI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize client
client = MongoClient(MONGO_URI)
db = client["agent_memory"]

logger.info("Connected to MongoDB at %s", MONGO_URI)
