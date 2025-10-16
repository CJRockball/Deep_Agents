// Initialize MongoDB collections
db = db.getSiblingDB('agent_memory');
db.createCollection('documents');
