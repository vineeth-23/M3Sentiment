import json

class Config:
    def __init__(self, path="configs/default.json"):
        with open(path, "r") as f:
            self.cfg = json.load(f)
    def __getattr__(self, name):
        return self.cfg[name]
