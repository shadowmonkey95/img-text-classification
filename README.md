# Image-text classifications  

## Create venv and install libs  

```
python -m venv venv
source venv/bin/activate
pip install pip-tools
pip-compile requirements.in
pip-sync
```

## Data Structure  
data/  
├── csv/  
│   ├── trademark_codes.csv  
│   ├── trademark_updated.csv  
│   └── trademark.csv  
└── img/  
    └──[image files]  
