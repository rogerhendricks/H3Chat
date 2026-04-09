Run command:
```
uvicorn main:app --reload --port 8000
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
url: http://localhost:8000/docs
Ollama url: http://localhost:11434

python ingest-v2.py "/mnt/HeartRhythm/Singles/Electrocardiographic interpretation of pacemaker algorithms enabling minimal ventricular pacing.pdf" \
  --title "Electrocardiographic interpretation of pacemaker algorithms enabling minimal ventricular pacing" \
  --author "" \
  --version "" \
  --publication-date "2024-05-01T00:00:00Z"\
  --description "Electrocardiographic interpretation of pacemaker algorithms enabling minimal ventricular pacing" \
  --document-type "research" \
  --source-uri ""
