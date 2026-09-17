# Setup & Ausführung
## 1. Modell herunterladen
Lade das Modell aus dem Kaggle-Notebook https://www.kaggle.com/code/thomasanschtz/cv-project/notebook herunter: /kaggle/working/resnet_augmented.keras und lege es im `models` Verzeichnis ab.


## 2. Umgebung einrichten und Pakete installieren

Führe diesen Befehl aus, um die virtuelle Umgebung zu erstellen, zu aktivieren und alle Abhängigkeiten zu installieren:

### Unter Linux / macOS:

```bash
python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

```

### Unter Windows (PowerShell):

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install -r requirements.txt

```

## 3. Skript ausführen

Starte die Web-App mit:

```bash
cd web-app
streamlit run start.py
```