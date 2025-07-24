# 1. Basis-Image mit Python
FROM python:3.11-slim

# 2. Arbeitsverzeichnis
WORKDIR /app

# pip aktualisieren, bevor requirements installiert werden
RUN python -m pip install --upgrade pip

# 3. Abhängigkeiten kopieren und installieren
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# 4. Anwendungscode kopieren
COPY . .

# 6. Port freigeben
EXPOSE 5000

# 7. Startbefehl
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:5000", "app:app"]
