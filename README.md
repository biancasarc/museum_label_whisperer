# Installation

Follow the steps below to install all required software before running the Museum Label Whisperer.

## 1. Install Docker Desktop

Download and install Docker Desktop from:

https://www.docker.com/products/docker-desktop/

After installation, make sure Docker Desktop is running before continuing.

---

## 2. Install CVAT

Clone the CVAT repository:

```bash
git clone https://github.com/cvat-ai/cvat
cd cvat
```

Start CVAT:

```bash
docker compose up -d
```

Create an administrator account:

```bash
docker exec -it cvat_server bash -c "python manage.py createsuperuser"
```

You will be prompted to create a **username** and **password**.

> **Important:** Remember these credentials—you will use them to log into CVAT from the Museum Label Whisperer application.

---

## 3. Clone the Museum Label Whisperer repository

```bash
git clone https://github.com/biancasarc/museum_label_whisperer
cd museum_label_whisperer
```

---

## 4. Create a Python environment (recommended)

It is strongly recommended to install the Python dependencies in a virtual environment.

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```
