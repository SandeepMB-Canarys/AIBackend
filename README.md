# AI Document Analysis Backend

A Django REST API backend for analyzing documents using AI. The system supports document uploads, AI-based analysis, and report generation.

## Features

- Document upload and management
- AI-powered document analysis
- Report generation in Excel format
- RESTful API with Swagger documentation
- Support for multiple AI providers and LLM models

## Prerequisites

- Python 3.12 or higher
- Django 4.x
- TensorFlow (for AI processing)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd fileupload_project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Apply database migrations:
```bash
python manage.py migrate
```

## Running the Server

1. Start the development server:
```bash
python manage.py runserver
```

2. Access the application:
- Main application: http://localhost:8000
- API documentation: http://localhost:8000/swagger/

## API Endpoints

- `POST /`: Upload and analyze documents
- `GET /analyze/<id>/`: Get analysis results
- `GET /media/report/<filename>`: Download generated reports

## Environment Variables

Make sure to set up the following environment variables:
- `TF_ENABLE_ONEDNN_OPTS`: Set to 0 to disable TensorFlow oneDNN optimizations if needed
