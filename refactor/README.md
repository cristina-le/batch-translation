# Batch Translation API

A FastAPI-based asynchronous API for translating text and files from Japanese to English using the OpenRouter API.

## Features

-   **Asynchronous Task Processing**: Translation jobs are handled in the background, ensuring the API remains responsive.
-   **Text and File Translation**: Supports direct text input (`translate_context`) and translation from a file URL (`translate_file`).
-   **Job Status Tracking**: Provides an endpoint to retrieve the status and results of a translation job.
-   **Dockerized**: Includes a `Dockerfile` for easy containerization and deployment.

## Installation

### Prerequisites

-   Python 3.8+
-   An OpenRouter API Key

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/cristina-le/batch-translation.git
    cd batch-translation
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```
    OPENROUTER_API_KEY="your_openrouter_api_key_here"
    ```

## Usage

### Running the Server

Start the development server using Uvicorn:
```bash
uvicorn app.main:app --reload --port 8000
```
The API documentation will be available at `http://0.0.0.0:8000/docs`.

### API Documentation

The primary endpoint for all operations is `POST /api/task-execution`. The action to be performed is determined by the `task` field in the JSON payload.

---

#### 1. Translate Raw Text

-   `task`: `translate_context`
-   **Description**: Translates a block of Japanese text provided in the `context` field.
-   **Context**: A string containing the Japanese text to be translated.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/task-execution" \
-H "Content-Type: application/json" \
-d '{
    "task": "translate_context",
    "context": "こんにちは世界\nこれはテストです"
}'
```

**Success Response (Job Started):**
```json
{
    "status": "Success",
    "data": [
        {
            "status": "started",
            "job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef"
        }
    ]
}
```

---

#### 2. Translate from a File

-   `task`: `translate_file`
-   **Description**: Downloads and translates a text file from a given URL.
-   **Context**: A string containing the URL of the raw text file.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/task-execution" \
-H "Content-Type: application/json" \
-d '{
    "task": "translate_file",
    "context": "https://raw.githubusercontent.com/someuser/some-repo/main/sample.txt"
}'
```

**Success Response (Job Started):**
```json
{
    "status": "Success",
    "data": [
        {
            "status": "started",
            "job_id": "b2c3d4e5-f6a7-8901-2345-67890abcdef1"
        }
    ]
}
```

---

#### 3. Get Job Results

-   `task`: `get_job`
-   **Description**: Retrieves the results of a completed translation job using its `job_id`.
-   **Context**: A string containing the `job_id` returned from a `translate_context` or `translate_file` request.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/task-execution" \
-H "Content-Type: application/json" \
-d '{
    "task": "get_job",
    "context": "a1b2c3d4-e5f6-7890-1234-567890abcdef"
}'
```

**Success Response (Job Found):**
```json
{
    "status": "Success",
    "data": [
        {
            "job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
            "completed_at": "2025-07-13T11:45:00.123456",
            "content": [
                {
                    "line": 0,
                    "japanese": "こんにちは世界",
                    "english": "Hello World"
                },
                {
                    "line": 1,
                    "japanese": "これはテストです",
                    "english": "This is a test"
                }
            ]
        }
    ]
}
```

**Error Response (Job Not Found):**
```json
{
    "status": "Success",
    "data": [
        {
            "error": "Job ID a1b2c3d4-e5f6-7890-1234-567890abcdef not found"
        }
    ]
}
```

## Docker

You can also build and run the application using Docker.

1.  **Build the Docker image:**
    ```bash
    docker build -t batch-translation-api .
    ```

2.  **Run the Docker container:**
    Make sure to pass the `OPENROUTER_API_KEY` as an environment variable.
    ```bash
    docker run -d -p 8000:8000 --env OPENROUTER_API_KEY="your_operouter_api_key_here" --name translation-api batch-translation-api
