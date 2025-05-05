from google import genai
from google.genai.types import Content, Part, SafetySetting, GenerateContentConfig

def generate():
    client = genai.Client(
        vertexai=True,
        project="your-project-id",
        location="us-central1"
    )

    model = "gemini-2.5-pro-preview-03-25"

    contents = [
        Content(
            role="user",
            parts=[
                Part.from_text(text="INSERT_INPUT_HERE"),
            ],
        ),
    ]

    generate_content_config = GenerateContentConfig(
        safety_settings=[
            SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ],
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
