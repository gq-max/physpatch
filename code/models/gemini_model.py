import openai
import base64

api_key = ""
base_url = ""
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class GeminiModel:
    def __init__(self, model_name='gemini-2.0-flash-exp'):
        self.model_name = model_name
        openai.api_key = api_key
        openai.api_base = base_url

    def response(self, image_path, query):
        try:
            base64_image = encode_image(image_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }]

            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=4096
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"
