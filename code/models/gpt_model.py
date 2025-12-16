import openai
import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
class Gpt_model:
    def __init__(self, model_name):
        self.model_name = model_name
        openai.api_key = ""
        openai.api_base = ""
        
    def response(self, image_url, query):
        base64_image = encode_image(image_url)
        message = [{"role": "user", "content": [
                {"type": "text", "text": f"{query}"},
                {"type": "image_url", "image_url": {
                    "url":  f"data:image/png;base64,{base64_image}"}
                }
            ]}]
        
        response = openai.ChatCompletion.create(
                    model=self.model_name, 
                    messages=message,
                    temperature=0,
                    # max_tokens=1024,
                    max_completion_tokens=1024,
                    # api_key=api_key,
                )
        gen = response['choices'][0]['message']['content']
        return gen