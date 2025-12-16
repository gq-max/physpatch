import anthropic
import base64
import os

def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
api_base=""
api_key = ""

class ClaudeModel:
    def __init__(self, model_name="claude-3-5-sonnet-20241022", api_key=api_key, base_url=api_base):
        self.model_name = model_name       
        self.client = anthropic.Anthropic(auth_token=api_key, base_url=base_url)
        
    def response(self, image_path, query):
        try:
            base64_image = encode_image(image_path)
 
            if image_path.lower().endswith('.png'):
                media_type = "image/png"
            elif image_path.lower().endswith(('.jpg', '.jpeg')):
                media_type = "image/jpeg"
            elif image_path.lower().endswith('.gif'):
                media_type = "image/gif"
            elif image_path.lower().endswith('.webp'):
                media_type = "image/webp"
            else:
                media_type = "image/png"  # Default fallback
            
            message = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image
                    }
                },
                {
                    "type": "text",
                    "text": query
                }
            ]

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,  # Increased for more detailed responses
                messages=[{"role": "user", "content": message}],
                temperature=0,
            )

            return response.content[0].text
            
        except FileNotFoundError:
            return f"Error: Image file not found at {image_path}"
        except Exception as e:
            return f"Error: {str(e)}"