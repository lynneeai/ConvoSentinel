class PIIDetectorPrompter(object):
    def __init__(self):
        self.template = (
            "Message: {message}\n\n" + 
            "Question: What PII information is requested in this message?"
        )
        
    def generate_prompt(self, message, piis=None):
        prompt = self.template.format(message=message)
        answer = ", ".join(piis) if piis else "None"
        return prompt, answer
    
    def get_response(self, generated_text):
        generated_text = generated_text.replace("<pad>", "")
        generated_text = generated_text.replace("<s> ", "")
        generated_text = generated_text.replace("</s>", "")
        if generated_text == "None":
            return []
        else:
            return generated_text.split(", ")