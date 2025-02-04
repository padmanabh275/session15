import gradio as gr
import requests
import json

MODEL_API = "http://model-service:5000/generate"

def generate_text(prompt, max_length=100, temperature=0.7, top_p=0.9, top_k=50):
    try:
        # Log the request
        print(f"\nSending request to model service:")
        print(f"Prompt: {prompt}")
        print(f"Parameters: length={max_length}, temp={temperature}, top_p={top_p}, top_k={top_k}")
        
        response = requests.post(MODEL_API, json={
            'prompt': prompt,
            'max_length': max_length,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k
        })
        response.raise_for_status()
        
        # Log the response
        result = response.json()['generated_text']
        print(f"\nReceived response from model:")
        print(f"Generated text: {result}")
        
        return result
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"\nError occurred: {error_msg}")
        return error_msg

# Create Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=10, maximum=200, value=100, step=1, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top-p"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="SmolLM2 Text Generation",
    description="Enter a prompt and adjust generation parameters to create text with SmolLM2"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 