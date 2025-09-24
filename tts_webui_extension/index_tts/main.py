import gradio as gr
from .ui import index_tts_ui

def extension__tts_generation_webui():
    index_tts_ui()
    
    return {
        "package_name": "tts_webui_extension.index_tts",
        "name": "Index tts",
        "requirements": "git+https://github.com/rsxdalv/tts_webui_extension.index_tts@main",
        "description": "A template extension for TTS Generation WebUI",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "Your Name",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/rsxdalv/tts_webui_extension.index_tts",
        "extension_website": "https://github.com/rsxdalv/tts_webui_extension.index_tts",
        "extension_platform_version": "0.0.1",
    }


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        with gr.Tab("Index tts", id="index_tts"):
            index_tts_ui()

    demo.launch(
        server_port=7772,  # Change this port if needed
    )
