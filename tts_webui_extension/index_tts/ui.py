# Adapted from IndexTTS repository, Licensed under Bilibili license.
import html
import os
import time
import gradio as gr
import pandas as pd

from tts_webui.utils.manage_model_state import is_model_loaded
from tts_webui.utils.list_dir_models import unload_model_button

from .loader.get_model import get_model

MODE = 'local'
# Supported languages
LANGUAGES = {
    "Chinese": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [
    "Same as speaker reference",
    "Use emotion reference audio",
    "Use emotion vectors",
    "Use emotion description text",
]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]  # skip experimental features
CONFIG_MAX_TEXT_TOKENS = 600
CONFIG_MAX_MEL_TOKENS = 1815

voices_prompts_dir = os.path.join("voices", "index-tts")
os.makedirs(voices_prompts_dir, exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70


def tts(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment=120,
                *args, progress=gr.Progress(), verbose=None):
    outputs_dir = os.path.join("outputs-rvc", "index-tts")
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, f"spk_{int(time.time())}.wav")

    model = get_model("IndexTeam/IndexTTS-2")
    model.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = model.normalize_emo_vec(vec, apply_bias=True)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    output = model.infer(spk_audio_prompt=prompt, text=text,
                         output_path=output_path,
                         emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                         emo_vector=vec,
                         use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                         verbose=verbose,
                         max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                         **kwargs)
    return gr.update(value=output,visible=True)

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message('Note: This feature is experimental and results may be unstable. We are continuously improving it.')

def index_tts_ui():
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    with gr.Tab("Audio Generation"):
        with gr.Row():
            with gr.Column():
                try:
                    prompt_list = sorted(os.listdir(voices_prompts_dir))
                except Exception:
                    prompt_list = []
                prompt_dropdown = gr.Dropdown(label="Saved Audios", choices=prompt_list)
                prompt_audio = gr.Audio(label="Speaker reference audio", key="prompt_audio",
                                        sources=["upload", "microphone"], type="filepath")
                unload_model_button("index-tts")
            with gr.Column():
                input_text_single = gr.TextArea(label="Text",key="input_text_single", placeholder="Enter target text")
                gen_button = gr.Button("Generate", key="gen_button", variant="primary")
            output_audio = gr.Audio(label="Result", visible=True,key="output_audio")

    # Handler: when user selects a shipped prompt, update the `prompt_audio` file path
    def on_prompt_select(selected):
        if not selected:
            return gr.update(value=None)
        selected_path = os.path.join(voices_prompts_dir, selected)
        # Ensure file exists; if not, return no-op
        if not os.path.exists(selected_path):
            return gr.update(value=None)
        return gr.update(value=selected_path)

    # Wire the dropdown change to update the prompt_audio component
    prompt_dropdown.change(on_prompt_select, inputs=[prompt_dropdown], outputs=[prompt_audio])

    experimental_checkbox = gr.Checkbox(label="Show experimental features", value=False)

    with gr.Accordion("Settings"):
        # emotion control options
        with gr.Row():
            emo_control_method = gr.Radio(
                choices=EMO_CHOICES_OFFICIAL,
                type="index",
                value=EMO_CHOICES_OFFICIAL[0], label="Emotion control method")
            # we MUST have an extra, INVISIBLE list of *all* emotion control
            # methods so that gr.Dataset() can fetch ALL control mode labels!
            # otherwise, the gr.Dataset()'s experimental labels would be empty!
            emo_control_method_all = gr.Radio(
                choices=EMO_CHOICES_ALL,
                type="index",
                value=EMO_CHOICES_ALL[0], label="Emotion control method",
                visible=False)  # do not render

        # emotion reference audio group
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label="Upload emotion reference audio", type="filepath")

        # emotion randomization
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(label="Randomize emotion", value=False)

        # emotion vector control group
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label="Joy", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec2 = gr.Slider(label="Anger", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec3 = gr.Slider(label="Sadness", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec4 = gr.Slider(label="Fear", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label="Disgust", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec6 = gr.Slider(label="Low", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec7 = gr.Slider(label="Surprise", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec8 = gr.Slider(label="Calm", minimum=0.0, maximum=1.0, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            create_experimental_warning_message()
            with gr.Row():
                emo_text = gr.Textbox(label="Emotion description text",
                                        placeholder="Enter an emotion description (or leave empty to use the target text)",
                                        value="",
                                        info="E.g.: feeling wronged, danger is approaching")

        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(label="Emotion weight", minimum=0.0, maximum=1.0, value=0.65, step=0.01)

        with gr.Accordion("Advanced generation parameters", open=False, visible=True) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**GPT2 sampling settings** _Parameters affect audio diversity and generation speed; see_ [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info="Whether to perform sampling")
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=CONFIG_MAX_MEL_TOKENS, step=10, info="Maximum number of generated tokens; too small may truncate audio", key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown('**Segmentation settings** _Parameters may affect audio quality and generation speed_')
                    with gr.Row():
                        initial_value = max(20, min(CONFIG_MAX_TEXT_TOKENS, 120))
                        max_text_tokens_per_segment = gr.Slider(
                            label="Max tokens per segment", value=initial_value, minimum=20, maximum=CONFIG_MAX_TEXT_TOKENS, step=2, key="max_text_tokens_per_segment",
                            info="Recommended between 80 and 200. Larger values produce longer segments; smaller values produce shorter segments. Too small or too large may reduce audio quality",
                        )
                    with gr.Accordion("Preview segments", open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=["Index", "Segment content", "Token count"],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]

        # Examples removed: no dataset or example click handlers

    def on_input_text_change(text, max_text_tokens_per_segment):
        # Provide a preview of segments. If the model/tokenizer hasn't been loaded,
        # use a lightweight fallback tokenizer (split on whitespace) to avoid
        # instantiating the heavy model just to preview segmentation.
        if text and len(text) > 0:
            if is_model_loaded("index-tts"):
                _model = get_model("IndexTeam/IndexTTS-2")
                text_tokens_list = _model.tokenizer.tokenize(text)
                segments = _model.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
                data = []
                for i, s in enumerate(segments):
                    segment_str = ''.join(s)
                    tokens_count = len(s)
                    data.append([i, segment_str, tokens_count])
            else:
                # naive fallback: split into words and chunk by max_text_tokens_per_segment
                tokens = text.split()
                seg_size = max(1, int(max_text_tokens_per_segment))
                data = []
                for i in range(0, len(tokens), seg_size):
                    s = tokens[i:i+seg_size]
                    segment_str = ' '.join(s)
                    tokens_count = len(s)
                    data.append([i//seg_size, segment_str, tokens_count])

            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=["Index", "Segment content", "Token count"])
            return {
                segments_preview: gr.update(value=df),
            }

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                    emotion_randomize_group,
                    emotion_vector_group,
                    emo_text_group,
                    emo_weight_group]
    )

    def on_experimental_change(is_experimental, current_mode_index):
        # 切换情感控制选项
        new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
        # if their current mode selection doesn't exist in new choices, reset to 0.
        # we don't verify that OLD index means the same in NEW list, since we KNOW it does.
        new_index = current_mode_index if current_mode_index < len(new_choices) else 0

        return gr.Radio(choices=new_choices, value=new_choices[new_index])

    experimental_checkbox.change(
        on_experimental_change,
        inputs=[experimental_checkbox, emo_control_method],
        outputs=[emo_control_method]
    )

    input_text_single.change(
        on_input_text_change,
        concurrency_limit=1,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        concurrency_limit=1,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    gen_button.click(tts,
                        inputs=[emo_control_method,prompt_audio, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                                emo_text,emo_random,
                                max_text_tokens_per_segment,
                                *advanced_params,
                        ],
                        outputs=[output_audio])

if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()  # avoid multiple launches in notebook
    with gr.Blocks(title="IndexTTS Demo") as demo:
        index_tts_ui()
    demo.launch(server_port=8002)
