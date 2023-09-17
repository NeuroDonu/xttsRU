import gradio as gr
from TTS.api import TTS


tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")
tts.to("cuda")

def predict(prompt, language, audio_file_pth, agree):
    if agree == True:
        tts.tts_to_file(
            text=prompt,
            file_path="output.wav",
            speaker_wav=audio_file_pth,
            language=language,
        )

        return (
            gr.make_waveform(
                audio="output.wav",
            ),
            "output.wav",
        )
    else:
        gr.Warning("Пожалуйста примите наши правила использования!")

title = "XTTS RU by NeuroDonu"

description = """
Полная документация проекта на <a href='https://huggingface.co/coqui/XTTS-v1'>HuggingFace</a>
"""

article = """
<div style='margin:20px auto;'>
<p>Используя эту программу, вы автоматически соглашаетесь с <a href='https://coqui.ai/cpml'>нашей политикой</a></p>
</div>
"""

examples = [
    [
        "Один раз я пошел в лес за грибами.",
        "ru",
        "examples/male.wav",
        True,
    ],
]

gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(
            label="Text Prompt",
            info="One or two sentences at a time is better",
            value="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
        ),
        gr.Dropdown(
            label="Language",
            info="Select an output language for the synthesised speech",
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cz",
                "ar",
                "zh",
            ],
            max_choices=1,
            value="en",
        ),
        gr.Audio(
            label="Reference Audio",
            info="Click on the ✎ button to upload your own target speaker audio",
            type="filepath",
            value="examples/female.wav",
        ),
        gr.Checkbox(
            label="Agree",
            value=False,
            info="I agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml",
        ),
    ],
    outputs=[
        gr.Video(label="Waveform Visual"),
        gr.Audio(label="Synthesised Audio"),
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
).launch(share=True)
