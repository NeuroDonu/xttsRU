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
            label="Ваш промпт",
            info="Лучше одно или два предложения за раз.",
            value="Мне потребовалось довольно много времени, чтобы обрести голос, и теперь, когда он у меня есть, я не собираюсь молчать.",
        ),
        gr.Dropdown(
            label="Язык",
            info="Выберите один из языков ниже:",
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
            value="ru",
        ),
        gr.Audio(
            label="Эталонное аудио",
            info="Нажмите кнопку ✎, чтобы загрузить собственный звук диктора.",
            type="file",
            value="examples/female.wav",
        ),
        gr.Checkbox(
            label="Согласен",
            value=True,
            info="Я согласен с условиями лицензии публичной модели Coqui на странице <a href='https://coqui.ai/cpml'>https://coqui.ai/cpml</a>",
        ),
    ],
    outputs=[
        gr.Video(label="Waveform визуал"),
        gr.Audio(label="Синтезированное аудио."),
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
).launch(share=True)
