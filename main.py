import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
language_id = EncoderClassifier.from_hparams(source="lang-id-voxlingua107-ecapa", savedir="tmp")

def lang_detect(audio_path:str):
    signal = language_id.load_audio(audio_path)
    prediction =  language_id.classify_batch(signal)

    lang = prediction[3][0]
    return lang

audio = "audio_en.wav"
my_func = lang_detect(audio)
print(f"Audio tili : {my_func}")