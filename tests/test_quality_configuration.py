"""Pin effective supported decoding defaults; no speech-quality or GPU claim."""
from dataclasses import replace
import inspect
from types import SimpleNamespace

from faster_whisper import WhisperModel
from faster_whisper.feature_extractor import FeatureExtractor

from transcribe_translate import asr


def test_layout_keeps_effective_quality_baseline(config):
    calls=[]
    defaults=inspect.signature(WhisperModel.transcribe).parameters
    class Model:
        model=SimpleNamespace(device='cpu',compute_type='int8_float32')
        def transcribe(self,audio,**options):
            calls.append(options)
            return iter([SimpleNamespace(start=0.,end=1.,text='Speech',avg_logprob=0.,no_speech_prob=0.)]),SimpleNamespace(language='zh',language_probability=1.,duration=1.)
    transcriber=asr.Transcriber(replace(config,translation_backend='local'),progress=lambda s:None)
    transcriber.model=Model()
    transcriber.transcribe(config.source/'test.wav')
    transcriber.translate(config.source/'test.wav','zh')
    for options in calls:
        effective=lambda key:options.get(key,defaults[key].default)
        assert effective('beam_size')==effective('best_of')==5
        assert effective('temperature')==[0.,0.2,0.4,0.6,0.8,1.]
        assert effective('condition_on_previous_text') is True
        assert effective('prompt_reset_on_temperature')==0.5
        assert effective('compression_ratio_threshold')==2.4
        assert effective('log_prob_threshold')==-1.0
        assert effective('no_speech_threshold')==0.6
        assert effective('vad_filter') is False
        assert effective('hallucination_silence_threshold') is None
        assert effective('chunk_length') is None
        assert effective('language')=='zh'
    assert calls[0]['task']=='transcribe' and 'without_timestamps' not in calls[0]
    assert calls[1]['task']=='translate' and calls[1]['without_timestamps'] is True
    assert inspect.signature(FeatureExtractor).parameters['chunk_length'].default==30
    assert config.asr_model=='large-v3'
