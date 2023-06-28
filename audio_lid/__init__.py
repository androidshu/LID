# __init__.py
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from .audio_lid import AudioLID
from .audio_lid import run
from .language_identify import LanguageIdentify
from .speech_detecting import SpeechDetecting, SpeechDetectObject
from .error_codes import *

import audio_lid.language_identify
import audio_lid.speech_detecting
import audio_lid.error_codes

__all__ = ['AudioLID', 'LanguageIdentify', 'SpeechDetecting', 'SpeechDetectObject', 'run']
