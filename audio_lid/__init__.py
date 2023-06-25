# __init__.py

from .audio_lid import AudioLID
from .language_identify import LanguageIdentify
from .speech_detecting import SpeechDetecting, SpeechDetectObject
from .error_codes import *

import audio_lid.language_identify
import audio_lid.speech_detecting
import audio_lid.error_codes

__all__ = ['AudioLID', 'LanguageIdentify', 'SpeechDetecting', 'SpeechDetectObject']
