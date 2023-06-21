import argparse
import os
import scipy.io.wavfile as wav
import numpy as np

from fairseq import options
from speech_detecting import SpeechDetecting
from language_identify import LanguageIdentify
from error_codes import *


class LID:

    def __init__(self,
                 language_model,
                 lang_dict_dir,
                 debug=False,
                 temp_path='./temp',
                 speech_segment_count=5,
                 speech_segment_duration=5,
                 speech_score_threshold=0.7,
                 parse_start_offset=60,
                 top_k=3,
                 denoise_model=None):
        np.random.seed(123)
        args_parser = options.get_generation_parser(default_task="audio_classification")
        input_args = []
        input_args.append(lang_dict_dir)
        input_args.append('--path')
        input_args.append(language_model)
        self.args = options.parse_args_and_arch(args_parser, input_args=input_args)
        self.args.debug = debug
        self.args.temp_path = temp_path
        self.args.speech_segment_count = speech_segment_count
        self.args.speech_segment_duration = speech_segment_duration
        self.args.speech_score_threshold = speech_score_threshold
        self.args.parse_start_offset = parse_start_offset
        self.args.top_k = top_k
        self.args.denoise_model = denoise_model
        self.language_identify = LanguageIdentify(self.args)
        self.speech_detecting = SpeechDetecting(self.args)

    def infer_language(self, audio_file):
        ret, samples = self.speech_detecting.load_audio_samples(audio_file)
        if ret < 0:
            print(f'load audio file failed, ret:{ret}')
            exit(1)
        ret, speech_list = self.speech_detecting.find_speech_list(samples, args.speech_score_threshold,
                                                                  args.speech_segment_duration)
        print(f'find speech result code:{ret}')

        if 0 <= ret < args.speech_segment_count:
            # try again
            ret, speech_list = self.speech_detecting.find_speech_list(samples, max(args.speech_score_threshold - 0.2, 0.4),
                                                                      max(args.speech_segment_duration - 2, 3))
            print(f'try to find speech result code:{ret}')

        if args.debug:
            if ret > 0:
                dir_path = args.temp_dir
                print(f'save audio seg and manifest file to dir:{dir_path}')
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                index = 0
                file_path_list = []
                for speech_obj in speech_list:
                    print(f'speech_obj:{speech_obj}')
                    index += 1
                    file_path = os.path.join(dir_path, f'index_{index}.wav')
                    wav.write(file_path, 16000, speech_obj.samples)
                    file_path_list.append(os.path.abspath(file_path))

                manifest_tsv_file = os.path.join(dir_path, 'manifest.tsv')
                manifest_lang_file = os.path.join(dir_path, 'manifest.lang')
                with open(manifest_tsv_file, mode='w+', encoding='utf-8') as f:
                    f.write("/\n")
                    for file_path in file_path_list:
                        f.write(f'{file_path}\t16000\n')
                with open(manifest_lang_file, mode='w+', encoding='utf-8') as f:
                    for _ in file_path_list:
                        f.write('eng\t1\n')

        if ret < 0:
            return ret, None
        predictions = self.language_identify.infer([speech_obj.samples for speech_obj in speech_list])
        if args.debug:
            print(f'prediction origin result:{predictions}')

        total_score = 0
        min_threshold = 0.4
        valid_count = 0
        language_score_map = {}
        for key, prediction in predictions.items():
            for language_str, score in prediction:
                if score < min_threshold:
                    continue
                valid_count += 1
                total_score += score

                language_total_score = 0
                if language_str in language_score_map:
                    language_total_score = language_score_map[language_str]
                language_total_score += score
                language_score_map[language_str] = language_total_score

        if args.debug:
            print(f'language resort map:{language_score_map}')
        result_list = []
        if len(language_score_map) == 0:
            return ERROR_CODE_NO_VALID_LANGUAGE, None

        for language_str, score in language_score_map.items():
            result_list.append((language_str, float(score * 100 / total_score)))
        result_list = sorted(result_list, key=lambda language_obj: language_obj[1], reverse=True)

        if args.debug:
            print(f'result_list:{result_list}')
        return valid_count, result_list


if __name__ == '__main__':
    audio_file = "/Users/bevis/PycharmProjects/LID/dataset/test_100/mp3/a03f2c4780798a5398c86b196e479275.mp3"
    # audio_file = "http://vfx.mtime.cn/Video/2019/06/27/mp4/190627231412433967.mp4"
    parser = argparse.ArgumentParser(add_help=True)
    # speech detecting
    parser.add_argument(
        '--audio-file', type=str, default=audio_file,
        help='Audio file to detect language, support local file and online url')
    parser.add_argument(
        '--debug', type=bool, default=True,
        help='Debug mode.')
    parser.add_argument(
        '--speech-segment-count', type=int, default=5,
        help='The segment count need to detect language, default:5')
    parser.add_argument(
        '--speech-segment-duration', type=int, default=5,
        help='The length of each speech segment, unit:second, default:5')
    parser.add_argument(
        '--speech-score-threshold', type=float, default=0.7, help='The threshold, range:0-1,unit:float, default:0.7')
    parser.add_argument(
        '--parse-start-offset', type=int, default=60,
        help='The file start offset that need to skip, unit:second, default:60')
    parser.add_argument(
        '--temp-dir', type=str, default='./temp',
        help='The temp dir use to save temp file, default:./temp')

    # language identify
    parser.add_argument("--infer-num-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)

    parser.add_argument(
        '--language-model', type=str, default='./pretrain/mms1b_l126.pt',
        help='The model use to detect language class, default:./pretrain/mms1b_l126.pt')

    parser.add_argument(
        '--lang-dict-dir', type=str, default='./pretrain',
        help='The dir contains the file of language, default:./pretrain')

    parser.add_argument(
        '--denoise-model', type=str, default=None,
        help='The model use to denosie to make speech clear, default:None')
    args = parser.parse_args()

    lid = LID(language_model=args.language_model, lang_dict_dir=args.lang_dict_dir, debug=args.debug,
              speech_segment_count=args.speech_segment_count, speech_segment_duration=args.speech_segment_duration,
              speech_score_threshold=args.speech_score_threshold, parse_start_offset=args.parse_start_offset,
              top_k=args.top_k, denoise_model=args.denoise_model)
    ret, language_list = lid.infer_language(args.audio_file)
    print(f'infer result:{ret}, language list:{language_list}')






