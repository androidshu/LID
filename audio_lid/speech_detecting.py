from paddlespeech.cli.cls.infer import CLSExecutor
from paddleaudio.backends import soundfile_load as load
from paddle.audio.features import LogMelSpectrogram
from scipy import signal
from error_codes import *

import os
import paddle
import numpy as np
import argparse
import scipy.io.wavfile as wav
import audio_utils


class SpeechDetectObject:
    def __str__(self):
        return f'label:{self.label}, score:{self.score}, start:{self.start}, stop:{self.stop}'


class SpeechDetecting:
    def __init__(self, args):
        self.speech_checker = CLSExecutor()
        self.args = args
        device = paddle.get_device()
        paddle.set_device(device)
        self.speech_checker._init_from_path('panns_cnn14')
        self.denoise_option = ""
        if args.denoise_model is not None and os.path.exists(args.denoise_model):
            self.denoise_option += f' -af "arnndn=m= {args.denoise_model}"'
        print(f'init_speech_checker completely, use device:{device}')

    def check_speech(self, wav_samples, topk=1):
        #pre handle
        feat_conf = self.speech_checker._conf['feature']
        # Feature extraction
        feature_extractor = LogMelSpectrogram(
            sr=feat_conf['sample_rate'],
            n_fft=feat_conf['n_fft'],
            hop_length=feat_conf['hop_length'],
            window=feat_conf['window'],
            win_length=feat_conf['window_length'],
            f_min=feat_conf['f_min'],
            f_max=feat_conf['f_max'],
            n_mels=feat_conf['n_mels'], )
        feats = feature_extractor(
            paddle.to_tensor(paddle.to_tensor(wav_samples).unsqueeze(0)))
        infer_input = paddle.transpose(feats, [0, 2, 1]).unsqueeze(1)  # [B, N, T] -> [B, 1, T, N]
        infer_output = self.speech_checker.model(infer_input)
        # self.speech_checker.infer()
        result = infer_output.squeeze(0).numpy()

        if topk > len(self.speech_checker._label_list):
            print("Value of topk is larger than number of labels")
            return None

        topk_idx = (-result).argsort()[:topk]
        ret_list = []
        for idx in topk_idx:
            label, score = self.speech_checker._label_list[idx], result[idx]
            speech_ret = {'label': label, 'score': score}
            ret_list.append(speech_ret)
        # return speech_checker.postprocess(topk)  # Retrieve result of cls.
        return ret_list

    def is_empty_wav(self, wav_file, threshold=90, print_log=False):
        # print(wav_file)
        if isinstance(wav_file, np.ndarray):
            samples = wav_file
        else:
            return

        samples = samples if len(samples.shape) <= 1 else samples[:, 1]
        empty_count = np.sum(abs(samples) <= 0.1)
        total_count = len(samples)

        empty_rate = empty_count * 100 / total_count
        is_empty = empty_rate > threshold
        if print_log:
            print('is_empty:', is_empty, 'total_count:', total_count, ',empty_count:', empty_count, ',empty_rate:',
                  empty_rate)
        return is_empty

    def load_audio_samples(self, audio_file_path):
        sample_rate = 32000
        is_temp_file = False

        if audio_file_path.startswith("http"):
            temp_dir = self.args.temp_dir
            temp_name = f'{audio_utils.generate_md5(audio_file_path)}.wav'
            temp_file_path = os.path.join(temp_dir, temp_name)
            print(f'Warning: This is a online url, audio_file:{audio_file_path}, '
                  f'start downloading to temp file:{temp_file_path}')

            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            os.system(f'ffmpeg -i "{audio_file_path}" -ar {sample_rate} -ac 1 {self.denoise_option} -y {temp_file_path}')
            if not os.path.exists(temp_file_path):
                return ERROR_CODE_FILE_DOWNLOAD_FAILED, None
            is_temp_file = True
            audio_file_path = temp_file_path

        audio_file_path = os.path.abspath(os.path.expanduser(audio_file_path))
        if not os.path.exists(audio_file_path):
            return ERROR_CODE_FILE_NOT_EXISTS, None

        if not audio_file_path.endswith(".wav"):
            if self.args.debug:
                print("Warning: The file is not in WAV format. Start auto converting")
            filename = os.path.basename(audio_file_path)
            new_name = filename.split('.')[0]

            temp_file = os.path.join(os.path.dirname(audio_file_path), f'{new_name}.wav')
            os.system(f'ffmpeg -i "{audio_file_path}" -ar {sample_rate} -ac 1 {self.denoise_option} -y {temp_file}')
            if os.path.exists(temp_file):
                is_temp_file = True
                audio_file_path = temp_file
            else:
                return ERROR_CODE_MEDIA_FORMAT_CONVERT_ERROR, None

        samples, _ = load(
            file=audio_file_path,
            sr=sample_rate,
            mono=True,
            dtype='float32')
        samples = samples if len(samples.shape) <= 1 else samples[:, 1]

        if is_temp_file:
            if self.args.debug:
                print(f'deleting temp wav file:{audio_file_path}')
            os.remove(audio_file_path)
        return 0, samples

    def find_speech_list(self, samples, speech_score_threshold, speech_segment_duration):
        sample_rate = 32000
        total_samples_len = len(samples)
        duration = len(samples) / sample_rate
        if duration < 10:
            print(f'Error: The audio duration is too short! durationS:{duration}')
            return ERROR_CODE_MEDIA_FILE_TOO_SHORT, None

        start_offset_duration = self.args.parse_start_offset
        if duration - start_offset_duration < 300:
            start_offset_duration = 0

        start_offset_sample = start_offset_duration * sample_rate
        speech_list = []
        samples_per_zone = int((total_samples_len - start_offset_sample) / self.args.speech_segment_count)

        detect_duration = min(max(3, speech_segment_duration), 20)
        samples_per_window = int(detect_duration * sample_rate)

        step_duration = 2
        samples_per_step = int(step_duration * sample_rate)

        empty_duration = 1
        samples_per_empty = int(empty_duration * sample_rate)

        curr_speech_count = 0

        stop = start_offset_sample
        empty_length = 0
        empty_wav_count = 0

        while stop < total_samples_len:

            while True:
                start = stop
                stop = start + samples_per_empty
                if stop >= total_samples_len:
                    break  # TODO
                new_samples = samples[start: stop]
                if self.is_empty_wav(samples[start: stop]):
                    empty_wav_count += 1
                    empty_length += len(new_samples)
                    continue
                break

            stop = start + samples_per_window
            if start >= total_samples_len or stop >= total_samples_len:
                break

            new_samples = samples[start: stop]
            if self.is_empty_wav(new_samples):
                empty_wav_count += 1
                empty_length += len(new_samples)
                continue

            result_list = self.check_speech(new_samples, topk=3)
            if self.args.debug:
                print(f'check result:{result_list}')
            speech_score = 0
            speech_label = ''

            if result_list is None or len(result_list) == 0:
                continue

            for result in result_list:
                score = result['score']
                label = result['label']
                if label.__contains__('Speech') and score > speech_score:
                    speech_label = label
                    speech_score = score

            if speech_score > speech_score_threshold:
                speech_object = SpeechDetectObject()
                speech_object.label = speech_label
                speech_object.start = start
                speech_object.stop = stop
                speech_object.samples = signal.resample(new_samples, int(len(new_samples) * float(16000) / sample_rate))
                # speech_object.samples = librosa.resample(new_samples, sample_rate, 16000)
                speech_object.score = speech_score
                speech_list.append(speech_object)
                curr_speech_count += 1
                stop = max(stop, start_offset_sample + samples_per_zone * curr_speech_count)
            else:
                stop = start + samples_per_step

        return len(speech_list), speech_list


if __name__ == '__main__':
    # audio_file = "/Users/bevis/PycharmProjects/audio-lid/dataset/test_100/mp3/19d77ef6cc1fcc4b266b1886c4afb18d.mp3"
    audio_file = "http://vfx.mtime.cn/Video/2019/06/27/mp4/190627231412433967.mp4"

    parser = argparse.ArgumentParser(add_help=True)
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
    parser.add_argument(
        '--output-path', type=str, default=None,
        help='The output dir use to save result files, default:None')
    parser.add_argument(
        '--denoise-model', type=str, default=None,
        help='The model use to denosie to make speech clear, default:None')

    args = parser.parse_args()

    # args = parser.parse_args(sys.argv)
    # asr = ASRExecutor()
    # result = asr(audio_file="/Users/bevis/Downloads/en.wav")
    # print(result)

    # cls = CLSExecutor()
    # result = cls(audio_file="/Users/bevis/Downloads/zh.wav", topk=3)
    # print(result)
    #
    # # cls = CLSExecutor()
    # result = cls(audio_file="/Users/bevis/Downloads/en.wav", topk=3)
    # print(result)

    speech_detecting = SpeechDetecting(args)
    ret, samples = speech_detecting.load_audio_samples(args.audio_file)
    if ret < 0:
        print(f'load audio file failed, ret:{ret}')
        exit(1)
    ret, speech_list = speech_detecting.find_speech_list(samples, args.speech_score_threshold,
                                        args.speech_segment_duration)
    print(f'find speech result code:{ret}')

    if 0 <= ret < args.speech_segment_count:
        # try again
        ret, speech_list = speech_detecting.find_speech_list(samples, max(args.speech_score_threshold - 0.2, 0.4),
                                            max(args.speech_segment_duration - 2, 3))
        print(f'try to find speech result code:{ret}')

    if ret > 0 and args.output_path is not None:
        dir_path = args.output_path
        if args.debug:
            print(f'save audio seg and manifest file to dir:{dir_path}')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        index = 0
        file_path_list = []
        for speech_obj in speech_list:
            if args.debug:
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
            for file_path in file_path_list:
                f.write('eng\t1\n')
        if args.debug:
            print(f'Out put file dir:{args.output_path}')


