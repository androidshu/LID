import json
import os
import hashlib
import random
import sys
import threading
import _thread
import time

import scipy.io.wavfile as wav
import numpy as np
import torch
import torch.nn.functional as F
import shutil
from tqdm import tqdm
# import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ProcessPool
from concurrent.futures import ThreadPoolExecutor
import importlib
import PIL.Image as Image

dnoise_flag = '_dnoise'
language_flag_str = '_lang'


def adjust_wav_file_path(map_file, base_dir):
    with open(map_file, 'r') as f:
        wav_map = json.load(f)

    dir_path_wav_map = {}
    for dir in wav_map:
        dir_path = os.path.join(os.path.join(base_dir, dir), 'clips')
        if not os.path.isdir(dir_path):
            continue
        dir_path_wav_map[dir_path] = wav_map[dir]

    for dir_path in dir_path_wav_map:
        print('list dir_path:', dir_path)
        wav_files = os.listdir(dir_path)

        for wav_file in tqdm(wav_files):
            for dir_path_record in dir_path_wav_map:
                wav_record_files = dir_path_wav_map[dir_path_record]
                if wav_file in wav_record_files:
                    if dir_path_record != dir_path:
                        wav_file_path = os.path.join(dir_path, wav_file)
                        wav_dest_file_path = os.path.join(dir_path_record, wav_file)
                        print('find need adjust wav file:', wav_file, ',current dir:', dir_path, ',dest dir:',
                              dir_path_record)
                        shutil.move(wav_file_path, wav_dest_file_path)
                    break


def is_empty_wav(wav_file, threshold=90, print_log=False):
    # print(wav_file)
    if isinstance(wav_file, str):
        sr, samples = wav.read(wav_file)
    elif isinstance(wav_file, np.ndarray):
        samples = wav_file
    else:
        return

    samples = samples if len(samples.shape) <= 1 else samples[:, 1]

    empty_count = np.sum(abs(samples) <= 50)
    total_count = len(samples)

    # x = np.linspace(0, 100, 180000)
    #
    # plt.plot(x, samples[60000:])
    # plt.show()

    empty_rate = empty_count * 100 / total_count
    is_empty = empty_rate > threshold
    if print_log:
        print('is_empty:', is_empty, 'total_count:', total_count, ',empty_count:', empty_count, ',empty_rate:',
              empty_rate)
    return is_empty


def check_speech_or_lang_type(sr, samples, check_speech=True, threshold=0.95):
    if threshold > 1:
        raise Exception('threshold must below 1, threshold:{}'.format(threshold))
    start = time.time()
    acc = 1 #infer_recognition.infer_samples(sr, samples, check_speech)
    end = time.time()
    # print('time total coast:', (end - start) * 1000)
    acc = torch.squeeze(acc)
    softmax = F.softmax(acc, dim=0)
    max_index = softmax.argmax().item()
    max_rate = softmax[max_index].item()

    print('max_index:', max_index, ', max_rate:', max_rate)
    if max_rate >= threshold:
        return max_index  # 1:人说话， 0:背景音
    return -1


def check_image_speech_or_lang_type(image_data, check_speech=True, threshold=0.95):
    if threshold > 1:
        raise Exception('threshold must below 1, threshold:{}'.format(threshold))
    start = time.time()
    acc = 1 #infer_recognition.infer_img_data(image_data, check_speech)
    end = time.time()
    # print('time total coast:', (end - start) * 1000)
    acc = torch.squeeze(acc)
    softmax = F.softmax(acc, dim=0)
    max_index = softmax.argmax().item()
    max_rate = softmax[max_index].item()

    # print('max_index:', max_index, ', max_rate:', max_rate)
    if max_rate >= threshold:
        return max_index  # 1:人说话， 0:背景音
    return -1


def handle_input_file(movie_path, target_sr=None, target_channel=None, save_dir=None):
    name = os.path.basename(movie_path).split('.')[0]
    # 抽取音频
    if not movie_path.endswith('.wav') or dnoise_flag not in movie_path:
        dir_path = os.path.dirname(movie_path)
        if movie_path.startswith('http'):
            dir_path = save_dir
            name = getMd5(movie_path)
        audio_path = os.path.join(dir_path, name + '{}.wav'.format(dnoise_flag))
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if not os.path.exists(audio_path):
            option = ''
            post_option = ''
            if movie_path.startswith('/') and (movie_path.endswith('.m3u8') or movie_path.endswith('.m3u')):
                option = '-allowed_extensions ALL'

            target_sr = 48000
            if target_sr is not None:
                post_option += ' -ar {} '.format(target_sr)

            if target_channel is not None:
                post_option += ' -ac {} '.format(target_channel)

            dnoise = None
            if dnoise is not None:
                post_option += ' -af "arnndn=m= /ProjectRoot/Spoken-language-identification/dnoise/lq.rnnn"'
            # /usr/local/Cellar/ffmpeg/4.4.1_3/bin/ffmpeg
            command = '/ProjectRoot/ffmpeg-4.3/ffmpeg_g  {} -i "{}" {} -y {}'.format(option, movie_path, post_option,
                                                                                     audio_path)
            print('ffmpeg command:', command)
            os.system(command)
            # if os.path.exists(audio_path):
            #     os.remove(movie_path)
        # else:
        #
        return audio_path
    return movie_path


def infer_movie_language(movie_path, save_dir='./dataset/movie_test', background_dir='./dataset/movie_test/0',
                         unknown_dir='./dataset/movie_mp3/-1', move_whole_dir=True):
    # temp local
    print('save_dir:', save_dir)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(background_dir):
        os.makedirs(background_dir)
    if not os.path.exists(unknown_dir):
        os.makedirs(unknown_dir)

    temp_dir = os.path.join(os.path.dirname(save_dir), '../audio_lid/temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    print_log = True
    log_path = os.path.join(save_dir, 'log.txt')
    log_file_fd = os.open(log_path, os.O_CREAT | os.O_RDWR)
    os.write(log_file_fd, str.encode('start\n'))

    name = os.path.basename(movie_path).split('.')[0]
    dir = os.path.dirname(movie_path)

    if language_flag_str in movie_path:
        end = name.index(language_flag_str) + len(language_flag_str) + 1
        expect_save_dir = name[:end]
        if os.path.isdir(os.path.join(dir, expect_save_dir)):
            print('movie already handle, path:', movie_path)
            return

    audio_path = handle_input_file(movie_path, save_dir=os.path.dirname(save_dir))

    os.write(log_file_fd, str.encode('wav file path:{}\n'.format(audio_path)))
    if print_log:
        print('wav file path:', audio_path)
    if not os.path.exists(audio_path):
        os.write(log_file_fd, str.encode('movie_path not exists\n'))
        print('movie_path not exists:', audio_path)
        return False

    sample_rate, samples = wav.read(audio_path)
    samples = samples if len(samples.shape) <= 1 else samples[:, 1]

    new_path = audio_path.replace('.wav', '_new.wav')

    detect_duration = 5
    samples_per_window = int(detect_duration * sample_rate)

    step_duration = 3
    base_path = new_path
    samples_per_step = int(step_duration * sample_rate)

    empty_duration = 1
    samples_per_empty = int(empty_duration * sample_rate)

    low_language_count_arr = np.zeros(10)
    language_count_arr = np.zeros(10)
    empty_wav_count = 0
    total_samples_len = len(samples)

    os.write(log_file_fd, str.encode('start for samples\n'))
    bg_file_list = []
    stop = 0
    speech_length = 0
    empty_length = 0
    bg_length = 0
    unknown_length = 0

    while stop < total_samples_len:

        while True:
            start = stop
            stop = start + samples_per_empty
            if stop > total_samples_len:
                break  # TODO
            new_samples = samples[start: stop]
            if is_empty_wav(samples[start: stop]):
                empty_wav_count += 1
                empty_length += len(new_samples)
                continue
            break

        stop = start + samples_per_window
        if start >= total_samples_len or stop >= total_samples_len:
            break

        new_samples = samples[start: stop]
        if is_empty_wav(new_samples):
            empty_wav_count += 1
            empty_length += len(new_samples)
            continue

        # speech_type = check_speech_type(sample_rate, new_samples, True)
        # if speech_type == 1:
        #     speech_length += len(new_samples)
        # elif speech_type == 0:
        #     bg_length += len(new_samples)
        # else:
        #     unknown_length += len(new_samples)
        #
        # if True:
        #     continue
        new_path = base_path.replace('.wav', '_{}.wav'.format(start))
        new_path = os.path.join(temp_dir, os.path.basename(new_path))
        if os.path.exists(new_path):
            os.remove(new_path)

        wav.write(new_path, sample_rate, new_samples)
        # if is_empty_wav(new_path):
        #     os.remove(new_path)
        #     empty_wav_count += 1
        #     continue
        # if len(new_sameples) < 100 * 1024:
        #     continue

        save_png_path = os.path.dirname(new_path)
        check_speech = True
        acc = 1 #infer_recognition.infer(new_path, save_png_path, check_speech)
        if acc is None:
            continue
        acc = torch.squeeze(acc)
        softmax = F.softmax(acc, dim=0)
        max_index = softmax.argmax().item()
        max_rate = softmax[max_index].item()

        img_path = os.path.join(save_png_path, os.path.basename(new_path).replace('.wav', '.png'))
        audio_name = os.path.basename(new_path)
        new_audio_name = audio_name.replace('.wav', '_{}_{:.2f}.wav'.format(max_index, max_rate))
        # if max_rate >= 0.80:

        if print_log:
            log_msg = 'wav name:{}\n label:{}, rate:{}\n'.format(audio_name, max_index, max_rate)
            print(log_msg)
            os.write(log_file_fd, str.encode(log_msg))
        language_count_arr[max_index] += 1
        dest_dir = background_dir
        # if max_index > 0:
        dest_dir = os.path.join(save_dir, 'lang_{}'.format(max_index if max_rate >= 0.80 else -1))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        dest_wav_path = os.path.join(dest_dir, new_audio_name)
        if os.path.exists(dest_wav_path):
            os.remove(dest_wav_path)
            # print('exists')

        # dest_img_path = os.path.join(dest_dir, os.path.basename(img_path))
        # if os.path.exists(dest_img_path):
        #     os.remove(dest_img_path)
        print('move wav from cur:', new_path)
        print('move wav to dest_wav_path:', dest_wav_path)
        shutil.move(new_path, dest_wav_path)
        if max_index == 0:
            bg_file_list.append(dest_wav_path)
        elif 0 < max_index <= 3:
            stop -= (samples_per_window - samples_per_step)
            # shutil.move(img_path, dest_dir)
            # else:
            #     low_language_count_arr[max_index] += 1
            #     dest_file_path = os.path.join(unknown_dir, new_audio_name)
            #     shutil.move(new_path, dest_file_path)

            if print_log:
                print('delete low label:{}, rate:{}'.format(max_index, max_rate))
        try:
            os.remove(img_path)
        except Exception as e:
            print(e)
    # create_spectrograms.pcm_to_img()
    offset = 1
    index = np.argmax(language_count_arr[offset:]) + offset
    max_lang_count = language_count_arr[index]

    try:
        os.remove(audio_path)
    except Exception as e:
        print(e)
    if not check_speech and move_whole_dir and max_lang_count > 0:
        # if language_flag_str not in audio_path:
        #     new_audio_path = audio_path.replace(dnoise_flag, '{}{}{}'.format(language_flag_str, index, dnoise_flag))
        #     if os.path.exists(new_audio_path):
        #         os.remove(new_audio_path)
        #     shutil.move(audio_path, new_audio_path)

        if language_flag_str not in save_dir:
            new_save_dir = save_dir + '{}{}'.format(language_flag_str, index)
            if os.path.exists(new_save_dir):
                shutil.rmtree(new_save_dir)
            shutil.move(save_dir, new_save_dir)

    if print_log:
        print('max_lang_count:', max_lang_count)
        print('audio_path:', os.path.basename(audio_path))
        print('empty_wav_count:', empty_wav_count)
        print('language_count_arr:', language_count_arr)
        print('low_language_count_arr:', low_language_count_arr)

        total_length = len(samples)
        print('speech_length:', speech_length, ', rate:', speech_length / total_length)
        print('bg_length:', bg_length, ', rate:', bg_length / total_length)
        print('empty_length:', empty_length, ', rate:', empty_length / total_length)
        print('unknown_length:', unknown_length, ', rate:', unknown_length / total_length)
        print('total_length:', total_length)

    # keep_bg_files = 30
    # random.shuffle(bg_file_list)
    # if len(bg_file_list) > keep_bg_files:
    #     for bg_file_path in bg_file_list[keep_bg_files:]:
    #         os.remove(bg_file_path)
    os.write(log_file_fd, str.encode('end\n'))
    os.close(log_file_fd)


def getMd5(url):
    md5 = hashlib.md5()
    md5.update(url.encode('utf-8'))
    md5_name = md5.hexdigest()
    return md5_name


def get_file_md5(file_path):
    with open(file_path, 'rb') as f:
        md5 = hashlib.md5()
        md5.update(f.read())
        md5_name = md5.hexdigest()
    return md5_name


def movie_to_wav_file(movie_path, movie_wav_save_dir, extend_name, target_sr=32000, target_channel=1,
                      target_duration=None):
    md5_name = getMd5(movie_path)

    save_wav_path = os.path.join(movie_wav_save_dir, md5_name + extend_name)
    if os.path.exists(save_wav_path):
        if os.path.getsize(save_wav_path) > 100000:
            return save_wav_path
        os.remove(save_wav_path)
    temp_wav_path = os.path.join(movie_wav_save_dir, md5_name + '_temp.mp3')
    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)
    option = ''
    if target_sr is not None:
        option += ' -ar ' + str(target_sr)
    if target_channel is not None:
        option += ' -ac ' + str(target_channel)
    if target_duration is not None:
        option += ' -t ' + str(target_duration)
    command = 'ffmpeg -i "{}" {} -y {}'.format(movie_path, option, temp_wav_path)
    print('command:', command)
    os.system(command)
    if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) < 1024:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        return None
    shutil.move(temp_wav_path, save_wav_path)
    return save_wav_path


def download_top_movie_list(movie_list_lines, movie_wav_save_dir, extend_name='.wav'):
    for line in movie_list_lines:
        exit_file_path = os.path.join(movie_wav_save_dir, 'exit')
        if os.path.exists(exit_file_path):
            break
        try:
            origin_file = line.replace('\n', '').split('\t')[0].split(' ')[0]
            movie_to_wav_file(origin_file, movie_wav_save_dir, extend_name)
        except Exception as e:
            print(e)


def download_top_movie(movie_list_file_path, movie_wav_save_dir, extend_name='.wav', processes=50):
    with open(movie_list_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        f.close()
    if not os.path.exists(movie_wav_save_dir):
        os.makedirs(movie_wav_save_dir)

    # lock = threading.Lock()
    #
    # def loop():
    #     print('fuck start')
    #     while True:
    #         lock.acquire()
    #         global movie_index
    #         movie_index = movie_index + 1
    #         local_index = movie_index
    #         lock.release()
    #
    #         if local_index >= len(lines):
    #             break
    #         try:
    #             origin_file = lines[local_index].replace('\n', '').split('\t')[0]
    #             movie_to_wav_file(origin_file, movie_wav_save_dir, extend_name)
    #         except Exception as e:
    #             print(e)
    #     print('fuck end')
    #
    # threads = []
    # for index in range(6):
    #     thread = threading.Thread(target=loop)
    #     thread.start()
    #     threads.append(thread)
    #
    # for thread in threads:
    #     thread.join()
    total_count = len(lines)
    if total_count < processes * 3:
        processes = int(total_count / 3)
    if processes < 1:
        processes = 1
    step = int(total_count / processes)
    if step < 1:
        step = 1

    for index in range(processes):
        start = index * step
        end = min(start + step, total_count)

        pid = 0
        if index < processes - 1:
            pid = os.fork()

        if pid == 0:
            print(
                'child process, index:{}, pid:{}, start_index:{}, end_index:{}'.format(index, os.getpid(), start, end))
            download_top_movie_list(lines[start:end], movie_wav_save_dir, extend_name)


def convert_audio_format(input_dir, output_dir=None, origin_extend=None, target_extend=None, remove_origin=True):
    if output_dir is None:
        output_dir = input_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dir_list = [input_dir]
    while len(dir_list) > 0:
        curr_dir = dir_list.pop(0)
        audio_files = os.listdir(curr_dir)
        curr_dest_dir = curr_dir
        if output_dir != input_dir:
            curr_dest_dir = curr_dir.replace(input_dir, output_dir)
            if not os.path.exists(curr_dest_dir):
                os.makedirs(curr_dest_dir)

        for audio_file in tqdm(audio_files):
            print('audio_file:', audio_file)
            if audio_file.startswith('.'):
                continue
            audio_file_path = os.path.join(curr_dir, audio_file)
            audio_dest_file_path = os.path.join(curr_dest_dir, audio_file)
            if os.path.isdir(audio_file_path):
                dir_list.append(audio_file_path)
                continue
            if not audio_file.endswith(origin_extend):
                print('continue origin_extend:', origin_extend)
                continue

            audio_dest_file_path = audio_dest_file_path.replace(origin_extend, target_extend)
            command = 'ffmpeg -i {} -y {}'.format(audio_file_path, audio_dest_file_path)
            os.system(command)
            if remove_origin and os.path.exists(audio_dest_file_path):
                os.remove(audio_file_path)


def download_and_split_movie(movie_list_file_path, movie_wav_save_dir, extend_name, processes=20):
    with open(movie_list_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    urls = []
    for line in lines:
        movie_url = line.replace('\n', '').split('\t')[0].split(' ')[0]
        urls.append(movie_url)

    if not os.path.exists(movie_wav_save_dir):
        os.makedirs(movie_wav_save_dir)

    already_handle_mp3_md5_list = []
    mp3_files = os.listdir(movie_wav_save_dir)
    for mp3_file in mp3_files:
        if not mp3_file.endswith('.mp3'):
            continue
        mp3_file_path = os.path.join(movie_wav_save_dir, mp3_file)
        mp3_file_md5 = get_file_md5(mp3_file_path)
        already_handle_mp3_md5_list.append(mp3_file_md5)

    lock = threading.Lock()

    def download_and_read(url_list):
        for audio_url in url_list:
            md5 = getMd5(audio_url)
            audio_dir = os.path.join(movie_wav_save_dir, md5)
            if os.path.isdir(audio_dir):
                print('audio_dir exists, audio_url had handled:', audio_url)
                continue
            temp_audio_dir = audio_dir + '_temp'
            if os.path.exists(temp_audio_dir):
                shutil.rmtree(temp_audio_dir)
            background_dir = os.path.join(temp_audio_dir, '0')
            unknown_dir = os.path.join(temp_audio_dir, '-1')

            audio_path = movie_to_wav_file(audio_url, movie_wav_save_dir, extend_name)
            if audio_path is None:
                continue
            file_md5 = get_file_md5(audio_path)
            # already_handle_mp3_md5_list.append(file_md5)
            # print('lock outside, thread:', threading.currentThread().getName())
            # print('already_handle_mp3_md5_list:', already_handle_mp3_md5_list)
            # print('lock inside, thread:', threading.currentThread().getName())
            # i = 0
            # while True:
            #     i += 1
            lock.acquire()
            if file_md5 in already_handle_mp3_md5_list:
                print('file_md5 exists, audio_url had handled:', audio_url)
                continue
            lock.release()
            infer_movie_language(audio_path, temp_audio_dir, background_dir=background_dir, unknown_dir=unknown_dir,
                                 move_whole_dir=False)
            sub_dirs = os.listdir(temp_audio_dir)
            for sub_dir in sub_dirs:
                if sub_dir.startswith('.'):
                    continue
                sub_dir_path = os.path.join(temp_audio_dir, sub_dir)
                if not os.path.isdir(sub_dir_path):
                    continue
                short_dir_file(sub_dir_path, 30, extend='.wav')
                convert_audio_format(sub_dir_path, '.wav', '.mp3')
            shutil.move(temp_audio_dir, audio_dir)

            lock.acquire()
            already_handle_mp3_md5_list.append(file_md5)
            lock.release()

    total_count = len(lines)
    step = int(total_count / processes)
    thread_pool = ThreadPoolExecutor()
    futures = []
    for index in range(processes):
        start = index * step
        end = min(start + step, total_count)
        future = thread_pool.submit(download_and_read, urls[start: end])
        futures.append(future)

    for future in futures:
        future.result()
    thread_pool.shutdown()


def delete_png_file(input_dir):
    files = os.listdir(input_dir)
    for file in files:
        if '.png' in file:
            os.remove(os.path.join(input_dir, file))


def split_movie_inner(files, input_dir, output_dir=None):
    background_dir = os.path.join(input_dir, '0')
    if not os.path.exists(background_dir):
        os.makedirs(background_dir)

    if output_dir is None:
        output_dir = input_dir
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unknown_dir = os.path.join(input_dir, '-1')
    if not os.path.exists(unknown_dir):
        os.makedirs(unknown_dir)

    for file in tqdm(files):
        print(file)
        movie_file = os.path.join(input_dir, file)

        names = file.replace('.wav', '').replace('.mp3', '').split('_')

        sub_dir = os.path.join(output_dir, names[0])
        try:
            infer_movie_language(movie_file, sub_dir, background_dir, unknown_dir)
        except Exception as e:
            print(e)


def mulit_split_movie(input_dir, output_dir, files, processes=1):
    total_count = len(files)
    step = int(total_count / processes)
    args = []
    # pool = ProcessPool(processes=processes)
    # split_movie_inner_partial = partial(split_movie_inner, input_dir=input_dir)
    for index in range(processes):
        start = index * step
        end = min(start + step, total_count)

        pid = 0
        if index < processes - 1:
            pid = os.fork()

        if pid == 0:
            split_movie_inner(files[start:end], input_dir, output_dir)
        # args.append(files[start:end])

    # pool.map(split_movie_inner_partial, args)
    # pool.close()
    # pool.join()


def split_movie(input_dir, output_dir, processes=1):
    if not os.path.exists(input_dir):
        print('dir not exists:', input_dir)
        return

    files = os.listdir(input_dir)
    new_files = []

    for file in files:
        if file.startswith('.'):
            continue

        movie_file = os.path.join(input_dir, file)
        if os.path.isdir(movie_file):
            delete_png_file(movie_file)
            continue
        if '_temp' in file:
            continue

        md5_name = file.split('_')[0].split('.')[0]
        if dnoise_flag in file and language_flag_str in file:
            files.remove(md5_name + '.mp3')

            # 暂时添加，交给里面进行判断
            new_files.append(file)
            continue

        # # 仅仅dnoise 说明是之前没有完成的，可能数据也不完整，直接删除
        if dnoise_flag in file:
            # os.remove(movie_file)
            continue
        if '.wav' not in file and '.mp3' not in file:
            continue

        new_files.append(file)

    files = new_files
    list.sort(files, key=lambda x: os.path.getctime(os.path.join(input_dir, x)))
    args = sys.argv
    start = 0
    end = -1
    if len(args) >= 3:
        start = int(args[1])
        end = int(args[2])
        print('start:', start, ', end:', end)
    mulit_split_movie(input_dir, output_dir, files[start:end], processes)


def split_special_movie(input_dir, output_dir, movie_file_path, processes=1):
    with open(movie_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_files = []
    for line in lines[:500]:
        names = line.split(' ')
        if len(names) > 2:
            continue
        md5_name = names[1].replace('\n', '')
        new_files.append(md5_name + '.mp3')
    mulit_split_movie(input_dir, output_dir, new_files, processes)


def delete_empty_wav(input_dir):
    fliter_file_size = 10 * 1024 * 1024
    files = os.listdir(input_dir)
    for file in tqdm(files):
        if file.startswith('.'):
            continue

        file_path = os.path.join(input_dir, file)
        if os.path.isdir(file_path):
            delete_empty_wav(file_path)
        else:
            if not file.endswith('.wav'):
                continue
            if os.path.getsize(file_path) > fliter_file_size:
                continue

            empty = is_empty_wav(file_path, print_log=True)
            # if empty:
            # os.remove(file_path)


def exec_ffmpeg_command(input_dir, ffmpeg_command, delete_origin_file):
    files = os.listdir(input_dir)
    for file in tqdm(files):
        if file.startswith('.'):
            continue

        file_path = os.path.join(input_dir, file)
        if os.path.isdir(file_path):
            # delete_empty_wav(file_path)
            continue
        else:
            if not file.endswith('.wav'):
                continue
            new_path = file_path.replace('.wav', '_new.wav')
            os.system(ffmpeg_command.format(file_path, new_path))
            if delete_origin_file and os.path.exists(new_path):
                os.remove(file_path)


def merge_wav_segment(input_dir):
    if not os.path.exists(input_dir):
        print('dir not exists:', input_dir)
        return
    files = os.listdir(input_dir)

    for file in tqdm(files):
        if file.startswith('.'):
            continue
        sub_dir = os.path.join(input_dir, file)
        if not os.path.isdir(sub_dir):
            continue

        if language_flag_str not in file:
            continue
        lang_id = int(file[-1]).__str__()
        dest_dir = os.path.join(input_dir, language_flag_str + lang_id)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        sub_dir_files = os.listdir(sub_dir)
        for sub_file in sub_dir_files:
            if sub_file.startswith('.'):
                continue

            lang_dir = os.path.join(sub_dir, sub_file)
            if not os.path.isdir(lang_dir):
                continue
            lang_files = os.listdir(lang_dir)
            for lang_file in lang_files:
                if not lang_file.endswith('.wav'):
                    continue

                src_file = os.path.join(lang_dir, lang_file)
                dest_file = os.path.join(dest_dir, lang_file)

                shutil.copy(src_file, dest_file)
            # shutil.move(lang_dir, dest_dir)


def merge_special_lang_segment(input_dir, output_dir, special_lang, check_parent=False, select_count=20):
    if special_lang == 0 or special_lang == 8:
        check_parent = False

    if not os.path.exists(input_dir):
        print('dir not exists:', input_dir)
        return
    files = os.listdir(input_dir)
    lang_flag = 'lang_' + str(special_lang)

    music_dir = os.path.join(output_dir, lang_flag)
    if not os.path.exists(music_dir):
        os.makedirs(music_dir)

    already_in_online2_file_list = []
    # with open('record/online2_wav_file_list.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if '.wav' not in line:
    #             continue
    #         file_prefix = line.replace('\n', '')
    #         file_prefix = file_prefix[:file_prefix.rindex('_')]
    #         already_in_online2_file_list.append(file_prefix)
    merge_wav_count = 0
    for file in tqdm(files):
        if file.startswith('.'):
            continue
        if check_parent and lang_flag not in file:
            continue
        sub_dir = os.path.join(input_dir, file)
        if not os.path.isdir(sub_dir) or len(file) < 32:
            continue

        music_source_dir = os.path.join(sub_dir, lang_flag)
        print('music_source_dir:', music_source_dir)
        if not os.path.isdir(music_source_dir):
            continue
        sub_dir_files = os.listdir(music_source_dir)
        print('sub_dir_files:', len(sub_dir_files))
        new_sub_dir_files = []
        for music_file in sub_dir_files:
            if music_file.startswith('.'):
                continue
            if '.wav' not in music_file:
                continue
            file_prefix = music_file[:music_file.rindex('_')]
            if file_prefix in already_in_online2_file_list:
                print('already in online2 file list, file:', music_file)
                continue
            new_sub_dir_files.append(music_file)

        print('new_sub_dir_files:', len(new_sub_dir_files))
        count = len(new_sub_dir_files)
        if count > select_count:
            random.shuffle(new_sub_dir_files)
            new_sub_dir_files = new_sub_dir_files[:select_count]
        merge_wav_count += len(new_sub_dir_files)
        for music_file in new_sub_dir_files:
            music_src_file_path = os.path.join(music_source_dir, music_file)
            music_dest_file_path = os.path.join(music_dir, music_file)
            if os.path.exists(music_dest_file_path):
                continue
            shutil.copy(music_src_file_path, music_dest_file_path)
            # shutil.move(music_src_file_path, music_dest_file_path)
    return merge_wav_count


def pick_special_count_file(input_dir, output_dir=None, dest_count=1000, extend=None, shuffle=True, is_copy=True):
    if output_dir is None:
        output_dir = input_dir + '-pickout'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    files = os.listdir(input_dir)
    if shuffle:
        random.shuffle(files)

    copy_count = 0
    for file in tqdm(files):
        if file.startswith('.'):
            continue
        if extend is not None and file.endswith(extend):
            continue
        file_path = os.path.join(input_dir, file)
        if os.path.isdir(file_path):
            continue

        copy_count += 1
        file_path = os.path.join(input_dir, file)
        dest_file_path = os.path.join(output_dir, file)
        if is_copy:
            shutil.copy(file_path, dest_file_path)
        else:
            shutil.move(file_path, dest_file_path)
        if copy_count >= dest_count:
            break


def short_dir_file(input_dir, dest_count=1000, extend=None):
    files = os.listdir(input_dir)
    random.shuffle(files)
    file_paths = []
    for file in tqdm(files):
        if file.startswith('.'):
            continue
        if extend is not None and not file.endswith(extend):
            continue
        file_path = os.path.join(input_dir, file)
        if os.path.isdir(file_path):
            continue
        file_paths.append(file_path)

    if len(file_paths) > dest_count:
        file_paths = file_paths[dest_count:]
        for file_path in tqdm(file_paths):
            print('file_path:', file_path)
            os.remove(file_path)


def update_movie_list_file(list_file):
    with open(list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        url = line.replace('\n', '').split('\t')[0].split(' ')[0]
        md5_name = getMd5(url)
        new_line = '{}\t{}\n'.format(url, md5_name)
        new_lines.append(new_line)

    bak_file = list_file.replace('.', '_bak.')
    shutil.move(list_file, bak_file)
    with open(list_file, 'w+', encoding='utf-8') as f:
        f.writelines(new_lines)


def auto_flag_movie_language(movie_list_file_path, movie_wav_save_dir):
    with open(movie_list_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    md5_lines = []
    for index in range(len(lines)):
        line = lines[index]
        names = line.replace('\t', ' ').replace('\n', '').split(' ')
        md5_name = names[1]
        url = names[0]

        md5_lines.append(md5_name)
        lines[index] = '{} {}'.format(url, md5_name)

    files = os.listdir(movie_wav_save_dir)

    for file in files:
        file_path = os.path.join(movie_wav_save_dir, file)
        if not os.path.isdir(file_path):
            continue
        if language_flag_str not in file:
            continue
        names = file.split('_')
        if len(names) < 2:
            continue
        md5_name = names[0]
        if len(md5_name) != 32:
            continue
        language_id = int(file[-1])

        index = md5_lines.index(md5_name)
        lines[index] = '{} {}'.format(lines[index], language_id)

    bak_file = movie_list_file_path.replace('.', '_bak.')
    if os.path.exists(bak_file):
        os.remove(bak_file)

    shutil.move(movie_list_file_path, bak_file)
    with open(movie_list_file_path, 'w+', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


def remove_error_flag_file_and_dir(input_dir):
    files = os.listdir(input_dir)
    for file in files:
        if file.startswith('.'):
            continue
        if language_flag_str in file:
            file_path = os.path.join(input_dir, file)
            # 临时文件
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            # 降噪后的文件
            elif dnoise_flag in file:
                index = file.index(language_flag_str)
                replace_str = file[index:index + len(language_flag_str) + 1]
                new_file = file.replace(replace_str, '')
                new_file_path = os.path.join(input_dir, new_file)
                shutil.move(file_path, new_file_path)


def cal_result_diff(list_file_path, file_dir):
    with open(list_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    md5_language_map = {}
    mark_count = -1
    for line in lines[:mark_count]:
        names = line.replace('\n', '').split(' ')
        if len(names) < 2:
            continue
        print('line:', line)
        language = -1
        if len(names) > 2:
            language = int(names[2])
        md5_language_map[names[1]] = language

    files = os.listdir(file_dir)
    error_count = 0
    new_flag_count = 0
    total_count = 0

    for file in files:
        if file.startswith('.'):
            continue
        file_path = os.path.join(file_dir, file)
        if not os.path.isdir(file_path):
            continue
        if len(file) < 32:
            continue

        names = file.split('_')
        md5 = names[0]
        language = -1
        if len(names) > 1:
            language = int(file[-1])

        if md5 in md5_language_map:
            total_count += 1
            real_language = md5_language_map[md5]
            # TODO temp
            if language == -1 or language == 8 or real_language == -1:
                continue
            if language != real_language:
                # dest_file_path = os.path.join(file_dir, file.replace())
                if real_language != -1:
                    error_count += 1
                else:
                    new_flag_count += 1
                print('md5: {} , language predict:{}, real:{}'.format(md5, language, real_language))

    print('total_count:{}, error_count:{}, error_rate:{}, new_flag_count:{}'.format(total_count, error_count,
                                                                                    error_count / total_count,
                                                                                    new_flag_count))


def cut_audio(input_dir, output_dir, seconds=5):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    files = os.listdir(input_dir)
    for file in tqdm(files):
        if file.startswith('.'):
            continue
        if '.wav' not in file:
            continue
        if '_new.wav' in file:
            continue

        file_path = os.path.join(input_dir, file)
        resample_path = os.path.join(input_dir, file.replace('.wav', '_new.wav'))
        if os.path.exists(resample_path):
            continue

        os.system('ffmpeg -i {} -ac 1 -ar 44100 -y {}'.format(file_path, resample_path))
        if not os.path.exists(resample_path):
            continue
        sr, samples = wav.read(resample_path)

        total_byte = len(samples)
        per_byte = max(seconds * sr, int(total_byte / seconds))
        start = 0

        while start < total_byte:
            new_path = os.path.join(output_dir, file.replace('.wav', '_{}.wav'.format(start)))
            stop = min(start + per_byte, total_byte)
            if stop - start < 100 * 1024:
                break
            wav.write(new_path, sr, samples[start:stop])
            start = stop


def validate_segment_wav(input_dir, target_type=None, out_dir=None, check_speech=True, filter_check=True):
    dir_type_map = {}
    dir_type_map['zh'] = 1
    if check_speech:
        dir_type_map['ja'] = 1
    else:
        dir_type_map['ja'] = 3
    dir_type_map['bg'] = 0
    dir_type_map['mc'] = 8

    if target_type is None:
        for key in dir_type_map:
            if key in input_dir:
                target_type = dir_type_map[key]
                break

    files = os.listdir(input_dir)
    start = 0
    end = -1
    argv = sys.argv
    if len(argv) > 2:
        start = int(argv[1])
        end = int(argv[2])
        print('start:', start, ', end:', end, ', length:', len(files))
        if end > len(files):
            end = -1

    files = files[start:end]
    if target_type is None:
        target_type = int(input_dir[-1])

    check_file_list = []
    if filter_check:
        with open('record/online2_wav_check_file_list.txt', 'r', encoding='utf-8') as f:
            check_file_list = f.readlines()
            temp_list = []
            for file in check_file_list:
                file = file.replace('\n', '')
                temp_list.append(file)
            check_file_list = temp_list

    for file in tqdm(files):
        if '.wav' not in file:
            continue
        if file in check_file_list:
            print('already checked, file:', file)
            continue
        file_path = os.path.join(input_dir, file)
        sr, samples_data = wav.read(file_path)
        predicate_type = check_speech_or_lang_type(sr, samples_data, check_speech, threshold=0.6)
        if target_type != predicate_type:
            print('predicate type is not equal, predicate_type:', predicate_type, ',target_type:', target_type,
                  ',file:', file)
            temp_out_dir = out_dir
            if temp_out_dir is None:
                temp_out_dir = input_dir + '-error-' + str(predicate_type)

            if not os.path.isdir(temp_out_dir):
                os.makedirs(temp_out_dir)

            new_file_path = os.path.join(temp_out_dir, file)
            if os.path.exists(new_file_path):
                os.remove(new_file_path)
            shutil.move(file_path, new_file_path)
            # shutil.copy(file_path, new_file_path)


def merge_lang_segment_by_file_list(movie_file_path, input_dir, output_dir, target_lang, target_count=None,
                                    is_copy=True):
    with open(movie_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    target_lang_md5_list = []
    for line in lines:
        names = line.replace('\n', '').replace('\t', ' ').split(' ')
        if len(names) < 3:
            continue
        print('line:', line, ',lang:', str(names[2]))
        lang = int(names[-1])
        if lang != target_lang:
            continue
        target_lang_md5_list.append(names[1])

    files = os.listdir(input_dir)
    if target_count is not None and len(files) > target_count:
        random.shuffle(files)
        files = files[:target_count]

    for file in files:
        if '.wav' not in file:
            continue
        md5 = file.split('_')[0]
        if md5 not in target_lang_md5_list:
            continue
        file_path = os.path.join(input_dir, file)
        dest_file_path = os.path.join(output_dir, file)
        if is_copy:
            shutil.copy(file_path, dest_file_path)
        else:
            shutil.move(file_path, dest_file_path)


def find_and_remove_repeat_mp3_file(movie_file_list, input_dir, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    new_lines = []
    url_md5_lang_map = {}

    with open(movie_file_list, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    list.sort(lines, key=lambda x: len(x.split(' ')))
    for line in lines:
        line = line.replace('\t', ' ').replace('\n', '')
        for count in range(10):
            space_str = ' '
            for i in range(count):
                space_str += ' '
            line = line.replace(space_str, ' ')

        names = line.split(' ')
        url_md5 = names[1]
        if len(names) >= 3:
            url_md5_lang_map[url_md5] = names[2]
        new_lines.append(line)

    files = os.listdir(input_dir)
    file_id_name_map = {}
    list.sort(files, key=lambda x: os.path.getctime(os.path.join(input_dir, x)))
    for file in tqdm(files):
        if '.mp3' not in file:
            continue
        file_path = os.path.join(input_dir, file)
        file_size = os.path.getsize(file_path)
        if file_size <= 1000:
            continue
        with open(file_path, 'rb') as f:
            md5_obj = hashlib.md5()
            md5_obj.update(f.read(10000))
            file_md5 = md5_obj.hexdigest()
        file_id = str(file_size) + str(file_md5)

        url_md5_array = []
        if file_id in file_id_name_map:
            url_md5_array = file_id_name_map[file_id]

        url_md5 = file.split('.')[0]
        if url_md5 in url_md5_lang_map:
            url_md5_array.insert(0, url_md5)
        else:
            url_md5_array.append(url_md5)
        file_id_name_map[file_id] = url_md5_array

    need_remove_md5_list = []
    for real_md5 in file_id_name_map:
        url_md5_array = file_id_name_map[real_md5]
        if len(url_md5_array) <= 1:
            continue
        print('find, file array:', url_md5_array)
        need_remove_md5_list.extend(url_md5_array[1:])

    lines = new_lines
    new_lines = []
    for line in lines:
        names = line.split(' ')
        url_md5 = names[1]
        if url_md5 in need_remove_md5_list:
            mp3_name = url_md5 + '.mp3'
            mp3_path = os.path.join(input_dir, mp3_name)
            dest_mp3_path = os.path.join(output_dir, mp3_name)
            shutil.move(mp3_path, dest_mp3_path)
            continue
        new_lines.append(line + '\n')

    os.remove(movie_file_list)
    with open(movie_file_list, 'w+', encoding='utf-8') as f:
        f.writelines(new_lines)

    for real_md5 in file_id_name_map:
        url_md5_array = file_id_name_map[real_md5]
        if len(url_md5_array) <= 1:
            continue
        print('find, file array:', url_md5_array)
        # check language
        lang_arr = []
        for url_md5 in url_md5_array:
            lang = url_md5_lang_map[url_md5] if url_md5 in url_md5_lang_map else -1
            lang_arr.append(int(lang))

        first_lang = lang_arr[0]
        has_not_equal = False
        for lang in lang_arr[1:]:
            if first_lang != lang:
                has_not_equal = True
        if has_not_equal:
            print('fuck this url md5:', lang_arr)


def remove_repeat_segment(index_file, seg_dir, out_dir=None):
    with open(index_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    url_md5_record = []
    for line in lines:
        url_md5 = line.replace('\n', '').split(' ')[1]
        if len(url_md5) != 32:
            print('fuck this, url_md5:', url_md5, ',line:', line)
        url_md5_record.append(url_md5)

    if out_dir is None:
        out_dir = seg_dir + '-repeat'

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    seg_files = os.listdir(seg_dir)
    for seg_file in tqdm(seg_files):
        if '.wav' not in seg_file:
            continue
        url_md5 = seg_file.split('_')[0]
        if len(url_md5) != 32:
            print('fuck error, seg_file:', seg_file)
            continue
        if url_md5 not in url_md5_record:
            print('find url_md5:', url_md5)
            seg_file_path = os.path.join(seg_dir, seg_file)
            seg_dest_file_path = os.path.join(out_dir, seg_file)
            shutil.move(seg_file_path, seg_dest_file_path)


def export_wav_category_file(input_dir, output_file, flag_str='online2'):
    dir_wav_name_map = {}
    files = os.listdir(input_dir)
    for file in files:
        if flag_str not in file:
            continue
        file_path = os.path.join(input_dir, file)
        if not os.path.isdir(file_path):
            continue
        wav_files = os.listdir(file_path)
        key = file
        wav_file_list = []
        dir_wav_name_map[key] = wav_file_list
        for wav_file in wav_files:
            if wav_file.startswith('.'):
                continue
            wav_file_list.append(wav_file)

    json_str = json.dumps(dir_wav_name_map)
    with open(output_file, 'w+') as f:
        f.write(json_str)

    # with open(output_file, 'r') as f:
    #     map = json.load(f)
    #     print('map:', map)


def adjust_wav_file_path(map_file, base_dir):
    wav_map = np.load(map_file, allow_pickle=True).item()

    dir_path_wav_map = {}
    for dir in wav_map:
        dir_path = os.path.join(os.path.join(base_dir, dir), 'clips')
        if not os.path.isdir(dir_path):
            continue
        dir_path_wav_map[dir_path] = wav_map[dir]

    for dir_path in dir_path_wav_map:

        wav_files = os.listdir(dir_path)

        for wav_file in wav_files:
            if '.wav' not in wav_file:
                continue
            for dir_path_record in dir_path_wav_map:
                wav_record_files = dir_path_wav_map[dir_path_record]
                if wav_file in wav_record_files:
                    if dir_path_record != dir_path:
                        wav_file_path = os.path.join(dir_path, wav_file)
                        wav_dest_file_path = os.path.join(dir_path_record, wav_file)
                        print('find need adjust wav file:', wav_file, ',current dir:', dir_path, ',dest dir:',
                              dir_path_record)
                        # shutil.move(wav_file_path, wav_dest_file_path)
                    break


def delete_repeat_segment_wav(check_dir):
    md5_name_map = {}
    dirs = os.listdir(check_dir)
    for dir in dirs:
        dir_path = os.path.join(check_dir, dir)
        if not os.path.isdir(dir_path):
            continue
        files = os.listdir(dir_path)

        for file in files:
            file_path = os.path.join(dir_path, file)
            file_md5 = get_file_md5(file_path)

            file_arr = []
            if file_md5 in md5_name_map:
                file_arr = md5_name_map[file_md5]
            file_arr.append(file_path)
            md5_name_map[file_md5] = file_arr

    for file_md5 in md5_name_map:
        file_arr = md5_name_map[file_md5]
        if len(file_arr) > 1:
            print('find repeat:', file_arr)

            for file_path in file_arr[1:]:
                os.remove(file_path)

    # files = os.listdir(repeat_dir)
    # for file in files:
    #     file_path = os.path.join(repeat_dir, file)
    #     file_md5 = get_file_md5(file_path)
    #
    #     if file_md5 in check_wav_list:
    #         print('find repeat file:', file)
    #         # os.remove(file_path)


def merge_segment_by_file_list(record_file, parent_dir_path, output_dir=None, is_copy=False):
    with open(record_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    file_name_list = []
    for line in lines:
        file_name = line.replace('\n', '').split('?')[0].split('/')[-1]
        file_name_list.append(file_name)

    if output_dir is None:
        output_dir = os.path.join(parent_dir_path, 'output')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    dirs = os.listdir(parent_dir_path)
    for dir in dirs:
        if dir.startswith('.'):
            continue
        dir_path = os.path.join(parent_dir_path, dir)
        if not os.path.isdir(dir_path):
            continue
        if dir == 'output':
            continue
        files = os.listdir(dir_path)

        for file in tqdm(files):
            if '.wav' not in file:
                continue
            if file in file_name_list:
                file_path = os.path.join(dir_path, file)
                file_dest_path = os.path.join(output_dir, file)
                if os.path.exists(file_dest_path):
                    os.remove(file_dest_path)
                print('file_path:', file_path)
                if is_copy:
                    shutil.copy(file_path, file_dest_path)
                else:
                    shutil.move(file_path, file_dest_path)


def copy_download_mp3(moive_list_file, save_dir, new_dir):
    with open(moive_list_file, 'r', encoding='utf') as f:
        lines = f.readlines()
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for line in lines:
        url = line.replace('\n', '').split('\t')[0].split(' ')[0]
        md5 = getMd5(url)
        mp3_file_path = os.path.join(save_dir, md5 + '.mp3')
        exists = os.path.exists(mp3_file_path)
        if exists:
            print('mp3_file_path exists:', exists, ', mp3_file_path:', mp3_file_path)
            new_mp3_file_path = os.path.join(new_dir, md5 + '.mp3')
            shutil.move(mp3_file_path, new_mp3_file_path)


def delete_special_file(movie_mp3_save_dir, file_flag):
    files = os.listdir(movie_mp3_save_dir)
    for file in files:
        if file_flag in file:
            file_path = os.path.join(movie_mp3_save_dir, file)
            os.remove(file_path)


def copy_dir_to_dir(input_dir, output_dir, extend_name='.wav'):
    input_files = os.listdir(input_dir)
    for file in tqdm(input_files):
        if not file.endswith(extend_name):
            continue
        file_path = os.path.join(input_dir, file)
        dest_file_path = os.path.join(output_dir, file)
        shutil.move(file_path, dest_file_path)


def split_error_wav_check(input_file_path, split_count=5, no_equal_count=None):
    """ 将机器识别与人识别不一致的片段进行再拆分，看各个子片段推断结果与整个的差异 """
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if no_equal_count is None:
        no_equal_count = split_count

    for line in lines:
        splits = line.replace('\n', '').split(',')
        img_path = splits[0]
        h_flag = int(splits[1])
        m_flag = int(splits[2])

        if len(img_path.split('_')) < 6:
            continue

        image = Image.open(img_path)
        data = np.array(image)
        total_len = data.shape[1]
        step = int(total_len / 5)
        data = data[np.newaxis, np.newaxis, :]

        curr_no_equal_count = 0
        pre_flag = []
        for i in range(split_count):
            start = i * step
            end = min((i + 1) * step, total_len)
            part_data = data[:, :, :, start:end]
            part_m_flag = check_image_speech_or_lang_type(part_data, True, 0.1)
            pre_flag.append(part_m_flag)
            if part_m_flag != h_flag:
                curr_no_equal_count += 1
        if curr_no_equal_count >= no_equal_count and np.argmax(pre_flag) == np.argmin(
                pre_flag) and m_flag != 8 and h_flag != 8:
            print('no_equal_count:', curr_no_equal_count, ', find line:', line, ', pre_flags:', pre_flag)
            # vlue = line.replace('\n', '')
            # print(vlue)


if __name__ == '__main__':
    # http://30.211.97.239/ai/index.m3u8
    # with open('logs/lang500.txt', 'w', encoding='utf-8') as log_file:
    #     print('open')
    print('argv:', sys.argv)
    # print('listdir2', os.listdir('/ProjectRoot/Spoken-language-identification/dataset/online5-all-fuck'))
    movie_index = 0
    movie_list_file_path = '../movie/test_100_list.txt'
    movie_mp3_save_dir = '../dataset/test_100/mp3'
    movie_wav_save_dir = 'dataset/test_100/wav'
    movie_wav_flag_dir = 'dataset/test_100/flag'
    # copy_download_mp3(movie_list_file_path, movie_wav_input_dir, movie_wav_save_dir)
    download_top_movie(movie_list_file_path, movie_mp3_save_dir, '.mp3', processes=5)
    # download_and_split_movie(movie_list_file_path, movie_wav_save_dir, '.mp3')
    # split_error_wav_check('record/may_error_flag_all_nq_list.txt', no_equal_count=5)
    # split_error_wav_check('record/may_error_flag_special_list.txt', split_count=3, no_equal_count=None)

    # copy_dir_to_dir('/ProjectRoot/Spoken-language-identification/dataset/online5-all/flag/lang_0',
    # '/ProjectRoot/Spoken-language-identification/dataset/online5-all/flag/lang_1')

    # convert_audio_format('/ProjectRoot/Spoken-language-identification/dataset/online5-all/flag',
    # '/ProjectRoot/Spoken-language-identification/dataset/online5-all/flag-out-mp3', '.wav', '.mp3', False)

    # export_wav_category_file('/Users/bevis/dataset/online2-all', 'record/dir_wav_file_map2.json')
    # adjust_wav_file_path('dir_wav_file_map.npy', 'dataset')

    # split_movie(movie_mp3_save_dir, movie_wav_save_dir)
    # split_special_movie(movie_wav_save_dir, movie_list_file_path)

    # merge_wav_segment(movie_wav_save_dir)
    # merge_count = merge_special_lang_segment(movie_wav_save_dir, movie_wav_flag_dir, 8, select_count=15)
    # print('merge_count:', merge_count)
    # merge_lang_segment_by_file_list(movie_list_file_path, '/Users/bevis/dataset/online3-all/speech',
    #                                 '/Users/bevis/dataset/online3-all/mc-online3', 8, is_copy=False)
    # merge_segment_by_file_list('record/online5_1th_reflag_simple_flag.txt', '/ProjectRoot/Spoken-language-identification/dataset/online5-all/flag', is_copy=False)

    # short_dir_file(movie_wav_flag_dir+'/lang_8', dest_count=16000)
    # pick_special_count_file('/Users/bevis/dataset/online4-all/0', dest_count=5000, is_copy=False)
    # validate_segment_wav(movie_wav_save_dir, check_speech=True)

    # find_and_remove_repeat_mp3_file(movie_list_file_path, movie_mp3_save_dir, movie_mp3_save_dir + '-repeat')
    # remove_repeat_segment(movie_list_file_path, '/Users/bevis/dataset/online2-all/bg-online2')

    # input_dir = '/Users/bevis/Downloads/Data/mc-google'
    # output_dir = '/Users/bevis/Downloads/Data/mc-google-out'
    # cut_audio(input_dir, output_dir)

    # update_movie_list_file(movie_list_file_path)

    # auto_flag_movie_language(movie_list_file_path, movie_wav_save_dir)

    # print(getMd5('http://server/long1.mp3'))
    # cal_result_diff(movie_list_file_path, movie_wav_save_dir)

    # remove_error_flag_file_and_dir(movie_wav_save_dir)
    # delete_empty_wav('/Users/bevis/Downloads/bg-online')
    # delete_special_file(movie_mp3_save_dir, '_dnoise')
    # delete_repeat_segment_wav('/Users/bevis/dataset/online2-all')
    # is_empty = is_empty_wav('/Users/bevis/Downloads/bg-online/7bd6e78b8643c91df30a3a7b8ca1e76a_1_dnoise_new_52272000_new.wav', print_log=True)
    # print(is_empty)

    # command = 'ffmpeg -i {} -ac 1 {}'
    # exec_ffmpeg_command('dataset/movie_wav/0-ea', command, True)
    # url = 'dataset/online5-all-fuck/3307157bc0ba63b32adda2871111189a.mp3'
    # infer_movie_language(url, save_dir='dataset/online5-all-test', background_dir='dataset/movie_mp3/0')
    # sr, samples = wav.read(url)
    # predicate_type = check_speech_or_lang_type(sr, samples, True, threshold=0.6)

    # file_md5 = get_file_md5('/Users/bevis/Downloads/st_test_mp3/49fa_speech.mp3')
    # file2_md5 = get_file_md5('/Users/bevis/Downloads/st_test_mp3/49fa_speech_bak.mp3')
    # print(file_md5)
    # print(file2_md5)