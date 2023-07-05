# Audio-LID
Automatically detect the language of the gaven audio url 


# Environment

We suggest use conda to create environment, link:https://www.anaconda.com .

installing anaconda first, then:
1. conda create -n AudioLID python=3.9
2. conda activate AudioLID


# Dependency Introduction

1. gcc >=7.1 (C++17) 
2. ssl >= 1.1
3. python >= 3.9
4. PyTorch >= 1.10.0

OS support: Linux, Mac OSX

# Installation Dependency

pip install git+https://github.com/facebookresearch/fairseq.git@a29952ce6d313a4daf3e90647f8bf84cc6d4df6d#egg=fairseq
   
If has any network problems，you can clone it and install by zip file  
1.git clone https://github.com/facebookresearch/fairseq.git  
2.cd fairseq  
3.git checkout a29952ce6d313a4daf3e90647f8bf84cc6d4df6d  
4.cd ..  
5.tar -czvf fairseq.tar.gz fairseq/  
6.pip install fairseq.tar.gz (speed up: pip install fairseq.tar.gz -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn)

# Installation

Using source code:
1. git clone https://github.com/androidshu/audio-lid.git
2. pip install -r requirements.txt

Using pip:
1. pip install git+https://github.com/androidshu/audio-lid.git

## For nvidia-gpu 
1. You need to check and install the GPU driver.
2. Checking the CUDA version by typing 'nvidia-smi' on the command line 
3. Installing pytorch-cuda through the website link:https://pytorch.org
4. pip install paddlepaddle-gpu==2.5.0
5. install cuda+cudnn and config the paths. (must > 11.4. refer to:https://blog.csdn.net/qq_43183860/article/details/127657306)

# Pretrain files  
model: https://dl.fbaipublicfiles.com/mms/lid/mms1b_l126.pt

language dict:https://dl.fbaipublicfiles.com/mms/lid/dict/l126/dict.lang.txt

1. make dir /your_path/pretrain and put the two files in it.
2. language_model=/your_path/pretrain/mms1b_l126.pt
3. lang_dict_dir=/your_path/pretrain

# Getting Started

In code:  
`//import`  
`from audio_lid import AudioLID`

`//init audio lid`  
`lid = AudioLID(language_model='your model path', lang_dict_dir='your lang dict dir', debug=True,
output_path='The output dir use to save result files')`

`//infer`  
`ret, language_list = lid.infer_language(audio_file_path_1)`  
`ret, language_list = lid.infer_language(audio_file_path_2) #`
`more `


## In command line(When git clone):   
python3 audio_lid.py --audio-file "your audio file path" --lang-dict-dir "your lang dict dir" 
                    --language-model "your model path" --debug True --output-dir "The output dir use to save result files"

## In command line(When pip install):   
audio_lid --audio-file "your audio file path" --lang-dict-dir "your lang dict dir" 
                    --language-model "your model path" --debug True --output-dir "The output dir use to save result files"

## Result value explain

ret: The result, if big then zero mean successful, otherwise error,
      the error code refer to error_codes.py

language_list: the language list inferred by the given audio file is sorted by score  
     format likes: [('eng', 90.0), ('ch', 10.0)]   
     or [('eng', 100.0)],   
     the total score always equal 100.  
     The language short name like 'eng' to full name map:https://github.com/androidshu/audio-lid/blob/main/audio_lid/pretrain/language.txt

