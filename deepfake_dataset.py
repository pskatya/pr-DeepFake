import json
import os
from os import path

import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds
from absl import logging

# for regular run
from data.data_config import DATASET_CONFIG, RELATIVE_PATH, AUDIO_FILES_JSON, LABELS_PATH, DIRECTORY_NAME

SPLITS_PATH = ("{RELATIVE_PATH}/tmp/downloads/extracted/ZIP.train_test_split.zip")
VIDEOS_PATH = ("{RELATIVE_PATH}/tmp/downloads/extracted/ZIP.{DIRECTORY_NAME}_{split}.zip")
_LABELS_PATH = f"fakeout/data/labels.txt"


class DeepfakeConfig(tfds.core.BuilderConfig):

    def __init__(self, *, width=None, height=None, **kwargs):
        super(DeepfakeConfig, self).__init__(
            version=tfds.core.Version('2.0.0'),
            release_notes={
                '2.0.0': 'New split API (https://tensorflow.org/datasets/splits)',
            },
            **kwargs,
        )
        self.width = width
        self.height = height


class Deepfake(tfds.core.GeneratorBasedBuilder):
    def __init__(self, dataset_name, labels_path=None, train=True, audio_modality=False, **kwargs):
        self.audio_modality = audio_modality
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.train = train
        self.labels_path = labels_path
        self.dataset_config = DATASET_CONFIG[self.dataset_name]
        self.dataset_relative_path = self.dataset_config[RELATIVE_PATH]
        self.dataset_directory_name = self.dataset_config[DIRECTORY_NAME]
        if path.exists(os.path.join(self.dataset_relative_path, AUDIO_FILES_JSON)):
            with open(os.path.join(self.dataset_relative_path, AUDIO_FILES_JSON), 'rb') as f:
                self.proper_audio_files = json.load(f)
        else:
            self.proper_audio_files = None

    def _info(self):
        ffmpeg_extra_args = ('-qscale:v', '2', '-r', '29', '-t', '00:00:59')
        video_shape = (None, 224, 224, 3)
        labels_names_file = _LABELS_PATH
        if self.audio_modality:
            features = tfds.features.FeaturesDict({
                'video': tfds.features.Video(video_shape,
                                             ffmpeg_extra_args=ffmpeg_extra_args,
                                             encoding_format='jpeg'),
                'audio': tfds.features.Audio(file_format='wav', shape=(None, 2), dtype=tf.float32),
                'label': tfds.features.ClassLabel(names_file=labels_names_file),
                'path': tfds.features.Text()
            })
        else:  # audio modality not used
            features = tfds.features.FeaturesDict({
                'video': tfds.features.Video(video_shape,
                                             ffmpeg_extra_args=ffmpeg_extra_args,
                                             encoding_format='jpeg'),
                'label': tfds.features.ClassLabel(names_file=labels_names_file),
                'path': tfds.features.Text()
            })
        return tfds.core.DatasetInfo(
            builder=self,
            description='A deepfake classification dataset.',
            features=features,
        )

    def _split_generators(self, dl_manager):
        if not self.train:
            return [
                tfds.core.SplitGenerator(
                    name=tfds.Split.TEST,
                    gen_kwargs={
                        'videos_dir':
                            VIDEOS_PATH.format(RELATIVE_PATH=self.dataset_relative_path,
                                               DIRECTORY_NAME=self.dataset_directory_name,
                                               split='test'),
                        'splits_dir':
                            SPLITS_PATH.format(RELATIVE_PATH=self.dataset_relative_path),
                        'data_list':
                            'testlist.txt',
                    }),
            ]
        else:  # self.train == True
            return [
                tfds.core.SplitGenerator(
                    name=tfds.Split.TRAIN,
                    gen_kwargs={
                        'videos_dir':
                            VIDEOS_PATH.format(RELATIVE_PATH=self.dataset_relative_path,
                                               DIRECTORY_NAME=self.dataset_directory_name,
                                               split='train'),
                        'splits_dir':
                            SPLITS_PATH.format(RELATIVE_PATH=self.dataset_relative_path),
                        'data_list':
                            'trainlist.txt',
                    })
            ]

    def _generate_examples(self, videos_dir, splits_dir, data_list):
        data_list_path = os.path.join(splits_dir, data_list)
        with tf.io.gfile.GFile(data_list_path, 'r') as data_list_file:
            paths = data_list_file.readlines()
        files_in_dir = os.listdir(videos_dir)
        for path in sorted(paths):
            path = path.strip().split(' ')[0]
            video_path = os.path.join(videos_dir, path)
            if not tf.io.gfile.exists(video_path):
                logging.error(f'Example {video_path} not found')
                continue
            file_name = path.split('.')[0].split('@')[0].split('#')[0]
            if self.audio_modality:
                if self.train:
                    if self.proper_audio_files is not None:
                        if file_name in self.proper_audio_files:
                            if f"{file_name}.wav" not in files_in_dir:
                                raise Exception(
                                    f"{file_name} has no audio file although it should according to "
                                    f"the proper audio files list! You may set audio_modality to False "
                                    f"in order to use video modality only.")
                            audio_path = f"{os.path.join(videos_dir, file_name)}.wav"
                        else:  # proper audio does not exist, use dummy audio for video modality only
                            audio_path = "fakeout/data/dummy_audio.wav"
                    else:
                        if f"{file_name}.wav" in files_in_dir:
                            audio_path = f"{os.path.join(videos_dir, file_name)}.wav"
                        else:  # proper audio does not exist, use dummy audio for video modality only
                            audio_path = "fakeout/data/dummy_audio.wav"
                else:  # test
                    if f"{file_name}.wav" in files_in_dir:
                        audio_path = f"{os.path.join(videos_dir, file_name)}.wav"
                    else:  # proper audio does not exist, use dummy audio for video modality only
                        audio_path = "fakeout/data/dummy_audio.wav"
            if self.labels_path is None:
                VIDEO_LABELS_PD = pd.read_csv(os.path.join(self.dataset_relative_path, LABELS_PATH))
            else:  # self.labels_path is not None:
                VIDEO_LABELS_PD = pd.read_csv(self.labels_path)

            label = int(VIDEO_LABELS_PD[VIDEO_LABELS_PD['filename'] == file_name + '.mp4']['label'].iloc[0])

            if self.audio_modality:
                yield path, {'video': video_path,
                             'audio': audio_path,
                             'label': label,
                             'path': path}
            else:
                yield path, {'video': video_path,
                             'label': label,
                             'path': path}
