import json
import os
import pickle
from os import path
from flask import Flask, render_template, request, redirect, url_for
import os
from datetime import datetime
import json
from time import time as current_time
import importlib
from typing import Any, Dict
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from absl import app
from absl import flags
import config
from config import MLP_LAYERS_NAMES
from data.data_config import DATASET_FOLDER, DATASET_CONFIG, DATASET_MIDFIX, AUDIO_FILES_JSON, \
    RELATIVE_PATH, DIRECTORY_NAME
from data.data_utils import define_usable_audio, get_dummy_data, generate_dataset, union_part_videos, get_auc_score
from models import mm_embeddings
from utils.deepfake_dataset import DeepfakeConfig
import time


flags.DEFINE_string('checkpoint_path', '', 'The path to load the pre-trained weights from.')
flags.DEFINE_string('dataset_name', 'dfdc', 'The directory with the challenge dataset.')
flags.DEFINE_integer('eval_batch_size', 1, 'The batch size for evaluation.')
flags.DEFINE_integer('num_test_windows', 5, 'How many windows to average on during test.')
flags.DEFINE_integer('num_frames', 32, 'Number of video frames.')
flags.DEFINE_integer('mlp_first_layer_size', 6144, 'First layer size of MLP classifier.')
flags.DEFINE_integer('mel_filters', 80, 'Number of mel filters for audio modality.')
flags.DEFINE_bool('use_audio', True, 'Whether to use audio in current run or not')

FLAGS = flags.FLAGS


def mlp_builder(input_size, input=None):
    if input is None:
        input = jnp.zeros(input_size)

    first_layer = hk.Linear(
        name=MLP_LAYERS_NAMES[0],
        output_size=512,
        w_init=hk.initializers.TruncatedNormal(0.02)
    )(input)

    second_layer = hk.Linear(
        name=MLP_LAYERS_NAMES[1],
        output_size=128,
        w_init=hk.initializers.TruncatedNormal(0.02)
    )(first_layer)

    clf_logits = hk.Linear(
        name=MLP_LAYERS_NAMES[2],
        output_size=2,
        w_init=hk.initializers.TruncatedNormal(0.02)
    )(second_layer)

    return clf_logits


def helper_fn():
    clf_logits = mlp_builder(input_size=FLAGS.mlp_first_layer_size, input=None)
    return clf_logits

@app.route('/')
def index():
    return render_template('index.html')

def forward_fn(images: jnp.ndarray,
               audio_spectrogram: jnp.ndarray,
               word_ids: jnp.ndarray,
               is_training: bool,
               model_config: Dict[str, Any]):
    language_model_vocab_size = 65536
    word_embedding_dim = 300
    dummy_embedding_matrix = jnp.zeros(shape=(language_model_vocab_size,
                                              word_embedding_dim))

    module = mm_embeddings.AudioTextVideoEmbedding(
        **model_config,
        word_embedding_matrix=dummy_embedding_matrix)
    model_result = module(images=images,
                          audio_spectrogram=audio_spectrogram,
                          word_ids=word_ids,
                          is_training=is_training,
                          )

    if FLAGS.use_audio:
        clf_logits = mlp_builder(input_size=FLAGS.mlp_first_layer_size,
                                 input=jnp.concatenate((model_result['vid_repr'], model_result['aud_repr']), axis=1))
        return clf_logits, model_result['vid_repr'], model_result['aud_repr']
    else:
        clf_logits = mlp_builder(input_size=FLAGS.mlp_first_layer_size, input=model_result['vid_repr'])
        return clf_logits, model_result['vid_repr']


def load_model(FLAGS):
    with open(FLAGS.checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    params = checkpoint['params']
    state = checkpoint['state']
    return params, state

def main(argv):
    start_time = time.time()
    del argv
    model_config = config.get_model_config(FLAGS.checkpoint_path)
    params, state = load_model(FLAGS)
    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    dataset_configuration = DATASET_CONFIG.get(FLAGS.dataset_name, None)
    dataset_folder = dataset_configuration.get(DATASET_FOLDER)
    dset_config = DeepfakeConfig(name=FLAGS.dataset_name)
    test_ds = generate_dataset(dset_config,
                               dataset_configuration,
                               model_config,
                               dataset_name=FLAGS.dataset_name,
                               FLAGS=FLAGS,
                               train=False)
    dataset_path = os.path.join(dataset_folder, DATASET_MIDFIX,
                                f"ZIP.{dataset_configuration.get(DIRECTORY_NAME)}_test.zip")
    files_in_dir = os.listdir(dataset_path)
    dummy_audio, dummy_word_ids = get_dummy_data(FLAGS)

    proper_audio_files_path = os.path.join(dataset_configuration[RELATIVE_PATH], AUDIO_FILES_JSON)
    if path.exists(proper_audio_files_path):
        with open(proper_audio_files_path, 'rb') as f:
            proper_audio_files = json.load(f)
    else:
        proper_audio_files = None

    test_labels, files_names, test_softmax = [], [], []

    print("Testing...")
    for test_chunk in test_ds:
        test_chunk_tfds = tfds.as_numpy(test_chunk)
        if FLAGS.use_audio:
            audio = define_usable_audio(test_chunk_tfds, files_in_dir, FLAGS, proper_audio_files, train=False)
        else:
            audio = dummy_audio
        file_name = test_chunk_tfds['path'][0].decode('utf-8')
        output = forward.apply(
            params=params,
            state=state,
            images=test_chunk_tfds['video'],
            audio_spectrogram=audio,
            word_ids=dummy_word_ids,
            is_training=False,
            model_config=model_config,
        )
        logits = output[0]
        softmax = jax.nn.softmax(logits[0])
        test_softmax.append(softmax)
        test_labels.append(test_chunk_tfds['label'])
        files_names.append(np.array([file_name]))
    test_softmax = np.concatenate(test_softmax, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    files_names = np.concatenate(files_names, axis=0)

    pred_test_reshaped = np.reshape(test_softmax[:, 1], (len(test_labels), -1, FLAGS.num_test_windows))
    pred_test_smoothed = pred_test_reshaped[:, :, 0:FLAGS.num_test_windows].mean(axis=2)
    pred_test_smoothed = pred_test_smoothed[:, 0]
    results_pd = pd.DataFrame(zip(pred_test_smoothed, test_labels, files_names),
                              columns=['softmax', 'label', 'filename'])
    results_pd = union_part_videos(results_pd)
    softmax = results_pd['softmax']
    labels = results_pd['label']
    return labels, softmax

def accuracy (labels, softmax):
    return round(get_auc_score(labels, softmax) * 100, 1)

def upload():
    file = request.files['file']

    if file:
        timestamp = int(current_time())
        filename = f"uploaded_video_{timestamp}.mp4"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        video_path2 = os.path.join(app.config['UPLOAD_FOLDER'], "1" + filename)
        result_from_det = accuracy(video_path ,video_path2)
        print(result_from_det)

        video_info = {
            'name': file.filename,
            'size': f"{os.path.getsize(video_path) / (1024):.2f} KB",
            'source': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'per': result_from_det
        }

        video_info_json = json.dumps(video_info)
        return redirect(url_for('result', video_info=video_info_json, video_path2=video_path2))

@app.route('/result')
def result():
    video_info_json = request.args.get('video_info')
    video_path2 = request.args.get('video_path2')  
    print(video_path2)

    video_info = json.loads(video_info_json)
    print(video_info['name'])

    return render_template('result.html', video_url=video_path2, video_info=video_info)

if __name__ == '__main__':
    app.run(debug=True)