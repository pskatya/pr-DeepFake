from typing import Any, Dict, Optional

import haiku as hk
import jax
import jax.numpy as jnp

from models import normalization
from models import resnet
from models import s3d
from models import tsm_resnet


_DEFAULT_CFG_AUDTXT = {
    "totxt_head_mode": "linear",
    "toaud_head_mode": "linear",
    "toaud_bn_after_proj": False,
    "totxt_bn_after_proj": False,
    "embedding_dim": 512}

_DEFAULT_CFG_VIDAUD = {
    "tovid_head_mode": "linear",
    "toaud_head_mode": "mlp@512",
    "tovid_bn_after_proj": False,
    "toaud_bn_after_proj": True,
    "embedding_dim": 512}

_DEFAULT_CFG_VIDTXT = {
    "tovid_head_mode": "linear",
    "totxt_head_mode": "mlp@512",
    "tovid_bn_after_proj": False,
    "totxt_bn_after_proj": True,
    "embedding_dim": 512}

_DEFAULT_CFG_BN = {"decay_rate": 0.9, "eps": 1e-5,
                   "create_scale": True, "create_offset": True}


def _setkey_if_not_exists(d, key, value):
  if key not in d:
    d[key] = value


class AudioTextVideoEmbedding(hk.Module):

  def __init__(
      self,
      # Language parameters.
      word_embedding_matrix,
      sentence_dim=2048,
      # Audio parameters.
      audio_backbone="resnet18",
      audio_model_kwargs=None,
      # Vision parameters.
      visual_backbone="s3d",
      vision_model_kwargs=None,
      # Common parameters.
      mm_embedding_graph="fac_relu",
      use_xreplica_bn=True,
      bn_config_proj=None,
      config_video_text=None,
      config_video_audio=None,
      config_audio_text=None,
      use_audio_text=False,
      name="audio_text_video_model"):
   
    super(AudioTextVideoEmbedding, self).__init__(name=name)
    # Audio parameters.
    self._audio_backbone = audio_backbone
    self._audio_model_kwargs = audio_model_kwargs

    # Language parameters.
    self._sentence_dim = sentence_dim
    self._word_embedding_matrix = word_embedding_matrix

    # Vision parameters.
    self._visual_backbone = visual_backbone
    self._vision_model_kwargs = vision_model_kwargs

    # Joint parameters.
    self._use_xreplica_bn = use_xreplica_bn
    if self._use_xreplica_bn:
      self._normalizer_name = "cross_replica_batch_norm"
    else:
      self._normalizer_name = "batch_norm"

    # Projection head parameters.
    if config_video_text is None:
      config_video_text = _DEFAULT_CFG_VIDTXT
    for k, v in _DEFAULT_CFG_VIDTXT.items():
      _setkey_if_not_exists(config_video_text, k, v)
    self._cfg_vid_txt = config_video_text

    if config_video_audio is None:
      config_video_audio = _DEFAULT_CFG_VIDAUD
    for k, v in _DEFAULT_CFG_VIDAUD.items():
      _setkey_if_not_exists(config_video_audio, k, v)
    self._cfg_vid_aud = config_video_audio

    if config_audio_text is None:
      config_audio_text = _DEFAULT_CFG_AUDTXT
    for k, v in _DEFAULT_CFG_AUDTXT.items():
      _setkey_if_not_exists(config_audio_text, k, v)
    self._cfg_aud_txt = config_audio_text
    self._use_audio_text = use_audio_text

    self._mm_embedding_graph = mm_embedding_graph
    self._use_separate_heads = (
        mm_embedding_graph == "disjoint" or
        mm_embedding_graph.startswith("fac"))

    self._bn_config_proj = bn_config_proj or _DEFAULT_CFG_BN

  def _get_pair_embedding_heads(self,
                                embedding_dim_1, embedding_dim_2,
                                mode1, mode2,
                                use_bn_out1, use_bn_out2,
                                name1, name2):
    embd1_module = EmbeddingModule(
        embedding_dim_1,
        mode=mode1,
        use_bn_out=use_bn_out1,
        bn_config=self._bn_config_proj,
        use_xreplica_bn=self._use_xreplica_bn,
        name=name1)
    if self._use_separate_heads:
      embd2_module = EmbeddingModule(
          embedding_dim_2,
          mode=mode2,
          use_bn_out=use_bn_out2,
          use_xreplica_bn=self._use_xreplica_bn,
          bn_config=self._bn_config_proj,
          name=name2)
    else:
      assert embedding_dim_1 == embedding_dim_2, (
          "Using shared heads but inconsistent embedding dims where provided.")
      assert mode1 == mode2, (
          "Using shared heads but inconsistent modes where provided.")
      assert use_bn_out1 == use_bn_out2, (
          "Using shared heads but inconsistent bn conf where provided.")
      embd2_module = embd1_module
    return embd1_module, embd2_module

  def _activate_interaction(self, inputs, activation_fn, is_training,
                            activation_module=None):
    """Activation function for the interaction modules."""
    if activation_fn == "relu":
      inputs = jax.nn.relu(inputs)
    elif activation_fn == "bnrelu":
      if activation_module is None:
        activation_module = normalization.get_normalize_fn(
            normalizer_name=self._normalizer_name,
            normalizer_kwargs=self._bn_config_proj)
      inputs = activation_module(inputs, is_training=is_training)
      inputs = jax.nn.relu(inputs)
    else:
      raise ValueError(f"{activation_fn} not supported.")
    return inputs, activation_module

  def __call__(self,
               images,
               audio_spectrogram,
               word_ids,
               is_training,
               return_intermediate_audio=False):
   
    video_cnn = VisualModule(backbone=self._visual_backbone,
                             use_xreplica_bn=self._use_xreplica_bn,
                             model_kwargs=self._vision_model_kwargs)
    visual_representation = video_cnn(images, is_training=is_training)

    vid2txt_embd_module, vid2aud_embd_module = self._get_pair_embedding_heads(
        embedding_dim_1=self._cfg_vid_txt["embedding_dim"],
        embedding_dim_2=self._cfg_vid_aud["embedding_dim"],
        mode1=self._cfg_vid_txt["totxt_head_mode"],
        mode2=self._cfg_vid_aud["toaud_head_mode"],
        use_bn_out1=self._cfg_vid_txt["totxt_bn_after_proj"],
        use_bn_out2=self._cfg_vid_aud["toaud_bn_after_proj"],
        name1="vis_embd",
        name2="vid2audio_embd")

    video_embd = {}
    if self._mm_embedding_graph in ["shared", "disjoint"]:
      video_embd["toaud"] = vid2aud_embd_module(visual_representation,
                                                is_training=is_training)
      video_embd["totxt"] = vid2txt_embd_module(visual_representation,
                                                is_training=is_training)
    elif self._mm_embedding_graph.startswith("fac"):
      activation_fn = None
      if len(self._mm_embedding_graph.split("_")) == 2:
        activation_fn = self._mm_embedding_graph.split("_")[1]

      video_embd["toaud"] = vid2aud_embd_module(visual_representation,
                                                is_training=is_training)
      fine_rep = video_embd["toaud"]
      if activation_fn:
        fine_rep, activation_module = self._activate_interaction(
            inputs=fine_rep, activation_fn=activation_fn,
            is_training=is_training)

      video_embd["totxt"] = vid2txt_embd_module(fine_rep,
                                                is_training=is_training)
    else:
      raise ValueError(
          f"{self._mm_embedding_graph} is not a valid MM embedding graph.")

    # Computes the audio representation.
    audio_cnn = AudioModule(backbone=self._audio_backbone,
                            use_xreplica_bn=self._use_xreplica_bn,
                            model_kwargs=self._audio_model_kwargs)
    if return_intermediate_audio:
      return audio_cnn(audio_spectrogram,
                       is_training=is_training,
                       return_intermediate=True)

    audio_representation = audio_cnn(audio_spectrogram, is_training=is_training)

    # Projection heads: Audio -> Video and Audio -> Text.
    aud2vid_embd_module, aud2txt_embd_module = self._get_pair_embedding_heads(
        embedding_dim_1=self._cfg_vid_aud["embedding_dim"],
        embedding_dim_2=self._cfg_aud_txt["embedding_dim"],
        mode1=self._cfg_vid_aud["tovid_head_mode"],
        mode2=self._cfg_aud_txt["totxt_head_mode"],
        use_bn_out1=self._cfg_vid_aud["tovid_bn_after_proj"],
        use_bn_out2=self._cfg_aud_txt["totxt_bn_after_proj"],
        name1="audio_embd",
        name2="audio2txt_embd")
    audio_embd = {}

    audio_embd["tovid"] = aud2vid_embd_module(audio_representation,
                                              is_training=is_training)

    if (self._mm_embedding_graph.startswith("fac") and
        (self._use_audio_text or (not is_training))):
      fine_rep_audio = audio_embd["tovid"]
      if activation_fn:
        fine_rep_audio, _ = self._activate_interaction(
            inputs=fine_rep_audio, activation_fn=activation_fn,
            is_training=is_training, activation_module=activation_module)
      audio_embd["totxt"] = vid2txt_embd_module(fine_rep_audio,
                                                is_training=is_training)
    else:
      audio_embd["totxt"] = aud2txt_embd_module(audio_representation,
                                                is_training=is_training)

    txt_representation = TextModule(
        sentence_dim=self._sentence_dim,
        word_embedding_matrix=self._word_embedding_matrix)(
            word_ids, is_training=is_training)

    txt2vid_embd_module, txt2aud_embd_module = self._get_pair_embedding_heads(
        embedding_dim_1=self._cfg_vid_txt["embedding_dim"],
        embedding_dim_2=self._cfg_aud_txt["embedding_dim"],
        mode1=self._cfg_vid_txt["tovid_head_mode"],
        mode2=self._cfg_aud_txt["toaud_head_mode"],
        use_bn_out1=self._cfg_vid_txt["tovid_bn_after_proj"],
        use_bn_out2=self._cfg_aud_txt["toaud_bn_after_proj"],
        name1="txt_embd",
        name2="txt2audio_embd")
    txt_embd = {}
    txt_embd["tovid"] = txt2vid_embd_module(txt_representation,
                                            is_training=is_training)
    txt_embd["toaud"] = txt2aud_embd_module(txt_representation,
                                            is_training=is_training)

    return {
        "vid_embd": video_embd,
        "aud_embd": audio_embd,
        "txt_embd": txt_embd,
        "vid_repr": visual_representation,
        "aud_repr": audio_representation,
    }


class EmbeddingModule(hk.Module):
  """Final Embedding module."""

  def __init__(self,
               embedding_dim: int,
               mode: str = "linear",
               use_bn_out: bool = False,
               bn_config: Optional[Dict[str, Any]] = None,
               use_xreplica_bn: bool = True,
               name="embedding_module"):
    self._embedding_dim = embedding_dim
    self._use_bn_out = use_bn_out
    self._mode = mode
    # Set default BN config.
    bn_config = bn_config or _DEFAULT_CFG_BN
    if use_xreplica_bn:
      normalizer_name = "cross_replica_batch_norm"
    else:
      normalizer_name = "batch_norm"
    self._batch_norm = normalization.get_normalize_fn(
        normalizer_name=normalizer_name,
        normalizer_kwargs=bn_config)

    super(EmbeddingModule, self).__init__(name=name)

  def __call__(self, input_feature, is_training):
    if self._mode == "linear":
      proj = hk.Linear(self._embedding_dim, name="final_projection")
      embedding = proj(input_feature)
      inner_dims = [int(dim) for dim in self._mode.split("@")[1:]]
      embedding = input_feature
      for inner_dim in inner_dims:
        embedding = hk.Linear(inner_dim, with_bias=True,
                              name="final_projection_inner")(embedding)
        if not self._mode.startswith("mlp_nobn"):
          embedding = self._batch_norm(embedding, is_training=is_training)
        embedding = jax.nn.relu(embedding)

      # Final projection.
      embedding = hk.Linear(self._embedding_dim, name="final_projection",
                            with_bias=not self._use_bn_out)(embedding)
    else:
      raise NotImplementedError

    if self._use_bn_out:
      embedding = self._batch_norm(embedding, is_training=is_training)
    return embedding


class VisualModule(hk.Module):
  def __init__(self,
               use_xreplica_bn=True,
               backbone="s3d",
               model_kwargs=None,
               name="visual_module"):
    self._backbone = backbone
    super(VisualModule, self).__init__(name=name)
    if model_kwargs is None:
      model_kwargs = {}
    bn_config = model_kwargs.get("bn_config", _DEFAULT_CFG_BN)
    if use_xreplica_bn:
      normalizer_name = "cross_replica_batch_norm"
    else:
      normalizer_name = "batch_norm"

    normalize_fn = normalization.get_normalize_fn(
        normalizer_name=normalizer_name,
        normalizer_kwargs=bn_config)
    if backbone == "s3d":
      self._cnn = s3d.S3D(normalize_fn=normalize_fn)
    elif backbone == "resnet50tsm":
      width_mult = model_kwargs.get("width_mult", 1)
      self._cnn = tsm_resnet.TSMResNetV2(
          normalize_fn=normalize_fn,
          depth=50,
          num_frames=model_kwargs["n_frames"],
          width_mult=width_mult)
    else:
      raise NotImplementedError

  def __call__(self, images, is_training):
    """Connects graph to images."""
    features = self._cnn(images, is_training=is_training)
    return features


class AudioModule(hk.Module):

  def __init__(self,
               backbone="resnet18",
               use_xreplica_bn=True,
               model_kwargs=None,
               name="audio_module"):
    super(AudioModule, self).__init__(name=name)
    model_kwargs = model_kwargs or {}
    bn_config = model_kwargs.get("bn_config", _DEFAULT_CFG_BN)
    backbone_to_depth = {
        "resnet18": 18,
        "resnet34": 34,
        "resnet50": 50,
        "resnet101": 101
    }
    assert backbone in backbone_to_depth, (
        f"backbone should be in {backbone_to_depth.keys()}")

    if use_xreplica_bn:
      normalizer_name = "cross_replica_batch_norm"
    else:
      normalizer_name = "batch_norm"

    self._cnn = resnet.ResNetV2(
        depth=backbone_to_depth[backbone],
        normalize_fn=normalization.get_normalize_fn(
            normalizer_name=normalizer_name,
            normalizer_kwargs=bn_config),
        num_classes=None)

  def __call__(self,
               audio_spectrogram,
               is_training,
               return_intermediate=False):
    """Connects graph to audio spectrogram."""
    final_endpoint = "output"
    if return_intermediate:
      final_endpoint = "last_conv"

    return self._cnn(audio_spectrogram,
                     is_training=is_training,
                     final_endpoint=final_endpoint)


class TextModule(hk.Module):
  def __init__(self,
               word_embedding_matrix,
               sentence_dim=1024,
               name="text_module"):

    super(TextModule, self).__init__(name=name)
    self._word_embedding_module = hk.Embed(
        embedding_matrix=word_embedding_matrix)
    self._conv1d_module = hk.Conv1D(sentence_dim, 1, name="text_conv1")

  def __call__(self, word_ids, is_training):
    word_embeddings = self._word_embedding_module(word_ids)
    word_embeddings = jax.lax.stop_gradient(word_embeddings)
    output = self._conv1d_module(word_embeddings)
    output = jax.nn.relu(output)
    output = jnp.amax(output, axis=1)
    return output
