# yo-xing.github.io
## Final Project for MUSIGR6610 - SOUND: ADVANCED TOPICS at Columbia University

## Investigating the role of Artificial Intelligence in Music with JukeBox and the Amen Break
### Yo Jeremijenko-Conley  

#### Motivation
In recent months, the role of Artificial Intelligence (AI) in art has become increasingly prevalent. Software like DALL-E 2, MidJourney, and DiffusionBee have enabled users to generate images based on textual descriptions. This software uses generative AI to create brand-new images (or artworks) that had previously never existed, based on datasets of millions of existing images/artwork. These software services have slightly different implementations, but their overall structure is relatively similar. For example, The DALL-E 2 model works as follows: 

First, the text prompt given by the user is fed into a trained text encoder model which maps the prompt into a representation space. After that, an OpenAI model called CLIP (Contrastive Language-Image Pre-Training) maps the representation space of the text encoding to a corresponding image representation space, CLIP is trained on hundreds of millions of images with associated captions, and learns how related any given piece of text is to an image. This captures the semantic information of the text prompt. A model called the prior then takes the CLIP embedding and generates a specific image embedding from it, you can think of it as forming a “mental image” of the text. Lastly, a diffusion model is used to stochastically generate the image by removing noise (or upsampling) from the image embedding. Diffusion models are trained with existing images with gaussian noise added to them iteratively by a Markov chain, which eventually leaves the image as nothing but noise. The diffusion model then learns how to iteratively reverse this noise to generate a (low-resolution) image from seemingly random noise. However, this general diffusion process does not correspond to a specific text prompt. To fix this, DALL-E 2 uses a modified version of their diffusion model (called GLIDE) which incorporates projected CLIP text embeddings into this process. At each time step in the iterative process of noise removal, the CLIP text embeddings are added to the image embedding at that specific time-step to “guide” the removal of noise in the direction of the texts representation space, classifier free guidance is also used to further guide the image towards the text prompt. 

![1*_wtvEU05qyDISFBT4xRVgw](https://user-images.githubusercontent.com/40434203/208201013-bfaeac92-8ea9-4773-82b0-4c582fcfee48.png)

  
  The results of these machine learning processes have been incredibly impressive in their ability to capture and translate semantic information and their grasp of aesthetics and art styles. This has sparked a discourse on whether or not these models are exploitative of artists whose work is being used to train these models, and among other things, their potential to generate artwork in other mediums, eventually becoming a cheap alternative to replace creative professionals. After experiencing the capabilities of these models, I viewed them as a “proof of concept” to creative applications of AI in the arts. This made me consider the possibilities of this technology being used to generate music, as well as the cultural and economic implications of this. In this project, I attempt to examine the current state of music-generating AI as well as speculate on its future capabilities and impact on the world. 
Some may assume that AI models would be better able to generate music and capture its stylistic patterns than those in paintings and visual arts due to the inherent mathematical structure of the basis of most music (such as the diatonic scale). However, the current capabilities of AI models to generate music are far behind their capability to generate visual art in the form of images. This is mainly due to two reasons, the first being the sequential nature of music, the interpretation of each tone or sound in a song is reliant on those that came before it and necessary for the interpretation of later sounds in the piece. While this can also be said for images, the non-sequential nature of how we experience them allows us to discount specific sections of an image easier than parts of a song. The second reason for the disparity between AI-generated images and music is the immense size of an audio file compared to an image, this makes encoding and decoding audio vastly more computationally expensive than doing so for an image. 

#### OpenAI’s JukeBox

To better understand the current capabilities and limitations of AI in music generation, I decided to experiment with JukeBox –a neural net that generates music–, made by OpenAI (the company responsible for DALL-E 2 and ChatGPT). While algorithmic composition has existed for many years, JukeBox is different in that it models music as raw audio rather than symbolic generators that use MIDI files or musical scores. This allows JukeBox to capture the subtle timbres and dynamics found in music as well as the human voice, however, the use of raw audio also gives the models an extremely high computational cost. For instance, a 16-bit 4-minute song at 44 kHz has over 10 million timesteps. To address this problem high dimensional input problem, JukeBox uses an autoencoder to compress raw audio to a lower-dimensional space, while attempting to only preserve the relevant information in the audio. JukeBox’s prior models can then generate audio in this compressed, lower-dimenstional space (similar to DALL-E 2’s prior for generating images) and then transformer models can be used to upsample the compressed audio to the raw audio space. JukeBox’s encoding approach uses VQ-VAE (Vector Quantised-Variational AutoEncoder). However, many VQ-VAE models suffer from codebook collapse, to address this, JukeBox uses a more simplified architecture called VQ-VAE-2 which only uses feed-foward encoders and decoders. However they modify this architecture by adding in random restarts of the codebook, to further prevent codebook collapse, as well as spectral loss which allows easier reconstruction of high frequencies. JukeBox’s VQ-VAE compresses raw audio (44kHz) at three levels, by factors of 8x, 32x, and 128x, with codebook size of 2048 for all levels. This compression losses a great deal of the audio detail, and sounds noticeably more noisy/scratchy as you go down levels, however the important structural information of the music such as pitch, timbre and volume is kept at all levels. 
The prior models are then trained to learn the distribution of the music encoded by the VQ-VAE and generate music in this compressed space. Similar to the VQ-VAE models, there are three levels of priors, a top level prior that generates the music in the most compressed space, and then two priors that upsample, generating less compressed audio conditional on the top level prior’s output. The top level prior is essential for modeling the overall structure of the music, and has very low audio quality, but still captures basic semantics such as the melody or voices. The middle and bottom upsampling priors add more local musical semantics like timbre, and improves the overall quality of the audio. Attached below is the output of a generated sample at each of the three levels:




[top level](https://drive.google.com/file/d/1AHoF42S0TAQgeaL0c2aoZpCZsF9uHMp8/view?usp=sharing)

[middle level](https://drive.google.com/file/d/1WCSQKTXc-Ev_c-stTAAQS5y0snd5o4az/view?usp=share_link)

[bottom level](https://drive.google.com/file/d/1Zl4fHxYXYAq7qJ1KXULl_oNxyb5ogp39/view?usp=share_link)


These prior models are trained using Sparse transformers, each containing 72 layers of factorization over a context of 8192 codes, for the top level this corresponds to 24 seconds of audio, while it corresponds to 6 seconds on the middle level prior and 1.5 seconds on the bottom level.
	To curate the dataset for training JukeBox, OpenAI crawled the web and gathered 1.2 million songs, half of which were in english, along with corresponding lyrics and metadata such as genre, album, artist, mood and release date from LyicWiki. It is unlikely that artists were asked for consent/permission to use thier songs, or the associated metadata. OpenAI is not currently making any profit directly from JukeBox as it is free open source software, however it is possible that they profit from later versions of the software, or that third party individuals are currently making profit using the software. 
	Furthermore, unlike software such as DALL-E 2 where the model’s input is an open ended text string, JukeBox is conditioned on directly on preset genre and artist tokens as well as a ‘promt’ wav file, for the model to expand on (it is also possible to generate audio in ‘ancestral’ mode without a wav input, but in my experience it performed much worse than when in ‘primed’ mode). Both genre and artist information is optional and not needed to generate music, however providing both gives an advantage by reducing the entropy of the audio generation which generally gives better results and a more consistent style throughout, this also makes the genre output more deterministic compared to generation without this information, however there is still a large amount of variance between outputs when given the same input. It is also possible to condition the model on lyrics in the form of text, however I will not be using it for the purposes of this project. 


#### The Amen Break

To investigate JukeBox’s understanding and interpretation of different genres and artists, I decided to produce a number of samples using the same audio input, while changing the genre and artist parameters and holding all else equal. I decided to use the Amen Break as the audio input for the model. 

[The Amen Break - Six second sample](https://drive.google.com/file/d/1e3xXfnYlhkfcvIXA-MORcVo5q_Z1QUto/view?usp=share_link)


Originally released as a part of "Color Him Father" by the soul group the Winstons, the Amen Break is a widely sampled drum break that has been used in thousands of songs in many different genres, it is believed to be one of the most sampled recordings in history. It appears in songs such as "Straight Outta Compton" by N.W.A and "Keep It Going Now" by Rob Base and even the Futurama theme song. I thought that this would make it a very interesting input audio file, as it is likely included in a large portion of the songs that JukeBox was trained on. Additionally, it is an example of the social problem of sampling and plagiarism in the music industry as a whole, while generative AI like JukeBox will only increase the complexity of these issues in the future. Sadly, the Amen Break's performer, Gregory Coleman, received no royalties for the sample, despite it being widely used, and passed away homeless in 2006, unaware of the astounding impact he had made on music. 


``` python:
import jukebox
import torch as t
import librosa
import os
from IPython.display import Audio
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, \
                           sample_partial_window, upsample, \
                           load_prompts
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache
# MPI Connect. MPI doesn't like being initialized twice, hence the following
try:
    if device is not None:
        pass
except NameError:
    rank, local_rank, device = setup_dist_from_mpi()

model = "5b" #@param ["5b_lyrics", "5b", "1b_lyrics"]
if model == '5b':
  your_lyrics = """
  """

save_and_load_models_from_drive = False

#START GDRIVE MODEL LOADER
if save_and_load_models_from_drive == True:
  import os.path
  !apt install pv
  !mkdir /root/.cache ; mkdir /root/.cache/jukebox ; mkdir /root/.cache/jukebox/models ; mkdir /root/.cache/jukebox/models/1b_lyrics ; mkdir /root/.cache/jukebox/models/5b_lyrics ; mkdir /root/.cache/jukebox/models/5b
  !mkdir /content/gdrive/MyDrive/jukebox
  !mkdir /content/gdrive/MyDrive/jukebox/models
  !mkdir /content/gdrive/MyDrive/jukebox/models/5b
  !mkdir /content/gdrive/MyDrive/jukebox/models/5b_lyrics
  !mkdir /content/gdrive/MyDrive/jukebox/models/1b_lyrics


def load_5b_vqvae():
    if os.path.exists("/root/.cache/jukebox/models/5b/vqvae.pth.tar") == False:
      if os.path.exists("/content/gdrive/MyDrive/jukebox/models/5b/vqvae.pth.tar") == False:
        print("5b_vqvae not stored in Google Drive. Downloading for the first time.")
        !wget https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar -O /content/gdrive/MyDrive/jukebox/models/5b/vqvae.pth.tar
      else:
        print("5b_vqvae stored in Google Drive.")
      print('Copying 5b VQVAE')
      !pv /content/gdrive/MyDrive/jukebox/models/5b/vqvae.pth.tar > /root/.cache/jukebox/models/5b/vqvae.pth.tar

def load_1b_lyrics_level2():
  if os.path.exists("/root/.cache/jukebox/models/1b_lyrics/prior_level_2.pth.tar") == False:
    if os.path.exists("/content/gdrive/MyDrive/jukebox/models/1b_lyrics/prior_level_2.pth.tar") == False:
      print("1b_lyrics_level_2 not stored in Google Drive. Downloading for the first time. This will take a few more minutes.")
      !wget https://openaipublic.azureedge.net/jukebox/models/1b_lyrics/prior_level_2.pth.tar -O /content/gdrive/MyDrive/jukebox/models/1b_lyrics/prior_level_2.pth.tar
    else:
      print("1b_lyrics_level_2 stored in Google Drive.")
    print("Copying 1B_Lyrics Level 2")
    !pv /content/gdrive/MyDrive/jukebox/models/1b_lyrics/prior_level_2.pth.tar > /root/.cache/jukebox/models/1b_lyrics/prior_level_2.pth.tar

def load_5b_lyrics_level2():
  if os.path.exists("/root/.cache/jukebox/models/5b_lyrics/prior_level_2.pth.tar") == False:
    if os.path.exists("/content/gdrive/MyDrive/jukebox/models/5b_lyrics/prior_level_2.pth.tar") == False:
      print("5b_lyrics_level_2 not stored in Google Drive. Downloading for the first time. This will take up to 10-15 minutes.")
      !wget https://openaipublic.azureedge.net/jukebox/models/5b_lyrics/prior_level_2.pth.tar -O /content/gdrive/MyDrive/jukebox/models/5b_lyrics/prior_level_2.pth.tar
    else:
      print("5b_lyrics_level_2 stored in Google Drive.")
    print("Copying 5B_Lyrics Level 2")
    !pv /content/gdrive/MyDrive/jukebox/models/5b_lyrics/prior_level_2.pth.tar > /root/.cache/jukebox/models/5b_lyrics/prior_level_2.pth.tar

def load_5b_level1():
  if os.path.exists('/root/.cache/jukebox/models/5b/prior_level_1.pth.tar') == False:
    if os.path.exists("/content/gdrive/MyDrive/jukebox/models/5b/prior_level_1.pth.tar") == False:
      print("5b_level_1 not stored in Google Drive. Downloading for the first time. This may take a few more minutes.")
      !wget https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_1.pth.tar -O /content/gdrive/MyDrive/jukebox/models/5b/prior_level_1.pth.tar
    else:
      print("5b_level_1 stored in Google Drive.")
    print("Copying 5B Level 1")
    !pv /content/gdrive/MyDrive/jukebox/models/5b/prior_level_1.pth.tar > /root/.cache/jukebox/models/5b/prior_level_1.pth.tar

def load_5b_level0():
  if os.path.exists('/root/.cache/jukebox/models/5b/prior_level_0.pth.tar') == False:
    if os.path.exists("/content/gdrive/MyDrive/jukebox/models/5b/prior_level_0.pth.tar") == False:
      print("5b_level_0 not stored in Google Drive. Downloading for the first time. This may take a few minutes.")
      !wget https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_0.pth.tar -O /content/gdrive/MyDrive/jukebox/models/5b/prior_level_0.pth.tar
    else:
      print("5b_level_0 stored in Google Drive.")
    print("Copying 5B Level 0")
    !pv /content/gdrive/MyDrive/jukebox/models/5b/prior_level_0.pth.tar > /root/.cache/jukebox/models/5b/prior_level_0.pth.tar

def load_5b_level2():
  if os.path.exists('/root/.cache/jukebox/models/5b/prior_level_2.pth.tar') == False:
    if os.path.exists("/content/gdrive/MyDrive/jukebox/models/5b/prior_level_2.pth.tar") == False:
      print("5b_level_2 not stored in Google Drive. Downloading for the first time. This will take up to 10-15 minutes.")
      !wget https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar -O /content/gdrive/MyDrive/jukebox/models/5b/prior_level_2.pth.tar
    else:
      print("5b_level_2 stored in Google Drive.")
  print("Copying 5B Level 2")
  !pv /content/gdrive/MyDrive/jukebox/models/5b/prior_level_2.pth.tar > /root/.cache/jukebox/models/5b/prior_level_2.pth.tar

if save_and_load_models_from_drive == True:
  if model == '5b_lyrics':
    load_5b_vqvae()
    load_5b_lyrics_level2()
    load_5b_level1()
    load_5b_level0()
  if model == '5b':
    load_5b_vqvae()
    load_5b_level2()
    load_5b_level1()
    load_5b_level0()
  elif model == '1b_lyrics':
    load_5b_vqvae()
    load_1b_lyrics_level2()
    load_5b_level1()
    load_5b_level0()
#END GDRIVE MODEL LOADER

hps = Hyperparams()
hps.sr = 44100
hps.n_samples =  2#@param {type:"integer"}
# Specifies the directory to save the sample in.
# We set this to the Google Drive mount point.
hps.name = '/content/gdrive/MyDrive/musicAI/musicAI20' #@param {type:"string"}
chunk_size = 16 if model in ('5b', '5b_lyrics') else 32
gpu_info = !nvidia-smi -L
if gpu_info[0].find('Tesla T4') >= 0:
  max_batch_size = 2
  print('Tesla T4 detected, max_batch_size set to 2')
elif gpu_info[0].find('Tesla K80') >= 0:
  max_batch_size = 8
  print('Tesla K80 detected, max_batch_size set to 8')
elif gpu_info[0].find('Tesla P100') >= 0:
  max_batch_size = 3
  print('Tesla P100 detected, max_batch_size set to 3')
elif gpu_info[0].find('Tesla V100') >= 0:
  max_batch_size = 3
  print('Tesla V100 detected, max_batch_size set to 3')
else:
  max_batch_size = 3
  print('Different GPU detected, max_batch_size set to 3.')
hps.levels = 3
speed_upsampling = False #@param {type: "boolean"}
if speed_upsampling == True:
  hps.hop_fraction = [1,1,.125]
else:
  hps.hop_fraction = [.5,.5,.125]

vqvae, *priors = MODELS[model]
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = 1048576)), device)
top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

# The default mode of operation.
# Creates songs based on artist and genre conditioning.
mode = 'primed' #@param ["ancestral", "primed"]
if mode == 'ancestral':
  codes_file=None
  audio_file=None
  prompt_length_in_seconds=None
if mode == 'primed':
  codes_file=None
  # Specify an audio file here.
  audio_file = '/content/gdrive/MyDrive/musicAI/Amen-break.wav' #@param {type:"string"}
  # Specify how many seconds of audio to prime on.
  prompt_length_in_seconds=6 #@param {type:"integer"}

sample_length_in_seconds = 24 #@param {type:"integer"}

if os.path.exists(hps.name):
  # Identify the lowest level generated and continue from there.
  for level in [0, 1, 2]:
    data = f"{hps.name}/level_{level}/data.pth.tar"
    if os.path.isfile(data):
      codes_file = data
      if int(sample_length_in_seconds) > int(librosa.get_duration(filename=f'{hps.name}/level_2/item_0.wav')):
        mode = 'continue'
      else:
        mode = 'upsample'
      break

print('mode is now '+mode)
if mode == 'continue':
  print('Continuing from level 2')
if mode == 'upsample':
  print('Upsampling from level '+str(level))

sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))

if mode == 'upsample':
  sample_length_in_seconds=int(librosa.get_duration(filename=f'{hps.name}/level_{level}/item_0.wav'))
  data = t.load(sample_hps.codes_file, map_location='cpu')
  zs = [z.cpu() for z in data['zs']]
  hps.n_samples = zs[-1].shape[0]

if mode == 'continue':
  data = t.load(sample_hps.codes_file, map_location='cpu')
  zs = [z.cpu() for z in data['zs']]
  hps.n_samples = zs[-1].shape[0]

hps.sample_length = (int(sample_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

# Note: Metas can contain different prompts per sample.
# By default, all samples use the same prompt.

select_artist = "pink_floyd" #@param {type:"string"}
select_genre = "psychedelic" #@param {type:"string"}
metas = [dict(artist = select_artist,
            genre = select_genre,
            total_length = hps.sample_length,
            offset = 0,
            lyrics = your_lyrics, 
            ),
          ] * hps.n_samples
labels = [None, None, top_prior.labeller.get_batch_labels(metas, 'cuda')]

sampling_temperature = .96 #@param {type:"number"}

if gpu_info[0].find('Tesla T4') >= 0:
  lower_batch_size = 12
  print('Tesla T4 detected, lower_batch_size set to 12')
elif gpu_info[0].find('Tesla K80') >= 0:
  lower_batch_size = 8
  print('Tesla K80 detected, lower_batch_size set to 8')
elif gpu_info[0].find('Tesla P100') >= 0:
  lower_batch_size = 16
  print('Tesla P100 detected, lower_batch_size set to 16')
elif gpu_info[0].find('Tesla V100') >= 0:
  lower_batch_size = 16
  print('Tesla V100 detected, lower_batch_size set to 16')
else:
  lower_batch_size = 8
  print('Different GPU detected, lower_batch_size set to 8.')
lower_level_chunk_size = 32
chunk_size = 16 if model in ('5b', '5b_lyrics') else 32
sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=lower_batch_size,
                        chunk_size=lower_level_chunk_size),
                    dict(temp=0.99, fp16=True, max_batch_size=lower_batch_size,
                         chunk_size=lower_level_chunk_size),
                    dict(temp=sampling_temperature, fp16=True, 
                         max_batch_size=max_batch_size, chunk_size=chunk_size)]

if sample_hps.mode == 'ancestral':
  zs = [t.zeros(hps.n_samples,0,dtype=t.long, device='cpu') for _ in range(len(priors))]
  zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
elif sample_hps.mode == 'upsample':
  assert sample_hps.codes_file is not None
  # Load codes.
  data = t.load(sample_hps.codes_file, map_location='cpu')
  zs = [z.cpu() for z in data['zs']]
  assert zs[-1].shape[0] == hps.n_samples, f"Expected bs = {hps.n_samples}, got {zs[-1].shape[0]}"
  del data
  print('One click upsampling!')
elif sample_hps.mode == 'primed':
  assert sample_hps.audio_file is not None
  audio_files = sample_hps.audio_file.split(',')
  duration = (int(sample_hps.prompt_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
  x = load_prompts(audio_files, duration, hps)
  zs = top_prior.encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
  zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
elif sample_hps.mode == 'continue':
  data = t.load(sample_hps.codes_file, map_location='cpu')
  zs = [z.cuda() for z in data['zs']]
  zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
else:
  raise ValueError(f'Unknown sample mode {sample_hps.mode}.')

# Set this False if you are on a local machine that has enough memory (this allows you to do the
# lyrics alignment visualization during the upsampling stage). For a hosted runtime, 
# we'll need to go ahead and delete the top_prior if you are using the 5b_lyrics model.
if True:
  del top_prior
  empty_cache()
  top_prior=None
upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1]]
labels[:2] = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in upsamplers]

zs = upsample(zs, labels, sampling_kwargs, [*upsamplers, top_prior], hps)

```

