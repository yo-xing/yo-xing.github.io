# yo-xing.github.io
## Final Project for MUSIGR6610 - SOUND: ADVANCED TOPICS at Columbia University

## Investigating the role of Artificial Intelligence in Music with JukeBox and the Amen Break
### Yo Jeremijenko-Conley  

#### Motivation
In recent months, the role of Artificial Intelligence (AI) in art has become increasingly prevalent. Software like DALL-E 2, MidJourney, and DiffusionBee have enabled users to generate images based on textual descriptions. This software uses generative AI to create brand-new images (or artworks) that had previously never existed, based on datasets of millions of existing images/artwork. These software services have slightly different implementations, but their overall structure is relatively similar. For example, The DALL-E 2 model works as follows: 

- First, the text prompt given by the user is fed into a trained text encoder model which maps the prompt into a representation space. After that, an OpenAI model called CLIP (Contrastive Language-Image Pre-Training) maps the representation space of the text encoding to a corresponding image representation space, CLIP is trained on hundreds of millions of images with associated captions, and learns how related any given piece of text is to an image. This captures the semantic information of the text prompt. 
- A model called the prior then takes the CLIP embedding and generates a specific image embedding from it, you can think of it as forming a “mental image” of the text. 
- Lastly, a diffusion model is used to stochastically generate the image by removing noise (or upsampling) from the image embedding.

Diffusion models are trained with existing images with gaussian noise added to them iteratively by a Markov chain, which eventually leaves the image as nothing but noise. The diffusion model then learns how to iteratively reverse this noise to generate a (low-resolution) image from seemingly random noise. However, this general diffusion process does not correspond to a specific text prompt. To fix this, DALL-E 2 uses a modified version of their diffusion model (called GLIDE) which incorporates projected CLIP text embeddings into this process. At each time step in the iterative process of noise removal, the CLIP text embeddings are added to the image embedding at that specific time-step to “guide” the removal of noise in the direction of the texts representation space, classifier free guidance is also used to further guide the image towards the text prompt. 

![1*_wtvEU05qyDISFBT4xRVgw](https://user-images.githubusercontent.com/40434203/208201013-bfaeac92-8ea9-4773-82b0-4c582fcfee48.png)

  
 The results of these machine learning processes have been incredibly impressive in their ability to capture and translate semantic information and their grasp of aesthetics and art styles. This has sparked discourse on whether these models exploit artists whose work is being used to train these models, and their potential to generate artwork in other media, eventually becoming a cheap alternative to replace creative professionals. After experiencing the capabilities of these models, I view them as a “proof of concept” to creative applications of AI in the arts. This has made me consider the possibilities of this technology being used to generate music, as well as the cultural and economic implications of such a task. In this project, I attempt to examine the current state of music-generating AI as well as speculate on its future capabilities and impact on the world. 
Some may assume that AI models would be better able to generate music and capture its stylistic patterns than create visual art due to the inherent mathematical structure of music (such as the diatonic scale). However, the current capability of AI models to generate music is far behind their capability to generate visual art in the form of images. This is mainly due to two reasons, the first being the sequential nature of music, the interpretation of each tone or sound in a song is reliant on those that came before it and necessary for the interpretation of later sounds in the piece. While this can also be said for images, the non-sequential nature of how we experience them allows us to discount specific sections of an image easier than parts of a song. The second reason for the disparity between AI-generated images and music is the immense size of an audio file compared to an image, this makes encoding and decoding audio vastly more computationally expensive than doing so for an image. 


#### OpenAI’s JukeBox

To better understand the current capabilities and limitations of AI in music generation, I decided to experiment with JukeBox –a neural net that generates music–, made by OpenAI (the company responsible for DALL-E 2 and ChatGPT). While algorithmic composition has existed for many years, JukeBox is different in that it models music as raw audio rather than symbolic generators that use MIDI files or musical scores. This allows JukeBox to capture the subtle timbres and dynamics found in music as well as the human voice, however, the use of raw audio also gives the models an extremely high computational cost. For instance, a 16-bit 4-minute song at 44 kHz has over 10 million timesteps. To address this problem high dimensional input problem, JukeBox uses an autoencoder to compress raw audio to a lower-dimensional space, while attempting to only preserve the relevant information in the audio. JukeBox’s prior models can then generate audio in this compressed, lower-dimenstional space (similar to DALL-E 2’s prior for generating images) and then transformer models can be used to upsample the compressed audio to the raw audio space. JukeBox’s encoding approach uses VQ-VAE (Vector Quantised-Variational AutoEncoder). However, many VQ-VAE models suffer from codebook collapse, to address this, JukeBox uses a more simplified architecture called VQ-VAE-2 which only uses feed-foward encoders and decoders. However they modify this architecture by adding in random restarts of the codebook, to further prevent codebook collapse, as well as spectral loss which allows easier reconstruction of high frequencies. JukeBox’s VQ-VAE compresses raw audio (44kHz) at three levels, by factors of 8x, 32x, and 128x, with codebook size of 2048 for all levels. This compression losses a great deal of the audio detail, and sounds noticeably more noisy/scratchy as you go down levels, however the important structural information of the music such as pitch, timbre and volume is kept at all levels. 
The prior models are then trained to learn the distribution of the music encoded by the VQ-VAE and generate music in this compressed space. Similar to the VQ-VAE models, there are three levels of priors, a top level prior that generates the music in the most compressed space, and then two priors that upsample, generating less compressed audio conditional on the top level prior’s output. The top level prior is essential for modeling the overall structure of the music, and has very low audio quality, but still captures basic semantics such as the melody or voices. The middle and bottom upsampling priors add more local musical semantics like timbre, and improves the overall quality of the audio. Attached below is the output of a generated sample at each of the three levels:




[top level](https://drive.google.com/file/d/1AHoF42S0TAQgeaL0c2aoZpCZsF9uHMp8/view?usp=sharing)

[middle level](https://drive.google.com/file/d/1WCSQKTXc-Ev_c-stTAAQS5y0snd5o4az/view?usp=share_link)

[bottom level](https://drive.google.com/file/d/1Zl4fHxYXYAq7qJ1KXULl_oNxyb5ogp39/view?usp=share_link)


These prior models are trained using Sparse transformers, each containing 72 layers of factorization over a context of 8192 codes, for the top level this corresponds to 24 seconds of audio, while it corresponds to 6 seconds on the middle level prior and 1.5 seconds on the bottom level.

To curate the dataset for training JukeBox, OpenAI crawled the web and gathered 1.2 million songs, half of which were in English, along with corresponding lyrics and metadata such as genre, album, artist, mood, and release date from LyicWiki. It is unlikely that artists were asked for consent/permission to use their songs or the associated metadata. OpenAI is not currently making any profit directly from JukeBox as it is free open-source software, however, it is possible that they profit from later versions of the software, or that third-party individuals are currently making a profit using the software. 
	Furthermore, unlike software such as DALL-E 2 where the model’s input is an open-ended text string, JukeBox is conditioned directly on preset genre and artist tokens as well as a ‘prompt’ wav file, for the model to expand on (it is also possible to generate audio in ‘ancestral’ mode without a wav input, but in my experience, it performed much worse than when in ‘primed’ mode). Both genre and artist information is optional and not needed to generate music, however, providing both gives an advantage by reducing the entropy of the audio generation which generally gives better results and a more consistent style throughout. This also makes the genre output more deterministic compared to generation without this information, but there is still a large amount of variance between outputs when given the same input. It is also possible to condition the model on lyrics in the form of text, though I will not be doing this for the purposes of this project. 



#### The Amen Break

To investigate JukeBox’s understanding and interpretation of different genres and artists, I decided to produce a number of samples using the same audio input, while changing the genre and artist parameters and holding all else equal. I decided to use the Amen Break as the audio input for the model. 

[The Amen Break - Six second sample](https://drive.google.com/file/d/1e3xXfnYlhkfcvIXA-MORcVo5q_Z1QUto/view?usp=share_link)


Originally released as a part of "Color Him Father" by the soul group the Winstons, the Amen Break is a widely sampled drum break that has been used in thousands of songs in many different genres, it is believed to be one of the most sampled recordings in history. It appears in songs such as "Straight Outta Compton" by N.W.A and "Keep It Going Now" by Rob Base and even the Futurama theme song. I thought that this would make it a very interesting input audio file, as it is likely included in a large portion of the songs that JukeBox was trained on. Additionally, it is an example of the social problem of sampling and plagiarism in the music industry as a whole, while generative AI like JukeBox will only increase the complexity of these issues in the future. Sadly, the Amen Break's performer, Gregory Coleman, received no royalties for the sample, despite it being widely used, and passed away homeless in 2006, unaware of the astounding impact he had made on music. 

#### Model Parameters

In addition to genre, artist, and input audio file, JukeBox takes in a number of other parameters when generating music, I held all these parameters constant besides genre and artist when generating music from the Amen Break. 
- The model parameter specifies which pre-trained model it will use to generate the audio, these are 5b_lyrics, 5b, and 1b_lyrics (a smaller and more efficient version of 5b lyrics that produces lower quality audio), I used the 5b model as I did not input any lyrics for conditioning. 
- The n_samples parameter specifies how many samples to generate based on the prompt, generating multiple samples in one run is more efficient than separately generating each sample as it will not force the model to encode and transform the input audio for each sample, I chose 2 for this parameter as a larger number would be too memory intensive for the Tesla T-4 GPU I have been using. 
- The speed_upsampling parameter is a binary that allows for faster upsampling, however, this results in choppier-sounding audio, so I have it turned off for my generations. 
- The prompt_length_in_seconds parameter specifies how many seconds of the input audio file will be used to prime the model before it “kicks in” and starts to add on, I have this set at 6 seconds for the first loop of the Amen Break. 
- Sample_length_in_seconds specifies the full length of the sample, I have this set at 24 seconds which leaves 18 seconds for the model to generate music after being primed on the Amen Break. 
- Sampling_tempurature determines the “creativity” of JukeBox when generating or how much the output will deviate from the input audio, this can be between 0.95 (low deviation) and 0.999(high deviation), after experimenting with it, I put the sampling_tempurature at 0.96 as higher values gave worse-sounding, and less consistent results. 
The code for selecting and setting these parameters is displayed below:


``` python:
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

```
#### Generated Tracks

While the music generated by JukeBox is far from the quality of human-made music, it is interesting to see which tracks sound better than others. Interestingly, some genres seemed to utilize the prompt more than others, this may be attributed to the amount of similarity between the Amen Break and the genre’s representation space. For instance, this [sample](https://soundcloud.com/yobot-952874550/jazz-0?in=yobot-952874550/sets/artificial-break-ai-generated-variations-of-the-amen-break) conditioned on Jazz sounds much more comprehensive than this [sample](https://soundcloud.com/yobot-952874550/pop-1?in=yobot-952874550/sets/artificial-break-ai-generated-variations-of-the-amen-break) conditioned on pop. Other samples seem to have some basic comprehension of the Amen Break’s musical structure at first only to break into seeming gibberish shortly after, such as this [bluegrass sample](https://soundcloud.com/yobot-952874550/bluegrass-0?in=yobot-952874550/sets/artificial-break-ai-generated-variations-of-the-amen-break). In general, it seemed like genres that did not commonly have vocal leads produced better songs than those which did since AI-generated human voices sound like gibberish and are difficult on the ear. Furthermore, genres that were more stylistically similar to the Amen Break such as Jazz, Blues, and Hip Hop seemed to do better than those that were less similar such as Bluegrass, Pop, and Trap. 
Much like DALL-E 2, JukeBox supposedly performs better when given more specified prompts as it reduces the entropy of the generation, so I generated samples that are conditioned on both genre and artist. These did not show any significant improvement over samples conditioned only on genre but this could be due to the choice of artist. Artist genre pairs used were Kendrick Lamar - Hip Hop, Pink Floyd - Psychedelic, and Duke Ellington - Jazz. Despite the importance of the Amen Break and its widespread use, it seemed that prior work that conditioned JukeBox on input audio that contained melodies – such as [this sample](https://soundcloud.com/yobot-952874550/grimez-but-ai)– performed better. The full Album of all Amen Break variations can be found [here](https://soundcloud.com/yobot-952874550/sets/artificial-break-ai-generated-variations-of-the-amen-break). 

#### The Future of AI in Music

The path forward for AI-generated music is filled with hurdles and nuanced issues. When speculating on its future, one must not only consider the technological hurdles but also the social consequences of the technology and how to best handle them. Potential future music-generating models may use a mix of raw audio generation and symbolic generation, with each main component of a track (such as melody, vocals, and drums) being generated in raw audio, but conditioned on symbolic representations of the track’s other components. Furthermore, the dataset used to train music-generating models could be greatly improved upon. Rather than curating a dataset of over a million songs without the consent of artists, a more socially responsible way to collect data would be to do so in a collaborative setting in which data on the process of music production is gathered, rather than just the final output. In my view, this could be accomplished by making a free music production tool that artists could use, knowing that they are helping to train a music-generating model. This music production tool could be much like a free version of Fruity Loops or Logic Pro, and provide AI-based tools for tackling much simpler tasks than music generation such as mastering songs or generating stems/samples from input audio files. This could provide much more structured musical data in which each instrument/component in a song is a separate wav file and plugins like reverb or pitch correction could be explicitly represented in the data. This would also allow generatively models to be evaluated and tested by users of the tool and allow artists to collaborate with AI to make full tracks (such as AI-generated drums that accompany human-made melodics and vocals). As these music-generating models advance and start to become profitable, shares of the royalties could be given to the artists that provided the training data for these models giving them alternative forms of income, since the data would be more structured, allowing training data to be better partitioned based on musical style and other attributes. Of course, this is all hypothetical, and implementing such a tool would be a great challenge and would likely be far from perfect.
Generative AI and its artistic capabilities will inevitably progress. In order to keep this progression ethical and responsible, we must ask ourselves a number of existential questions regarding the relationship between technology and art. What is creativity? How do we differentiate between novel artwork being created as a result of “artificial creativity” or as a conglomeration of existing pieces that were creatively made by humans? What is inspiration? How do we differentiate between artists that are heavily influenced by ones that came before them and generative models that are trained on the work of artists, but build new concepts that are not based on any single artist or work? As is recently evident with the use of DALL-E 2 and similar generative models, the artistic community is split on their view of AI-generated art. While many artists view this technology as a threat to their lively hood while already facing difficulties in monetizing their work, other artists view generative AI as a tool that can greatly increase the creative capability of humans. The use of others’ art as a means to create new artwork, is not a new issue by any means. It can result in breathtaking new pieces that benefit all parties involved such as Kendrick Lamar’s To Pimp a Butterfly which attributed samples to their original creators (such as George Clinton) or can result in Gregory Coleman passing away homeless despite the immense influence of his Amen Break. To avoid more outcomes like Colman’s, the standards and regulations of AI-generated art must not be based on current intellectual property laws or regulations, but on ethics, and the morals of the artistic community. 

#### Acknowledgments

I would like to thank Professor Miller Puckette for his guidance throughout this project and class and for exposing me to the Amen break and a number of other amazing samples. I would also like to thank OpenAI and the team behind JukeBox for creating a free and open-source tool for anyone to use and experiment with. Lastly, I would like to thank Gregory Coleman and the Winstons for producing and performing such an amazing and influential sample, as well as any artist whose work was used to train generative AI models.





