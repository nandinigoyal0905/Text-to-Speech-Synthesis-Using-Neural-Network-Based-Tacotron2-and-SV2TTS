import argparse
import os
from pathlib import Path
import spacy
import time
import numpy as np
import soundfile as sf
import torch
import tensorflow as tf
import json

from emotion_encoder.Model import TIMNET_Model
import speaker_encoder.inference
import speaker_encoder.params_data
from emotion_encoder.utils import get_mfcc
from synthesizer.inference import Synthesizer_infer
from synthesizer.utils.cleaners import add_breaks, english_cleaners_predict
from vocoder import inference as vocoder
from vocoder.display import save_attention_multiple, save_spectrogram, save_stop_tokens
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from speed_changer.fixSpeed import TransFormat, AudioAnalysis, DelFile, work

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run_id", type=str, default="default_emotion", help=(
        "Name for this model. By default, training outputs will be stored to saved_models/<run_id>/. "
        "If a model state from the same run ID was previously saved, the training will restart from there. "
        "Pass -f to overwrite saved states and restart from scratch."
    ))
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models",
                        help="Directory containing all saved models")
    parser.add_argument("--emotion_encoder_model_fpath", type=Path,
                        default="saved_models/default_emotion/INTERSECT_46_dilation_8_dropout_05_add_esd_npairLoss", help=(
                            "Path your trained emotion encoder model."
                        ))
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of input audio for voice filter")
    parser.add_argument("--griffin_lim",
                        action="store_true",
                        help="if True, use griffin-lim, else use vocoder")
    parser.add_argument("--cpu", action="store_false", help=(
        "If True, processing is done on CPU, even when a GPU is available."
    ))
    parser.add_argument("--no_sound", action="store_true", help=(
        "If True, audio won't be played."
    ))
    parser.add_argument("--seed", type=int, default=None, help=(
        "Optional random number seed value to make toolbox deterministic."
    ))
    args = parser.parse_args()
    arg_dict = vars(args)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print(f"Found {torch.cuda.device_count()} GPUs available. Using GPU {device_id} ({gpu_properties.name}) "
              f"of compute capability {gpu_properties.major}.{gpu_properties.minor} with "
              f"{gpu_properties.total_memory / 1e9:.1f}Gb total memory.\n")
    else:
        print("Using CPU for inference.\n")

    if not args.griffin_lim:
        print("Preparing the encoder, the synthesizer, and the vocoder...")
    else:
        print("Preparing the encoder and the synthesizer...")
    ensure_default_models(args.run_id, args.models_dir)

    speaker_encoder.inference.load_model(list(args.models_dir.glob(f"{args.run_id}/encoder.pt"))[0])
    synthesizer = Synthesizer_infer(list(args.models_dir.glob(f"{args.run_id}/synthesizer.pt"))[0], model_name="EmotionTacotron")
    if not args.griffin_lim:
        vocoder.load_model(list(args.models_dir.glob(f"{args.run_id}/vocoder.pt"))[0])

    # Prepare the emotion encoder
    json_fpath = os.path.join(args.emotion_encoder_model_fpath, "params.json")
    with open(json_fpath) as f:
        emotion_encoder_args = argparse.Namespace(**json.load(f))

    os.environ['CUDA_VISIBLE_DEVICES'] = emotion_encoder_args.gpu
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    print(f"###gpus: {gpus}")

    CLASS_LABELS = ("angry", "happy", "neutral", "sad", "surprise")

    emotion_encoder = TIMNET_Model(args=emotion_encoder_args, input_shape=(626, 39), class_label=CLASS_LABELS)
    emotion_encoder.create_model()

    print("Interactive generation loop")
    num_generated = 0

    nlp = spacy.load('en_core_web_sm')
    weight = arg_dict["weight"]
    amp = 1

    while True:
        in_fpath = Path(input("Reference voice: enter an audio folder of a voice to be cloned (mp3, wav, m4a, flac, ...):\n").replace("\"", "").replace("\'", ""))
        fpath_without_ext = os.path.splitext(str(in_fpath))[0]
        speaker_name = os.path.normpath(fpath_without_ext).split(os.sep)[-1]

        is_wav_file, single_wav, wav_path = TransFormat(in_fpath, 'wav')
        if not is_wav_file:
            os.remove(wav_path)
        wav = single_wav
        path_ori, _ = os.path.split(wav_path)
        file_ori = 'temp.wav'
        fpath = os.path.join(path_ori, file_ori)
        sf.write(fpath, wav, samplerate=speaker_encoder.params_data.sampling_rate)

        totDur_ori, nPause_ori, arDur_ori, nSyl_ori, arRate_ori = AudioAnalysis(path_ori, file_ori)
        DelFile(path_ori, '.TextGrid')
        os.remove(fpath)

        preprocessed_wav = speaker_encoder.inference.preprocess_wav(wav)

        print("Loaded input audio file successfully")

        speaker_embed = speaker_encoder.inference.embed_utterance(preprocessed_wav)
        mfcc = get_mfcc(wav, synthesizer.hparams.sample_rate, mean_signal_length=320000)
        emotion_embed = emotion_encoder.infer(np.array([mfcc]), model_dir=args.emotion_encoder_model_fpath)[0]

        start_syn = time.time()
        text = input("Write a sentence to be synthesized:\n")

        if args.seed is not None:
            torch.manual_seed(args.seed)
            synthesizer = Synthesizer_infer(args.syn_model_fpath)

        def preprocess_text(text):
            text = add_breaks(text)
            text = english_cleaners_predict(text)
            texts = [i.text.strip() for i in nlp(text).sents]
            return texts

        texts = preprocess_text(text)
        print(f"List of input texts:\n{texts}")

        specs = []
        alignments = []
        stop_tokens = []

        for text in texts:
            spec, align, stop_token = synthesizer.synthesize_spectrograms([text], [speaker_embed], [emotion_embed], require_visualization=True)
            specs.append(spec[0])
            alignments.append(align[0])
            stop_tokens.append(stop_token[0])

        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        alignments = np.array(alignments)
        stop_tokens = np.array(stop_tokens)

        if not os.path.exists("syn_results"):
            os.mkdir("syn_results")
        save_attention_multiple(alignments, "syn_results/attention")
        save_stop_tokens(stop_tokens, "syn_results/stop_tokens")
        save_spectrogram(spec, "syn_results/mel")
        print("Created the mel spectrogram")

        end_syn = time.time()
        print(f"Prediction time of synthesizer is {end_syn - start_syn}s")

        start_voc = time.time()
        print("Synthesizing the waveform:")

        if args.seed is not None:
            torch.manual_seed(args.seed)
            vocoder.load_model(args.voc_model_fpath)

        if not args.griffin_lim:
            wav = vocoder.infer_waveform(spec, target=vocoder.hp.voc_target, overlap=vocoder.hp.voc_overlap, crossfade=vocoder.hp.is_crossfade)
        else:
            wav = Synthesizer_infer.griffin_lim(spec)
        wav = wav.astype(np.float32)
        end_voc = time.time()
        print(f"Prediction time of vocoder is {end_voc - start_voc}s")
        print(f"Prediction time of TTS is {end_voc - start_syn}s")

        b_ends = np.cumsum(np.array(breaks) * Synthesizer_infer.hparams.hop_size)
        b_starts = np.concatenate([[0], b_ends[:-1]])
        wav = [wav[start:end] for start, end in zip(b_starts, b_ends)]
        wav = np.concatenate

        print("Data type of wav:", wav.dtype)

        if not args.no_sound:
            import sounddevice as sd
            try:
                sd.stop()
                sd.play(wav, synthesizer.sample_rate)
            except sd.PortAudioError as e:
                print(f"Caught exception: {repr(e)}")
                print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
            except:
                raise

        print("Synthesized waveform successfully.\n")

        filename = f"demo_output_{num_generated + 1}.wav"
        dir_path = os.getcwd()
        fpath = os.path.join(dir_path, "out_audios", filename)

        sf.write(fpath, wav.astype(np.float32), synthesizer.sample_rate)
        num_generated += 1

        path_ch = os.path.join(dir_path, 'out_audios', f"{filename[:-4]}_sync.wav")
        work(fpath, path_ch, amp, totDur_ori, nPause_ori)  # Corrected the number of arguments
        print(f"Done.\nYour synthesized audio file with sync and amplified is ready.\nFile: {path_ch}\n")

if __name__ == '__main__':
    main()
