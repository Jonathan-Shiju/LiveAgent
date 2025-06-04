from whisper_streaming.whisper_online import *
import sounddevice as sd
import logging
from dotenv import load_dotenv
import os
import queue
import numpy as np


load_dotenv('envs/dev.env')
model_path=os.getenv['model_path']
warmup_file_path=os.getenv['warmupfile']
class ASRStreaming:
    def __init__(self, MIN_CHUNK_SIZE=1000, VAC_CHUNK_SIZE=40, model='large-v3', enable_vac=True, loggingLevel='INFO', loggingfile=None, warmup=True):
        global model_path
        if logfile:
            logging.basicConfig(filename=logfile, level=logging.loggingLevel)
        else:
            logging.basicConfig(level=logging.loggingLevel)

        if os.path.exists(model_path):
            asr = FasterWhisperASR(modelsize=model, model_dir=model_path)
        else:
            logger.debug('Resolving model path dynamically')
            asr = FasterWhisperASR(modelsize=model)

        if enable_vac:
            self.online = VACOnlineASRProcessor(
                online_chunk_size= MIN_CHUNK_SIZE,
                online=asr,
            )
            min_chunk = VAC_CHUNK_SIZE
        else:
            self.online = OnlineASRProcessor(asr)
            min_chunk = MIN_CHUNK_SIZE

        self.CHUNK = 1024
        self.SAMPLING_RATE=16000
        self.MIN_CHUNK_SIZE = MIN_CHUNK_SIZE
        self.is_first = True
        self.audio_queue = queue.Queue()

        if warmup:
            if os.path.isfile(warmup_file_path):
                a = load_audio_chunk(warmup_file_path, 0, 1)
                asr.transcribe(a)
                logger.info('Whisper is warmed up')
            else:
                logger.critical('The warm up file is not available')
        else:
            logger.warning('Whisper is not warmed up. First chunk processing may take longer')

    def receive_audio_chunk(self):
        logger.info('Recoding Audio......')
        out=[]
        minlimit = self.MIN_CHUNK_SIZE*self.SAMPLING_RATE
        while sum(len(x) for x in out) < minlimit:
            try:
                audio = sd.rec(self.CHUNK, samplerate=self.SAMPLING_RATE, channels=1, dtype='int16')
                sd.wait()  # Wait until the recording is finished
                logger.debug('Received audio chunk of size: ', len(audio))
                audio = audio.flatten().astype(np.float32) / 32768.0  # Normalize audio
                out.append(audio)
            except Exception as e:
                logger.error(f"Error recording audio: {e}")
                break
        if not out:
            return None
        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            return None
        self.is_first = False
        return np.concatenate(out)

    def audio_callback(self, indata, frames, time, status):
        """This is called for each audio block."""
        if status:
            print(status)
        # Convert and normalize the audio data
        audio = indata.flatten().astype(np.float32) / 32768.0
        self.audio_queue.put(audio)

    def process(self):
        logger.info('Starting......')
        # Start the recording with callback
        try:
            with sd.InputStream(callback=self.audio_callback,
                            channels=1,
                            samplerate=self.SAMPLING_RATE,
                            blocksize=self.CHUNK):
                logger.info('Recording started')
                accumulated_transcription = ""
                while True:
                    # Collect enough audio data
                    audio_chunks = []
                    while sum(len(x) for x in audio_chunks) < self.MIN_CHUNK_SIZE * self.SAMPLING_RATE:
                        try:
                            # Get audio from queue with timeout
                            chunk = self.audio_queue.get(timeout=1.0)
                            audio_chunks.append(chunk)
                        except queue.Empty:
                            # No audio received in timeout period
                            continue

                    # Process the collected audio chunks
                    if audio_chunks:
                        audio_data = np.concatenate(audio_chunks)
                        self.online.insert_audio_chunk(audio_data)
                        result = self.online.process_iter()
                        logger.info(result)

                    accumulated_transcription = accumulated_transcription + result
        except KeyboardInterrupt:
            logger.info('Stopping')
            with open("logfile.txt", "w") as file:
                file.write(accumulated_transcription)
        except Exception as e:
            logger.info('Exception exit: ', str(e))
        finally:
            self.online.init()

if __name__ == "__main__":
    logger.debug('Creating Object for Class ASRStreaming')
    asr = ASRStreaming()
    logger.debug('Calling the process function')
    asr.process()










