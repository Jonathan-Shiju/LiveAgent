from whisper_streaming.whisper_online import *
import time
import numpy as np
import logging
from dotenv import load_dotenv
import os

load_dotenv('envs/dev.env')
model_path = os.getenv('model_path')
warmup_file_path = os.getenv('warmupfile')

class ASRStreaming:
    def __init__(self,
                 audio_path=None,
                 min_chunk_size=1.0,
                 vac_chunk_size=0.04,
                 model='large-v3',
                 enable_vac=True,
                 logging_level='INFO',
                 logfile=None,
                 warmup=True, ):

        # Configure logging
        numeric_level = getattr(logging, logging_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {logging_level}')

        if logfile:
            logging.basicConfig(filename=logfile, level=numeric_level)
        else:
            logging.basicConfig(level=numeric_level)

        # Get a logger for this class
        self.logger = logging.getLogger(__name__)

        if os.path.exists(model_path):
            asr = FasterWhisperASR(lan='en', modelsize=model, model_dir=model_path)
        else:
            self.logger.debug('Resolving model path dynamically')
            asr = FasterWhisperASR(modelsize=model, lan='en')

        # Initialize VAC if enabled
        if enable_vac:
            self.online = VACOnlineASRProcessor(
                online_chunk_size=min_chunk_size,
                asr=asr
            )
            self.min_chunk = vac_chunk_size
        else:
            self.online = OnlineASRProcessor(asr)
            self.min_chunk = min_chunk_size

        self.SAMPLING_RATE = 16000
        self.MIN_CHUNK_SIZE = min_chunk_size
        self.audio_path = audio_path
        self.start_time = None
        self.accumulated_transcription = ""

        # Warm up the model if required
        if warmup:
            if os.path.isfile(warmup_file_path):
                a = load_audio_chunk(warmup_file_path, 0, 1)
                asr.transcribe(a)
                self.logger.info('Whisper is warmed up')
            else:
                self.logger.critical('The warm up file is not available')
        else:
            self.logger.warning('Whisper is not warmed up. First chunk processing may take longer')

    def process_file_simultaneous(self):
        """Process audio file in simultaneous/online mode"""
        self.logger.info(f'Starting simultaneous processing of {self.audio_path}')

        if not self.audio_path:
            self.logger.error("No audio file path provided")
            return

        # Get audio duration
        audio = load_audio(self.audio_path)
        duration = len(audio) / self.SAMPLING_RATE
        self.logger.info(f"Audio duration is: {duration:.2f} seconds")

        # Initialize processing variables
        self.start_time = time.time()
        beg = 0
        end = 0

        self.logger.info("Processing in simultaneous mode...")

        try:
            while True:
                now = time.time() - self.start_time
                if now < end + self.min_chunk:
                    time.sleep(self.min_chunk + end - now)
                end = time.time() - self.start_time
                a = load_audio_chunk(self.audio_path, beg, end)
                beg = end
                self.online.insert_audio_chunk(a)

                try:
                    result = self.online.process_iter()
                    self.output_transcript(result)
                except AssertionError as e:
                    self.logger.error(f"Assertion error: {e}")

                now = time.time() - self.start_time
                self.logger.debug(f"## Last processed {end:.2f}s, now is {now:.2f}, latency is {now-end:.2f}")

                if end >= duration:
                    break

            # Process final chunk
            result = self.online.finish()
            self.output_transcript(result)

        except KeyboardInterrupt:
            self.logger.info('Keyboard interrupt detected. Stopping...')
        except Exception as e:
            self.logger.error(f'Error during processing: {str(e)}')
        finally:
            self.logger.info(f'Finished processing file. Total transcript length: {len(self.accumulated_transcription)} characters')
            return self.accumulated_transcription

    def output_transcript(self, result, now=None):
        """Output transcript with timestamps"""
        if now is None and self.start_time:
            now = time.time() - self.start_time

        if result and result[2]:  # Check if result has transcription text
            beg_ts, end_ts, text = result
            if beg_ts is not None:
                timestamp_line = f"{now*1000:.4f} {beg_ts*1000:.0f} {end_ts*1000:.0f} {text}"
                print(timestamp_line)
                self.accumulated_transcription += text + " "
                self.logger.info(f"Transcription: {text}")

# Simple usage example
if __name__ == "__main__":
    asr = ASRStreaming(
        audio_path="test.wav",
        model="large-v3",
    )
    transcription = asr.process_file_simultaneous()
    print(f"Final transcription: {transcription}")










