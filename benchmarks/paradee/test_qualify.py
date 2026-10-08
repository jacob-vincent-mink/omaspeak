"""Signal validation catches invalid artifacts without installing any inference runtime."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
import wave

spec = importlib.util.spec_from_file_location('paradee_qualify', Path(__file__).with_name('qualify.py'))
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


class WaveformValidation(unittest.TestCase):
    def write_wav(self, path, samples, rate=24000):
        with wave.open(str(path), 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(struct.pack('<' + 'h' * len(samples), *samples))

    def test_valid_pcm_records_signal_and_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audio.wav'
            self.write_wav(path, [32767, -32768, 100, -100])
            metrics = qualify.waveform(path)
            self.assertEqual(metrics['samples'], 4)
            self.assertEqual(metrics['clipped_samples'], 2)
            self.assertEqual(metrics['peak'], 1)
            self.assertEqual(len(metrics['wav_sha256']), 64)
            self.assertGreater(metrics['rms'], 0)

    def test_wrong_format_empty_and_silent_audio_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audio.wav'
            for samples, rate in [([], 24000), ([0, 0], 24000), ([100], 16000)]:
                self.write_wav(path, samples, rate)
                with self.assertRaises(ValueError):
                    qualify.waveform(path)


if __name__ == '__main__':
    unittest.main()
