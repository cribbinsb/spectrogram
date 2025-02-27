import argparse
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import librosa
import soundfile as sf
import math

def create_text_image(text, width, height):
    # Create a black image of the given size.
    im = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(im)
    # Use the default font.
    font = ImageFont.load_default()
    # Try to use textbbox (available in newer versions of Pillow)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback to textsize if textbbox is unavailable
        text_width, text_height = draw.textsize(text, font=font)
    # Compute coordinates to center the text.
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    # Draw the text in white.
    draw.text((x, y), text, fill=255, font=font)
    return im

def main():
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(
        description="Convert a JPEG image or generated text image into a WAV file using Griffin–Lim inversion. "
                    "The image is treated as a mel spectrogram (256 bins) spanning a logarithmic "
                    "frequency range between min_freq and max_freq."
    )
    # Positional input_file is optional if --text is provided.
    parser.add_argument("input_file", nargs="?", type=str,
                        help="Path to the input JPEG image. Ignored if --text is provided.")
    parser.add_argument("output_file", type=str, help="Path for the output WAV file.")
    parser.add_argument("--text", type=str, default=None,
                        help="Optional text string to render on a black background instead of using an input image.")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration of the output audio in seconds (default: 10 seconds).")
    parser.add_argument("--min_freq", type=float, default=100.0,
                        help="Minimum frequency in Hz (default: 100 Hz).")
    parser.add_argument("--max_freq", type=float, default=10000.0,
                        help="Maximum frequency in Hz (default: 10000 Hz).")
    parser.add_argument("--equalize", action="store_true",
                        help="Normalize the grayscale image histogram using histogram equalization.")
    parser.add_argument("--invert", action="store_true",
                        help="Invert the grayscale values of the image.")
    args = parser.parse_args()
    
    # Audio and spectrogram parameters.
    sr = 44100          # Sample rate.
    n_mels = 256        # Number of mel bands (vertical resolution of the image).
    n_time_bins = 256   # Number of time bins (horizontal resolution of the image).
    duration = args.duration
    min_freq = args.min_freq
    max_freq = args.max_freq

    # Either generate an image from text or load an existing image.
    if args.text is not None:
        # Create a 256x256 image with the given text.
        im = create_text_image(args.text, 256, int(24*math.log(max_freq/min_freq)))
        im = im.resize((n_time_bins, n_mels), Image.LANCZOS)
    elif args.input_file:
        # Load the image from file and convert to grayscale.
        im = Image.open(args.input_file).convert('L')
        # Optionally equalize the histogram.
        if args.equalize:
            im = ImageOps.equalize(im)
        # Optionally invert the grayscale.
        if args.invert:
            im = ImageOps.invert(im)
        # Resize the image to 256x256.
        im = im.resize((n_time_bins, n_mels), Image.LANCZOS)
    else:
        raise ValueError("You must provide either an input image file or a text string via --text.")

    # Normalize pixel values to [0, 1]
    mel_spec = np.array(im, dtype=np.float32) / 255.0
    # Flip vertically so that the bottom row (lowest frequencies) is at index 0.
    mel_spec = np.flipud(mel_spec)

    # Determine STFT parameters.
    n_samples = int(sr * duration)
    n_fft = 2048
    hop_length = max(1, (n_samples - n_fft) // (n_time_bins - 1))

    # Reconstruct the waveform using librosa's mel_to_audio (which uses Griffin–Lim inversion).
    y = librosa.feature.inverse.mel_to_audio(
        mel_spec, sr=sr, n_fft=n_fft, hop_length=hop_length,
        fmin=min_freq, fmax=max_freq, power=1.0, n_iter=64
    )

    # Write the reconstructed audio to the output file.
    sf.write(args.output_file, y, sr)

if __name__ == "__main__":
    main()
