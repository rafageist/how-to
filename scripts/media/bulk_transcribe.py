from pathlib import Path
import argparse
import subprocess
from faster_whisper import WhisperModel


MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v",
    ".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac",
}


def format_timestamp(seconds: float) -> str:
    millis = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def extract_audio(input_file: Path, output_audio: Path):
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_file),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_audio),
    ]

    subprocess.run(cmd, check=True)


def write_txt(segments, output_txt: Path):
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with output_txt.open("w", encoding="utf-8") as f:
        for segment in segments:
            text = segment.text.strip()
            if text:
                f.write(text + "\n")


def write_srt(segments, output_srt: Path):
    output_srt.parent.mkdir(parents=True, exist_ok=True)

    with output_srt.open("w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing media files")
    parser.add_argument("--model", default="small", help="tiny, base, small, medium, large-v3")
    parser.add_argument("--language", default="es", help="Audio language, for example: es, nl, en")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--compute-type", default="int8", help="int8 for CPU, float16 for GPU")
    args = parser.parse_args()

    input_folder = Path(args.folder)

    audio_folder = input_folder / "audio"
    txt_folder = input_folder / "transcripts"
    srt_folder = input_folder / "subtitles"

    files = [
        p for p in input_folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in MEDIA_EXTENSIONS
        and "audio" not in p.parts
        and "transcripts" not in p.parts
        and "subtitles" not in p.parts
    ]

    if not files:
        print("No audio or video files found.")
        return

    print(f"Files found: {len(files)}")
    print(f"Loading model: {args.model}")

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    for index, input_file in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] Processing: {input_file.name}")

        relative = input_file.relative_to(input_folder)
        safe_stem = relative.with_suffix("")

        output_audio = audio_folder / safe_stem.with_suffix(".wav")
        output_txt = txt_folder / safe_stem.with_suffix(".txt")
        output_srt = srt_folder / safe_stem.with_suffix(".srt")

        if output_txt.exists() and output_srt.exists():
            print("Transcript already exists. Skipping.")
            continue

        try:
            print("Extracting audio...")
            extract_audio(input_file, output_audio)

            print("Transcribing...")
            segments_generator, info = model.transcribe(
                str(output_audio),
                language=args.language,
                beam_size=5,
                vad_filter=True,
            )

            segments = list(segments_generator)

            write_txt(segments, output_txt)
            write_srt(segments, output_srt)

            print(f"TXT: {output_txt}")
            print(f"SRT: {output_srt}")

        except subprocess.CalledProcessError:
            print(f"FFmpeg error in: {input_file}")
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
