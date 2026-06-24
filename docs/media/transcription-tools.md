# Transcription Tools

Tools:

- FFmpeg
- faster-whisper
- Python

Verify tools with Python, FFmpeg, and pip package inspection.

Install the Python package with pip when needed.

## Workflow

1. Place local media files in one folder.
2. Run the helper script from `scripts/media/bulk_transcribe.py`.
3. Review the generated audio, transcript, and subtitle folders.

## Outputs

- `audio`: WAV files.
- `transcripts`: TXT files.
- `subtitles`: SRT files.

## Suggested settings

- Use model `small` for normal content.
- Use model `medium` for harder audio.
- Use language `es`, `nl`, or `en` according to the source audio.
- Use CPU with INT8 compute for ordinary machines.

## Notes

The script skips files that already have transcript and subtitle outputs.
