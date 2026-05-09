# Stream Simulator

The simulator loops local video files through FFmpeg into an RTSP server, so the
backend exercises the same RTSP code path it will use against real cameras.

## One-time setup

**1. Install ffmpeg**

```bash
# macOS
brew install ffmpeg
# Ubuntu / Debian
sudo apt-get install -y ffmpeg
# Windows: download from https://ffmpeg.org/download.html
```

**2. Install MediaMTX (RTSP server)**

Download the latest release for your platform from
https://github.com/bluenviron/mediamtx/releases — single binary, no install.

```bash
# Linux x86_64 example:
wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_linux_amd64.tar.gz
tar xzf mediamtx_linux_amd64.tar.gz
./mediamtx
```

By default MediaMTX listens on `rtsp://localhost:8554`. Leave it running in its
own terminal.

**3. Get construction footage**

Place your `.mp4` files anywhere. Recommended public sources:
- Pixabay / Pexels: search "construction site"
- Your own captures (best — matches the cameras you'll deploy)

## Running the simulator

```bash
python -m simulation.stream_simulator \
  --videos cam_01:/path/to/site_north.mp4 \
           cam_02:/path/to/site_east.mp4 \
           cam_03:/path/to/crane_zone.mp4 \
           cam_04:/path/to/perimeter.mp4
```

Each video is published as `rtsp://localhost:8554/<cam_id>` and loops forever.

## Pointing the backend at the simulator

```bash
export LABOR_INTEL_STREAM_SOURCES="cam_01=rtsp://localhost:8554/cam_01,cam_02=rtsp://localhost:8554/cam_02,cam_03=rtsp://localhost:8554/cam_03,cam_04=rtsp://localhost:8554/cam_04"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `connection refused` | MediaMTX isn't running, or wrong port |
| ffmpeg keeps reconnecting | Source file is broken/unsupported codec — try re-encoding once: `ffmpeg -i in.mp4 -c:v libx264 -an out.mp4` |
| App reports `seconds_since_last_frame: -1` | Stream never connected. Check `mediamtx` log for the RTSP path. |
| App reads at 0.5 FPS | Source video FPS reported as garbage; StreamManager defaults stride to 25/target. Re-encode with `-r 25 -fps_mode cfr`. |
