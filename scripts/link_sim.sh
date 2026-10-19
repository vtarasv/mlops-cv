#!/usr/bin/env bash
# The fake drone: a demo-store clip looped forever as OpenHD-like H.264 RTP on localhost.
# Run from the repo root (`make link-sim` does); override any LINK_* variable from the env.
set -euo pipefail

LINK_SEQ=${LINK_SEQ:-${HOST_SUBSET_DIR:-}/demo/images/uav0000073_00600_v}  # a demo-store clip dir
LINK_BITRATE=${LINK_BITRATE:-8M}  # OpenHD's default
LINK_BUFSIZE=${LINK_BUFSIZE:-4.8M}  # 0.6 × LINK_BITRATE: GStreamer's 600 ms VBV, as OpenHD runs it
LINK_GOP=${LINK_GOP:-5}  # keyframe every N frames; OpenHD's default
LINK_PORT=${LINK_PORT:-5600}
LINK_SDP=data/link-sim/link-$LINK_PORT.sdp

[[ -d $LINK_SEQ ]] || { echo "no demo-store clip at LINK_SEQ=[$LINK_SEQ]; run make ingest" >&2; exit 1; }
mkdir -p "$(dirname "$LINK_SDP")"
echo "sending $LINK_SEQ to rtp://127.0.0.1:$LINK_PORT; receive with --link $LINK_SDP"

args=(
  -hide_banner -loglevel warning
  -re  # read at the input's frame rate, as a camera delivers, not as fast as possible
  -stream_loop -1 -framerate 30 -i "$LINK_SEQ/%07d.jpg"  # the JPEGs as 30 fps video, looped
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  # 4:2:0 needs even sides; some clips are odd
  -pix_fmt yuv420p
  # OpenHD's software encoder (x264enc ultrafast zerolatency threads=2 sliced-threads=false):
  # no B-frames or lookahead, and one slice per frame, so a lost packet damages the whole frame.
  -c:v libx264 -preset ultrafast -tune zerolatency -threads 2 -x264-params sliced-threads=0
  -g "$LINK_GOP" -keyint_min "$LINK_GOP"  # a fixed GOP, as a drone encoder sends
  -b:v "$LINK_BITRATE" -maxrate "$LINK_BITRATE" -bufsize "$LINK_BUFSIZE"  # near-CBR, like a radio link
  -f rtp -sdp_file "$LINK_SDP" "rtp://127.0.0.1:$LINK_PORT"  # bare RTP, plus the SDP to open it
)
exec "$(uv run python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "${args[@]}"
