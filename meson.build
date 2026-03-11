# ─────────────────────────────────────────────────────────────────────────────
#  meson.build — Radxa Zero 3W / V4L2 target
#
#  Adds a second executable 'yolo_v4l2' alongside the existing 'yolo_libcamera'.
#  Shared sources (decoder, tracker, persistence, crop_saver, mjpeg_streamer,
#  stb) are compiled once and linked into both targets.
#
#  To build only the V4L2 target on the Zero 3W (which has no libcamera):
#    meson setup build -Dtarget=v4l2
#    ninja -C build
#
#  To build both on a machine that has libcamera:
#    meson setup build -Dtarget=all
#    ninja -C build
# ─────────────────────────────────────────────────────────────────────────────

project('yolo-trap', 'cpp',
  default_options : ['cpp_std=c++17', 'optimization=2'])

# ── Dependencies ──────────────────────────────────────────────────────────────

ncnn_dep    = dependency('ncnn',   required : true)
sqlite3_dep = dependency('sqlite3',required : true)
thread_dep  = dependency('threads',required : true)
openmp_dep  = dependency('openmp', required: true)

# libcamera only needed for the Pi 5 target
libcamera_dep = dependency('libcamera', required : false)

# ── Option: select build target ───────────────────────────────────────────────

target_opt = get_option('target')   # 'libcamera' | 'v4l2' | 'all'

# ── Shared sources ────────────────────────────────────────────────────────────

shared_sources = files(
  'src/decoder.cpp',
  'src/tracker.cpp',
  'src/persistence.cpp',
  'src/crop_saver.cpp',
  'src/mjpeg_streamer.cpp',
  'src/stb_image_write_impl.cpp',
)

shared_deps = [ncnn_dep, openmp_dep, sqlite3_dep, thread_dep]

# ── Pi 5 / libcamera target ───────────────────────────────────────────────────

if (target_opt == 'libcamera' or target_opt == 'all') and libcamera_dep.found()
  executable('yolo_libcamera',
    sources : shared_sources + files(
      'src/libcamera_capture.cpp',
      'src/main_libcamera.cpp',
    ),
    dependencies : shared_deps + [libcamera_dep],
  )
endif

# ── Radxa Zero 3W / V4L2 target ───────────────────────────────────────────────
#
#  V4L2 uses only kernel headers (linux/videodev2.h) — no extra library needed.

if target_opt == 'v4l2' or target_opt == 'all'
  executable('yolo_v4l2',
    sources : shared_sources + files(
      'src/v4l2_capture.cpp',
      'src/main_v4l2.cpp',
    ),
    dependencies : shared_deps,
  )
endif
