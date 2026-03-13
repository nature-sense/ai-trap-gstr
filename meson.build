# ─────────────────────────────────────────────────────────────────────────────
#  meson.build — Insect Trap Pipeline
#
#  Source layout:
#    src/              all C++ source and header files
#    trap_config.toml  runtime config (not compiled)
#
#  Build targets (set with -Dtarget=):
#    libcamera     Pi 5 + IMX708 via libcamera   → yolo_libcamera
#    v4l2          Zero 3W / Pico Zero via V4L2   → yolo_v4l2
#    all           both (requires libcamera)
#
#  Examples:
#    meson setup build -Dtarget=libcamera && ninja -C build
#    meson setup build -Dtarget=v4l2      && ninja -C build
# ─────────────────────────────────────────────────────────────────────────────

project('yolo-trap', 'cpp',
  default_options : ['cpp_std=c++17', 'optimization=2'])

# ── Dependencies ──────────────────────────────────────────────────────────────

ncnn_dep    = dependency('ncnn',    required : true)
sqlite3_dep = dependency('sqlite3', required : true)
thread_dep  = dependency('threads', required : true)
omp_dep     = dependency('openmp',  required : false)
libcamera_dep = dependency('libcamera', required : false)

# ── Option: select build target ───────────────────────────────────────────────

target_opt = get_option('target')   # 'libcamera' | 'v4l2' | 'all'

# ── Shared sources (compiled into both targets) ───────────────────────────────

shared_sources = files(
  'firmware/src/pipeline/decoder.cpp',
  'firmware/src/pipeline/tracker.cpp',
  'firmware/src/pipeline/persistence.cpp',
  'firmware/src/pipeline/crop_saver.cpp',
  'firmware/src/pipeline/exif_writer.cpp',
  'firmware/src/pipeline/sync_manager.cpp',
  'firmware/src/server/mjpeg_streamer.cpp',
  'firmware/src/server/sse_server.cpp',
  'firmware/src/server/http_server.cpp',
  'firmware/src/common/stb_image_write_impl.cpp',
)

shared_deps = [ncnn_dep, sqlite3_dep, thread_dep, omp_dep,]

inc = include_directories([
                            'firmware/src/common',
                            'firmware/src/io',
                            'firmware/src/pipeline',
                            'firmware/src/server'
                          ])

# ── Pi 5 / libcamera target ───────────────────────────────────────────────────

if (target_opt == 'libcamera' or target_opt == 'all') and libcamera_dep.found()
  executable('yolo_libcamera',
    sources : shared_sources + files(
      'firmware/src/io/libcamera_capture.cpp',
      'firmware/main_libcamera.cpp',
    ),
    include_directories : inc,
    dependencies        : shared_deps + [libcamera_dep],
  )
endif

# ── Radxa Zero 3W / Pico Zero / V4L2 target ──────────────────────────────────

if target_opt == 'v4l2' or target_opt == 'all'
  executable('yolo_v4l2',
    sources : shared_sources + files(
      'firmware/src/io/src/v4l2_capture.cpp',
      'firmware/main_v4l2.cpp',
    ),
    include_directories : inc,
    dependencies        : shared_deps,
  )
endif
