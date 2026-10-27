# Raspberry Pi's fork: upstream libcamera can't match the downstream rp1-cfe driver's entity names.
SRC_URI = "git://github.com/raspberrypi/libcamera.git;protocol=https;nobranch=1"
SRCREV = "06c385619acb10bbfb33f52f3abeb8f8c095f42b"
PV = "0.7.1+rpt20260609"
# :aarch64, because meta-oe's LIBCAMERA_PIPELINES:aarch64 ??= would override a plain assignment
LIBCAMERA_PIPELINES:aarch64 = "rpi/pisp"
