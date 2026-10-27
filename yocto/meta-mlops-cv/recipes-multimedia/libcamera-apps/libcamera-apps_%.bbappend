# rpicam-apps 1.12.0, the release Pi OS pairs with the libcamera 0.7.1 fork.
PV = "1.12.0"
SRCREV = "f5174e5099fd1e1646622d45f86e6ac62c912776"
# Already in v1.12.0
SRC_URI:remove = "file://add-missing-header.patch"
# Headless: libav encoder only, null preview
PACKAGECONFIG = "libav"
FILES:${PN} += "${libdir}/rpicam-apps-encoder"
# libav_encoder.cpp includes libdrm/drm_fourcc.h; the 1.9.0 recipe predates it
DEPENDS += "libdrm"
