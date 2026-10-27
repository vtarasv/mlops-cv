SUMMARY = "ncnn inference runtime, upstream's prebuilt cp314 aarch64 wheel"
HOMEPAGE = "https://github.com/Tencent/ncnn"
# The wheel vendors libgomp (GCC runtime).
LICENSE = "BSD-3-Clause & GPL-3.0-with-GCC-exception"
LIC_FILES_CHKSUM = " \
    file://${COMMON_LICENSE_DIR}/BSD-3-Clause;md5=550794465ba0ec5312d6919e203a55f9 \
    file://${COMMON_LICENSE_DIR}/GPL-3.0-with-GCC-exception;md5=aef5f35c9272f508be848cd99e0151df \
"

WHEEL = "ncnn-${PV}-cp314-cp314-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl"
SRC_URI = "https://files.pythonhosted.org/packages/f3/96/3f426860119054a2c1d3793d223d6f3a29ba4957a117ad9d6b2dbff8091e/${WHEEL}"
SRC_URI[sha256sum] = "7f25a67921657f4014f37175bd9829a78cb8f58202e034d14941a255c00c0c15"
S = "${UNPACKDIR}"

inherit python3-dir

DEPENDS += "unzip-native"

do_configure[noexec] = "1"
do_compile[noexec] = "1"

# No deps: Requires-Dist only serves ncnn.model_zoo, which `import ncnn` never loads.
do_install() {
    install -d ${D}${PYTHON_SITEPACKAGES_DIR}
    unzip -q ${UNPACKDIR}/${WHEEL} -d ${D}${PYTHON_SITEPACKAGES_DIR}
}

FILES:${PN} = "${PYTHON_SITEPACKAGES_DIR}"
INSANE_SKIP:${PN} += "already-stripped"
# auditwheel renamed the vendored libgomp's file but kept its SONAME (libgomp.so.1), so rpm's
# per-file deps on libgomp-<hash>.so have no provider. It loads through the .so's RPATH.
SKIP_FILEDEPS:${PN} = "1"
RDEPENDS:${PN} = "python3-core"
