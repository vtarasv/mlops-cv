SUMMARY = "OpenTelemetry Python API"
HOMEPAGE = "https://github.com/open-telemetry/opentelemetry-python"
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://LICENSE;md5=86d3f3a95c324c9479bd8986968f4327"

PYPI_PACKAGE = "opentelemetry_api"
SRC_URI[sha256sum] = "107d0d03857ea8fc7c5fcbbbd83f800c281f0d560553d61c1d675fccfd1761c1"

inherit pypi python_hatchling

RDEPENDS:${PN} = "python3-typing-extensions \
    python3-core python3-logging python3-netclient python3-numbers"
