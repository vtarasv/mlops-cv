SUMMARY = "OpenTelemetry Python proto"
HOMEPAGE = "https://github.com/open-telemetry/opentelemetry-python"
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://LICENSE;md5=86d3f3a95c324c9479bd8986968f4327"

PYPI_PACKAGE = "opentelemetry_proto"
SRC_URI[sha256sum] = "224778df17e1f3fafeaaa21d874236ca5f6ffc2f86e0899298ec7351aac27924"

inherit pypi python_hatchling

RDEPENDS:${PN} = "python3-protobuf \
    python3-core python3-datetime python3-numbers python3-pickle"
