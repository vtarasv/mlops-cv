SUMMARY = "OpenTelemetry semantic conventions"
HOMEPAGE = "https://github.com/open-telemetry/opentelemetry-python"
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://LICENSE;md5=86d3f3a95c324c9479bd8986968f4327"

PYPI_PACKAGE = "opentelemetry_semantic_conventions"
SRC_URI[sha256sum] = "72f76fb2d1582d9d033dd1fcd84532e961e6ff3d90d24ba6fabc72975a83864c"

inherit pypi python_hatchling

RDEPENDS:${PN} = "python3-opentelemetry-api python3-typing-extensions python3-core"
