SUMMARY = "OpenTelemetry Python SDK"
HOMEPAGE = "https://github.com/open-telemetry/opentelemetry-python"
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://LICENSE;md5=86d3f3a95c324c9479bd8986968f4327"

PYPI_PACKAGE = "opentelemetry_sdk"
SRC_URI[sha256sum] = "d8187c81c162df9913e4003dd6485f7390d9a24fc17026ec7387b8b8218b08e9"

inherit pypi python_hatchling

RDEPENDS:${PN} = "python3-opentelemetry-api python3-opentelemetry-semantic-conventions \
    python3-typing-extensions \
    python3-core python3-asyncio python3-datetime python3-io python3-json python3-logging \
    python3-math python3-netclient python3-numbers"
