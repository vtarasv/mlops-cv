SUMMARY = "Databricks SDK for Python (imported by mlflow at module level)"
HOMEPAGE = "https://github.com/databricks/databricks-sdk-py"
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://LICENSE;md5=368836afff094ebadae0b2c89dbf71d6"

PYPI_PACKAGE = "databricks_sdk"
SRC_URI[sha256sum] = "40e56ad56a8afb8150d4152c9fc7bf474ba1d44b6315ac4d9a5b18d57fd29900"

inherit pypi python_setuptools_build_meta

# Trimmed to the measured import set: no google-auth (and so no cryptography).
RDEPENDS:${PN} = "python3-protobuf python3-requests python3-urllib3 \
    python3-core python3-asyncio python3-compression python3-crypt python3-datetime \
    python3-email python3-html python3-io python3-json python3-logging python3-math python3-mime \
    python3-misc python3-netclient python3-netserver python3-numbers python3-pickle \
    python3-shell python3-threading python3-xml"
