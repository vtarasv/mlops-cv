SUMMARY = "MLflow tracking and registry client (skinny)"
HOMEPAGE = "https://mlflow.org"
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://LICENSE.txt;md5=c425c9f82a43ca9d7a32285832427f2d"

PYPI_PACKAGE = "mlflow_skinny"
SRC_URI[sha256sum] = "e50f4506422c7737157ae6643c165122af7898345f2e828fa93c4f10128653cf"

inherit pypi python_setuptools_build_meta

# Trimmed to the measured import set: no gitpython (and so no git), uvicorn or cloudpickle.
RDEPENDS:${PN} = "python3-cachetools python3-click python3-databricks-sdk python3-fastapi \
    python3-importlib-metadata python3-opentelemetry-api python3-opentelemetry-proto \
    python3-opentelemetry-sdk python3-packaging python3-protobuf python3-pydantic \
    python3-python-dotenv python3-pyyaml python3-requests python3-sqlparse python3-starlette \
    python3-typing-extensions \
    python3-core python3-asyncio python3-compression python3-crypt python3-ctypes \
    python3-datetime python3-difflib python3-email python3-html python3-image python3-io \
    python3-json python3-logging python3-math python3-mime python3-misc python3-multiprocessing \
    python3-netclient python3-netserver python3-numbers python3-pickle python3-pprint \
    python3-pydoc python3-shell python3-threading python3-unittest python3-unixadmin \
    python3-zoneinfo"
