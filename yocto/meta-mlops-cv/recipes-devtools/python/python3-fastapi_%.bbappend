# fastapi on PyPI doesn't require fastapi-cli (only fastapi[standard] does); it would pull uvicorn.
RDEPENDS:${PN}:remove = "python3-fastapi-cli"
