# requests takes chardet or charset-normalizer; chardet is 24 MiB on 3.14, charset-normalizer is
# what pip installs (and what uv.lock pins).
RDEPENDS:${PN}:remove = "python3-chardet"
RDEPENDS:${PN} += "python3-charset-normalizer"
