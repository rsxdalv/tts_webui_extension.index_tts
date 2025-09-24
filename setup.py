import setuptools
from pathlib import Path

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setuptools.setup(
    name="tts_webui_extension.index_tts",
    packages=setuptools.find_namespace_packages(),
    version="0.0.2",
    author="Your Name",
    description="A template extension for TTS Generation WebUI",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rsxdalv/tts_webui_extension.index_tts",
    project_urls={},
    scripts=[],
    install_requires=[
        "tts-webui.index-tts>=2.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
