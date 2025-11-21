from setuptools import setup, find_packages

setup(
    name="doc_parse_convert",
    version="0.6.1",
    author="Guthman",
    description="Utilities for document content extraction and conversion",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "google-cloud-aiplatform>=1.86.0,<2.0.0",
        "pymupdf>=1.19.0,<2.0.0",
        "google-api-core>=2.0.0,<3.0.0",
        "google-auth>=2.0.0,<3.0.0",
        "google-auth-oauthlib>=0.4.0,<2.0.0",
        "pillow>=9.0.0,<13.0.0",
        "tenacity>=8.0.0,<10.0.0",
        "ebooklib>=0.17.0,<1.0.0",
        "beautifulsoup4>=4.9.0,<5.0.0",
        "requests>=2.25.0,<3.0.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
