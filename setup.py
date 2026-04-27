from setuptools import setup, find_packages

setup(
    name="agentic-threat-detection",
    version="0.1.0",
    author="Awodola Olusola Ebenezer",
    author_email="oluebenawodola@gmail.com",
    description="Autonomous cyber defense via deep learning + reinforcement learning",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
    ],
)
