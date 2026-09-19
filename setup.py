"""Setup configuration for SceneMotion-LLM."""

from pathlib import Path

from setuptools import find_packages, setup


def read_requirements():
    """Use the repository's single runtime dependency list."""
    requirements_file = Path(__file__).parent / "scene_motion_llm" / "requirements.txt"
    return [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


requirements = read_requirements()

extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
    ]
}

setup(
    name="scene-motion-llm",
    version="2.0.0",
    description="Video sentiment classification from scene appearance and motion",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "scene-motion-train=scene_motion_llm.train_pipeline:main",
            "scene-motion-infer=scene_motion_llm.inference_llm:main",
        ]
    },
    include_package_data=True,
    long_description="Video sentiment classification from scene appearance and motion",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ]
)
