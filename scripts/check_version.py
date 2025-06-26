import importlib.metadata

packages = [
    "fastapi",
    "uvicorn",
    "torch",
    "scikit-learn",
    "joblib",
    "numpy",
    "pandas"
]

for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} is not installed")
