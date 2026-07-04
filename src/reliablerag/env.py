import os
import sys

_KEYS = ("PROVIDER", "EMBEDDING_MODEL", "GENERATOR_MODEL", "JUDGE_MODEL", "CHROMA_PERSIST_DIR", "HF_TOKEN")


def load_secrets() -> None:
    if "google.colab" in sys.modules:
        from google.colab import userdata  # type: ignore[import-untyped]
        for key in _KEYS:
            os.environ[key] = userdata.get(key)
    elif "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/input"):
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-untyped]
        _secrets = UserSecretsClient()
        for key in _KEYS:
            os.environ[key] = _secrets.get_secret(key)
    else:
        from dotenv import load_dotenv
        load_dotenv(override=True)
