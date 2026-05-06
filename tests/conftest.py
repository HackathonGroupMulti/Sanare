import os

os.environ["SANARE_ENABLE_OTEL"] = "0"
os.environ["LLM_PROVIDER"] = "offline"
os.environ["NVIDIA_API_KEY"] = ""
os.environ["NVIDIA_NIM_BASE_URL"] = ""
os.environ["LLM_BASE_URL"] = ""
os.environ["VLLM_BASE_URL"] = ""
