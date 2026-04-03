import requests
from bs4 import BeautifulSoup
import torch
import sys
import platform

py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_ver, k2_device = torch.__version__.split("+")
os_family = platform.system()
if os_family == "Linux":
    base_url = (
        "https://k2-fsa.github.io/k2/installation/pre-compiled-cuda-wheels-linux/"
    )
    k2_device = k2_device[: len("cu12")] + "." + k2_device[len("cu12") :]
    k2_device = k2_device.replace("cu", "cuda")
elif os_family == "Windows":
    base_url = (
        "https://k2-fsa.github.io/k2/installation/pre-compiled-cpu-wheels-windows/"
    )
    k2_device = "cpu"
elif os_family == "Darwin":
    base_url = "https://k2-fsa.github.io/k2/installation/pre-compiled-cpu-wheels-macos/"
    k2_device = "cpu"
else:
    raise NotImplementedError(
        f"Only support Linux, Windows or MacOS, found {os_family}"
    )

index_page = BeautifulSoup(requests.get(base_url).text, "html.parser")
whl_url = None
for a_el in BeautifulSoup(
    requests.get(base_url + torch_ver).text, "html.parser"
).select("a.external"):
    _whl_url = a_el.get("href")
    if k2_device in _whl_url and py_version in _whl_url:
        whl_url = _whl_url
        break

if whl_url is None:
    raise RuntimeError(
        f"k2 wheel for torch {torch_ver}, {k2_device} on {py_version} not found"
    )

print(f"k2 @ {whl_url}")
