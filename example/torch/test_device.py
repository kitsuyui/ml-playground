# ja: device の互換性のテスト
# en: test device compatibility

import sys
from typing import Iterator

import pytest
import torch
from torch import Tensor


def test_backends() -> None:
    # TODO: other platforms
    if sys.platform == "darwin":
        assert not torch.backends.cudnn.is_available()
        assert not torch.backends.mkl.is_available()
        assert not torch.backends.mkldnn.is_available()
        assert not torch.backends.openmp.is_available()
        assert torch.backends.mps.is_available()


def get_devices() -> Iterator[torch.device]:
    # TODO: other platforms
    if sys.platform == "darwin":
        # get all devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                yield torch.device(f"cuda:{i}")
        elif torch.backends.mps.is_available():
            yield torch.device("mps")
        yield torch.device("cpu")


def test_device_equality() -> None:
    # TODO: other platforms
    if sys.platform == "darwin":
        devices = list(get_devices())
        device1 = devices[0]
        device2 = devices[1]
        assert device1 != device2


def test_device_instance_equality() -> None:
    # TODO: other platforms
    if sys.platform == "darwin":
        devices1 = list(get_devices())
        devices2 = list(get_devices())
        # check that the devices are the same in both lists)
        for device1, device2 in zip(devices1, devices2):
            assert device1 == device2, "device1 != device2"
            assert device1 is not device2, "device1 is device2"


def test_device_compatibility() -> None:
    # TODO: other platforms
    if sys.platform == "darwin":
        devices = list(get_devices())
        device1 = devices[0]
        device2 = devices[1]
        tensor1 = Tensor([1, 2, 3]).to(device1)
        tensor2 = Tensor([1, 2, 3]).to(device2)
        with pytest.raises(RuntimeError) as e:
            tensor1 += tensor2
        assert "Expected all tensors to be on the same device" in str(e.value)

        with pytest.raises(RuntimeError) as e:
            tensor2 += tensor1
        assert "Expected all tensors to be on the same device" in str(e.value)


def test_default_device() -> None:
    # TODO: other platforms
    if sys.platform == "darwin":
        devices = list(get_devices())
        device1 = devices[0]
        tensor1 = Tensor([1, 2, 3]).to(device1)
        tensor2 = Tensor([1, 2, 3])  # default device

        with pytest.raises(RuntimeError) as e:
            tensor1 += tensor2
        assert "Expected all tensors to be on the same device" in str(e.value)
