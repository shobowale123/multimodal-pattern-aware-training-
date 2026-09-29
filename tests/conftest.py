import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def bounded_cpu_threads():
    """Small synthetic batches run faster with one CPU worker per operation."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
