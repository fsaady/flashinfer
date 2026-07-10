# Test transparent TRT-LLM P2P allreduce fusion dispatch in the unified API.

import multiprocessing as mp
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm.allreduce as allreduce_mod
from flashinfer.comm.allreduce import (
    allreduce_fusion,
    create_allreduce_fusion_workspace,
)
from flashinfer.comm.cuda_ipc import _find_loaded_library_from_maps
from flashinfer.comm.mnnvl import TorchDistBackend
from flashinfer.comm.trtllm_ar import (
    AllReduceFusionPattern,
    AllReduceStrategyConfig,
)

_TRTLLM_WORKSPACE_CLASS = allreduce_mod.TRTLLMAllReduceFusionWorkspace


def test_cudart_lookup_skips_stub_mappings(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text(
        "\n".join(
            [
                "7f00-7f01 r--p 00000000 00:00 1 /opt/cuda/lib64/stubs/libcudart.so",
                "7f02-7f03 r-xp 00000000 00:00 2 /opt/cuda/lib64/libcudart.so.12",
            ]
        )
    )

    assert (
        _find_loaded_library_from_maps("libcudart", str(maps))
        == "/opt/cuda/lib64/libcudart.so.12"
    )


def _fake_p2p_workspace(world_size=8, rank=0):
    workspace = allreduce_mod._TRTLLMP2PAllReduceFusionWorkspace.__new__(
        allreduce_mod._TRTLLMP2PAllReduceFusionWorkspace
    )
    workspace.world_size = world_size
    workspace.rank = rank
    workspace.config_code = AllReduceStrategyConfig.PUSH_MODE
    workspace._destroyed = True
    workspace.workspace_tensor = torch.zeros(world_size * 3 + 1, dtype=torch.int64)
    workspace._flag_ptr = 0
    workspace.metadata = {
        "tp_rank": rank,
        "tp_size": world_size,
        "max_token_num": 64,
        "hidden_dim": 16,
        "use_fp32_lamport": True,
    }
    workspace.ipc_handles = []
    workspace.is_buffer_size_sufficient = lambda *_, **__: True
    workspace.destroy = lambda: None
    return workspace


def _fake_trtllm_workspace(world_size=8, rank=0, use_p2p=False):
    workspace = _TRTLLM_WORKSPACE_CLASS.__new__(_TRTLLM_WORKSPACE_CLASS)
    workspace.world_size = world_size
    workspace.rank = rank
    workspace._use_p2p = use_p2p
    workspace._destroyed = True
    workspace.is_buffer_size_sufficient = lambda *_, **__: True
    workspace.destroy = lambda: None
    if use_p2p:
        workspace._p2p_workspace = _fake_p2p_workspace(world_size, rank)
        workspace.metadata = workspace._p2p_workspace.metadata
    else:
        workspace._p2p_workspace = None
        workspace.workspace_tensor = torch.zeros(1, dtype=torch.int64)
        workspace.metadata = {}
    return workspace


def test_trtllm_rejects_unsupported_p2p_workspace_dtype(monkeypatch):
    monkeypatch.setattr(allreduce_mod, "is_cuda_multicast_supported", lambda: False)
    monkeypatch.setattr(
        allreduce_mod, "_all_ranks_on_same_node", lambda *_, **__: True
    )
    monkeypatch.setattr(
        allreduce_mod, "_all_ranks_have_cuda_peer_access", lambda *_, **__: True
    )

    with pytest.raises(ValueError, match="trtllm allreduce fusion supports"):
        create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=8,
            rank=0,
            max_token_num=32,
            hidden_dim=16,
            dtype=torch.int8,
            gpus_per_node=8,
        )


def test_trtllm_allows_p2p_workspace_when_max_tokens_use_oneshot(
    monkeypatch,
):
    monkeypatch.setattr(allreduce_mod, "is_cuda_multicast_supported", lambda: False)
    monkeypatch.setattr(
        allreduce_mod, "_all_ranks_on_same_node", lambda *_, **__: True
    )
    monkeypatch.setattr(
        allreduce_mod, "_all_ranks_have_cuda_peer_access", lambda *_, **__: True
    )
    created = []

    def fake_trtllm_ctor(**kwargs):
        created.append(kwargs)
        return _fake_trtllm_workspace(
            world_size=kwargs["tp_size"],
            rank=kwargs["tp_rank"],
            use_p2p=kwargs["use_p2p"],
        )

    monkeypatch.setattr(
        allreduce_mod, "TRTLLMAllReduceFusionWorkspace", fake_trtllm_ctor
    )

    workspace = create_allreduce_fusion_workspace(
        backend="trtllm",
        world_size=8,
        rank=0,
        max_token_num=8,
        hidden_dim=16,
        dtype=torch.float16,
        gpus_per_node=8,
    )

    assert workspace.backend == "trtllm"
    assert workspace._use_p2p is True
    assert created[0]["use_p2p"] is True


def test_auto_selects_trtllm_p2p_fallback_when_multicast_is_unavailable(monkeypatch):
    created = []

    def fake_trtllm_ctor(**kwargs):
        created.append(("trtllm", kwargs))
        return _fake_trtllm_workspace(
            world_size=kwargs["tp_size"],
            rank=kwargs["tp_rank"],
            use_p2p=kwargs["use_p2p"],
        )

    monkeypatch.setattr(allreduce_mod, "is_cuda_multicast_supported", lambda: False)
    monkeypatch.setattr(
        allreduce_mod, "_all_ranks_on_same_node", lambda *_, **__: True
    )
    monkeypatch.setattr(
        allreduce_mod, "_all_ranks_have_cuda_peer_access", lambda *_, **__: True
    )
    monkeypatch.setattr(
        allreduce_mod, "TRTLLMAllReduceFusionWorkspace", fake_trtllm_ctor
    )

    workspace = create_allreduce_fusion_workspace(
        backend="auto",
        world_size=4,
        rank=1,
        max_token_num=64,
        hidden_dim=6144,
        dtype=torch.bfloat16,
        gpus_per_node=4,
    )

    assert workspace.backend == "trtllm"
    assert workspace._use_p2p is True
    assert [name for name, _ in created] == ["trtllm"]
    assert created[0][1]["use_p2p"] is True


def test_auto_rejects_trtllm_for_multi_node_group_without_multicast(monkeypatch):
    monkeypatch.setattr(allreduce_mod, "is_cuda_multicast_supported", lambda: False)
    monkeypatch.setattr(
        allreduce_mod, "_all_ranks_on_same_node", lambda *_, **__: False
    )
    monkeypatch.setattr(
        allreduce_mod, "_all_ranks_have_cuda_peer_access", lambda *_, **__: True
    )
    monkeypatch.setattr(
        allreduce_mod,
        "TRTLLMAllReduceFusionWorkspace",
        lambda **_: pytest.fail("trtllm workspace should not be created"),
    )

    with pytest.raises(ValueError, match="No suitable backend"):
        create_allreduce_fusion_workspace(
            backend="auto",
            world_size=8,
            rank=1,
            max_token_num=64,
            hidden_dim=6144,
            dtype=torch.bfloat16,
            gpus_per_node=4,
        )


def test_auto_rejects_trtllm_when_peer_access_is_unavailable(monkeypatch):
    monkeypatch.setattr(allreduce_mod, "is_cuda_multicast_supported", lambda: False)
    monkeypatch.setattr(
        allreduce_mod, "_all_ranks_on_same_node", lambda *_, **__: True
    )
    monkeypatch.setattr(
        allreduce_mod, "_all_ranks_have_cuda_peer_access", lambda *_, **__: False
    )
    monkeypatch.setattr(
        allreduce_mod,
        "TRTLLMAllReduceFusionWorkspace",
        lambda **_: pytest.fail("trtllm workspace should not be created"),
    )

    with pytest.raises(ValueError, match="No suitable backend"):
        create_allreduce_fusion_workspace(
            backend="auto",
            world_size=4,
            rank=1,
            max_token_num=64,
            hidden_dim=6144,
            dtype=torch.bfloat16,
            gpus_per_node=4,
        )


def test_explicit_trtllm_rejects_missing_peer_access_without_multicast(monkeypatch):
    monkeypatch.setattr(allreduce_mod, "is_cuda_multicast_supported", lambda: False)
    monkeypatch.setattr(
        allreduce_mod, "_all_ranks_on_same_node", lambda *_, **__: True
    )
    monkeypatch.setattr(
        allreduce_mod, "_all_ranks_have_cuda_peer_access", lambda *_, **__: False
    )
    monkeypatch.setattr(
        allreduce_mod,
        "TRTLLMAllReduceFusionWorkspace",
        lambda **_: pytest.fail("trtllm workspace should not be created"),
    )

    with pytest.raises(ValueError, match="CUDA peer access"):
        create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=4,
            rank=0,
            max_token_num=64,
            hidden_dim=6144,
            dtype=torch.bfloat16,
            gpus_per_node=4,
        )


def test_trtllm_p2p_dispatches_oneshot_by_auto_heuristic(monkeypatch):
    calls = []

    def fake_trtllm_allreduce_fusion(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        allreduce_mod, "trtllm_allreduce_fusion", fake_trtllm_allreduce_fusion
    )
    workspace = _fake_trtllm_workspace(use_p2p=True)
    input_tensor = torch.randn(32, 16, dtype=torch.float32)
    output = torch.empty_like(input_tensor)

    result = allreduce_fusion(
        input=input_tensor,
        workspace=workspace,
        pattern=AllReduceFusionPattern.kAllReduce,
        output=output,
        launch_with_pdl=True,
    )

    assert result is output
    assert len(calls) == 1
    call = calls[0]
    assert call["use_oneshot"] is True
    assert call["pattern_code"] == AllReduceFusionPattern.kAllReduce
    assert call["world_size"] == 8
    assert call["world_rank"] == 0
    assert call["token_num"] == 32
    assert call["hidden_dim"] == 16
    assert call["launch_with_pdl"] is True
    assert call["workspace_ptrs"] is workspace._p2p_workspace.workspace_tensor
    assert call["metadata"] is workspace._p2p_workspace.metadata
    assert call["allreduce_in"].shape == (32 * 16,)
    assert call["allreduce_out"].shape == (32 * 16,)


def test_trtllm_workspace_routes_supported_shape_to_p2p(monkeypatch):
    calls = []

    def fake_dispatch(**kwargs):
        calls.append(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(
        allreduce_mod, "_dispatch_trtllm_p2p_allreduce_fusion", fake_dispatch
    )
    workspace = _fake_trtllm_workspace(world_size=4, use_p2p=True)
    input_tensor = torch.randn(8, 16, dtype=torch.float32)
    output = torch.empty_like(input_tensor)

    result = allreduce_fusion(
        input=input_tensor,
        workspace=workspace,
        pattern=AllReduceFusionPattern.kAllReduce,
        output=output,
    )

    assert result is output
    assert len(calls) == 1
    assert calls[0]["workspace"] is workspace._p2p_workspace


def test_trtllm_p2p_workspace_capacity_uses_p2p_metadata():
    workspace = _fake_trtllm_workspace(world_size=4, use_p2p=True)

    assert (
        workspace.is_buffer_size_sufficient(
            4, 8, 16, torch.float32, use_oneshot=False
        )
        is True
    )


def test_trtllm_p2p_dispatches_residual_rmsnorm(monkeypatch):
    calls = []

    def fake_trtllm_allreduce_fusion(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        allreduce_mod, "trtllm_allreduce_fusion", fake_trtllm_allreduce_fusion
    )
    workspace = _fake_trtllm_workspace(use_p2p=True)
    input_tensor = torch.randn(32, 16, dtype=torch.float32)
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn(16, dtype=torch.float32)

    result = allreduce_fusion(
        input=input_tensor,
        workspace=workspace,
        pattern=AllReduceFusionPattern.kARResidualRMSNorm,
        residual_in=residual,
        rms_gamma=gamma,
        rms_eps=1e-5,
        use_oneshot=False,
    )

    assert result.shape == input_tensor.shape
    assert len(calls) == 1
    call = calls[0]
    assert call["use_oneshot"] is False
    assert call["pattern_code"] == AllReduceFusionPattern.kARResidualRMSNorm
    assert call["rms_gamma"] is gamma
    assert call["rms_eps"] == 1e-5
    assert call["residual_in"].shape == (32 * 16,)
    assert call["residual_out"].shape == (32 * 16,)
    assert call["norm_out"].shape == (32 * 16,)
    assert call["allreduce_out"] is None


def test_trtllm_p2p_rejects_quantized_patterns_before_launch(monkeypatch):
    monkeypatch.setattr(
        allreduce_mod,
        "trtllm_allreduce_fusion",
        lambda **_: pytest.fail("kernel launcher should not be called"),
    )
    workspace = _fake_trtllm_workspace(use_p2p=True)
    input_tensor = torch.randn(32, 16, dtype=torch.float32)

    with pytest.raises(ValueError, match="P2P fallback currently supports only"):
        allreduce_fusion(
            input=input_tensor,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
        )


def test_trtllm_p2p_rejects_unsafe_small_token_strategies(monkeypatch):
    calls = []

    monkeypatch.setattr(
        allreduce_mod,
        "trtllm_allreduce_fusion",
        lambda **kwargs: calls.append(kwargs),
    )
    workspace = _fake_trtllm_workspace(world_size=8, use_p2p=True)

    result = allreduce_fusion(
        input=torch.randn(16, 16, dtype=torch.float32),
        workspace=workspace,
        pattern=AllReduceFusionPattern.kAllReduce,
        output=torch.empty(16, 16, dtype=torch.float32),
        use_oneshot=True,
    )

    assert result.shape == (16, 16)
    assert calls[0]["use_oneshot"] is True

    with pytest.raises(ValueError, match="twoshot requires"):
        allreduce_fusion(
            input=torch.randn(8, 16, dtype=torch.float32),
            workspace=workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
            output=torch.empty(8, 16, dtype=torch.float32),
            use_oneshot=False,
        )

    result = allreduce_fusion(
        input=torch.randn(8, 16, dtype=torch.float32),
        workspace=workspace,
        pattern=AllReduceFusionPattern.kAllReduce,
        output=torch.empty(8, 16, dtype=torch.float32),
        use_oneshot=True,
    )
    assert result.shape == (8, 16)
    assert calls[-1]["use_oneshot"] is True


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def multi_process_parallel(
    world_size: int,
    test_target: Any,
    target_args: tuple = (),
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for rank in range(world_size):
        proc_args = (world_size, rank, distributed_init_port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{rank}")
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()
        assert proc.exitcode == 0, (
            f"Process {proc.name} failed with exit code {proc.exitcode}"
        )


def _run_actual_p2p_worker(
    world_size: int,
    rank: int,
    distributed_init_port: int,
    token_num: int,
    use_oneshot: bool,
    pattern: AllReduceFusionPattern,
):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )

    dtype = torch.float16
    hidden_dim = 1024
    workspace = None
    try:
        workspace = create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=max(token_num, world_size + 1),
            hidden_dim=hidden_dim,
            dtype=dtype,
            gpus_per_node=torch.cuda.device_count(),
            comm_backend=TorchDistBackend(),
        )
        if not getattr(workspace, "_use_p2p", False):
            raise RuntimeError("expected TRT-LLM workspace to select P2P fallback")

        input_tensor = torch.full(
            (token_num, hidden_dim),
            rank + 1,
            dtype=dtype,
            device=f"cuda:{rank}",
        )
        input_ref = input_tensor.clone()
        dist.all_reduce(input_ref)

        if pattern == AllReduceFusionPattern.kAllReduce:
            output = torch.empty_like(input_tensor)
            result = allreduce_fusion(
                input=input_tensor,
                workspace=workspace,
                pattern=pattern,
                output=output,
                use_oneshot=use_oneshot,
                launch_with_pdl=False,
            )
            torch.testing.assert_close(
                result.to(torch.float32),
                input_ref.to(torch.float32),
                atol=1e-3,
                rtol=1e-3,
            )
        else:
            residual = torch.full_like(input_tensor, 0.25)
            gamma = torch.ones(hidden_dim, dtype=dtype, device=f"cuda:{rank}")
            residual_out = torch.empty_like(input_tensor)
            norm_out = torch.empty_like(input_tensor)
            result = allreduce_fusion(
                input=input_tensor,
                workspace=workspace,
                pattern=pattern,
                residual_in=residual,
                residual_out=residual_out,
                norm_out=norm_out,
                rms_gamma=gamma,
                rms_eps=1e-5,
                use_oneshot=use_oneshot,
                launch_with_pdl=False,
            )

            ref_residual = input_ref + residual
            ref_norm = ref_residual.to(torch.float32) * torch.rsqrt(
                ref_residual.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
                + 1e-5
            )
            torch.testing.assert_close(
                residual_out.to(torch.float32),
                ref_residual.to(torch.float32),
                atol=1e-3,
                rtol=1e-3,
            )
            torch.testing.assert_close(
                result.to(torch.float32),
                ref_norm,
                atol=2e-3,
                rtol=2e-3,
            )
    finally:
        if workspace is not None:
            workspace.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize(
    "pattern",
    [
        AllReduceFusionPattern.kAllReduce,
        AllReduceFusionPattern.kARResidualRMSNorm,
    ],
)
@pytest.mark.parametrize("token_num,use_oneshot", [(1, True), (17, False)])
def test_actual_trtllm_p2p_correctness_skips_on_multicast(
    token_num: int,
    use_oneshot: bool,
    pattern: AllReduceFusionPattern,
):
    world_size = 4
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires at least {world_size} CUDA devices")
    if allreduce_mod.is_cuda_multicast_supported():
        pytest.skip("P2P fallback is not selected on CUDA multicast-capable nodes")
    if not allreduce_mod._all_ranks_have_cuda_peer_access(world_size):
        pytest.skip("requires CUDA peer access between all participating devices")

    multi_process_parallel(
        world_size,
        _run_actual_p2p_worker,
        target_args=(token_num, use_oneshot, pattern),
    )
