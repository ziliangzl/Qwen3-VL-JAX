import os

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_use_shardy_partitioner", True)

from model import DecoderBlock, build_text_rope

MEGATRON_TP_DUMP_DIR = "shardy_dumps_megatron_tp"
MEGATRON_TP_FILES = {
    "xla_entry_hlo": "megatron_tp_jit_sharded_forward_xla_entry_hlo.txt",
    "post_compile_hlo": "megatron_tp_decoder_post_compile_xla_hlo.txt",
    "stablehlo_shardy": "megatron_tp_decoder_stablehlo_shardy.mlir",
}

dump_path = os.path.abspath(os.path.join(".", MEGATRON_TP_DUMP_DIR))
os.makedirs(dump_path, exist_ok=True)
jax.config.update("jax_dump_ir_to", dump_path)
# jax_dump_ir_to 只覆盖 JAX 侧 MLIR 管线（含 StableHLO / sdy），不会记录 XLA 内部
# StableHLO→经典 HLO 的 lowering 与后续优化在 XLA 里完成。要 dump 那些阶段请用：
#   - lowered.as_text(dialect="hlo")：刚喂给 XLA 的 HLO（未跑 XLA 优化）
#   - compiled.runtime_executable().hlo_modules()：编译完成后的 HLO
#   - 或 compile(..., xla_dump_to=...) / XLA_FLAGS=--xla_dump_to=... --xla_dump_hlo_as_text
#     得到每个 XLA pass 前后的 HLO 文件

batch, seq_len, hidden_size = 8, 1024, 4096
num_heads, num_kv_heads = 32, 8
head_dim = 128
intermediate_size = 11008

devices = mesh_utils.create_device_mesh((1, 8))
mesh = Mesh(devices, axis_names=("dp", "tp"))
TP_AXIS = "tp"
TP_SIZE = 8

# Megatron 式 TP：列并行（q/k/v、gate/up）要求 hidden 在输入侧复制；
# 行并行（o_proj、down）在输出侧 all-reduce 后与残差对齐。下列维度需能被 TP 宽度整除。
assert hidden_size % TP_SIZE == 0, "hidden_size must divide tensor-parallel size"
assert (num_heads * head_dim) % TP_SIZE == 0, "Q output dim must divide TP size"
assert (num_kv_heads * head_dim) % TP_SIZE == 0, "K/V output dim must divide TP size"
assert intermediate_size % TP_SIZE == 0, "intermediate_size must divide TP size"


def test_shardy_dialect_ir() -> None:
    model = DecoderBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        rope_section=[64],
        eps=1e-6,
        tp_axis=TP_AXIS,
    )

    key = jax.random.PRNGKey(0)
    x = jnp.ones((batch, seq_len, hidden_size), dtype=jnp.bfloat16)
    positions = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch, seq_len)
    )
    cos, sin = build_text_rope(positions, [64], 1_000_000.0, jnp.bfloat16)
    variables = model.init(key, x, cos, sin)

    replicated_hidden = NamedSharding(mesh, P(None, None, None))

    @jax.jit
    def sharded_forward(params, x, cos, sin):
        x = jax.lax.with_sharding_constraint(x, replicated_hidden)
        out = model.apply(params, x, cos, sin, mesh=mesh)
        return jax.lax.with_sharding_constraint(out, replicated_hidden)

    print("正在生成 Lowered IR (MLIR 格式)...")

    lowered = sharded_forward.lower(variables, x, cos, sin)

    hlo_at_xla_entry = lowered.as_text(dialect="hlo", debug_info=True)
    entry_hlo = os.path.join(dump_path, MEGATRON_TP_FILES["xla_entry_hlo"])
    with open(entry_hlo, "w") as f:
        f.write(hlo_at_xla_entry)

    # 可选：编译时让 XLA 把每个 pass 的 HLO 落到目录（体积极大时慎用）
    # compiled = lowered.compile({
    #     "xla_dump_to": os.path.join(dump_path, "megatron_tp_xla_pass_dumps"),
    #     "xla_dump_hlo_as_text": True,
    # })
    compiled = lowered.compile()
    compiled_text = compiled.runtime_executable().hlo_modules()[0].to_string()
    compiled_out = os.path.join(dump_path, MEGATRON_TP_FILES["post_compile_hlo"])
    with open(compiled_out, "w") as f:
        f.write(compiled_text)

    mlir_module = lowered.compiler_ir(dialect="stablehlo")

    print("\n--- MLIR with Shardy Dialect ---")
    mlir_str = str(mlir_module)

    if "sdy" in mlir_str:
        print("检测到 Shardy Dialect (sdy)！")
    else:
        print("未检测到 sdy，请确保已安装 shardy 库并正确配置后端。")

    mlir_out = os.path.join(dump_path, MEGATRON_TP_FILES["stablehlo_shardy"])
    with open(mlir_out, "w") as f:
        f.write(mlir_str)

    print(f"\n[成功] Megatron 式 TP dump 已写入目录: {dump_path}")
    print(f"  - JAX→XLA 入口 HLO: {entry_hlo}")
    print(f"  - StableHLO+Shardy: {mlir_out}")
    print(f"  - 编译后 XLA HLO: {compiled_out}")


if __name__ == "__main__":
    test_shardy_dialect_ir()
