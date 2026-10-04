"""KP 骨架自检 —— 纯 CPU、夜间安全、无外部依赖。

运行：
    python -m kp.selftest          （在 D:/model 下）
    python kp/selftest.py

逐项验证设计稿里的**可验收不变量**：
    1. latent 打包/解包 往返一致（fp32 / bf16）+ 通道分离监督件
    2. ★ 门控全 0 ⇒ 与裸模型 **bit-exact**
    3. ★ Δ-Pack 谱检查：子空间初始化【合格】/ 正交扰动【不合格】
    4. ★ 擦除 `E⁻¹∘E` 可逆（KL ≈ 0）
    5. ∥-Pack 零初始化 + 短路（不可表示为 ΔW）
    6. 主干前向形状 + 挂包前后 bit-exact + VAE 32× + 文本塔接口
    7. 角色卡 + CharacterFitter 输出身份 token
    8. Rectified Flow + Matryoshka 采样可跑且可复现
    9. NVFP4 模拟量化：与参考实现对拍 + STE 可微 + 量化开关
   10. 能力包 落盘/加载 往返（save_adapter → load_adapter）
   11. QAD：冻结主干 + 只训 Δ-Pack（loss 下降且梯度只进包）
   12. SVDPack：子空间约束（按构造通过谱检查）+ 跨版本可迁移
   13. CharaBridge：身份+几何双分支，关断返回 None（bit-exact）
   14. Layout Planner：中文折行/避头尾/竖排 + ROIBranch 低压缩分支
   15. caption 语料审计：语言配比 / 标签串检测 / 汉字覆盖（P1.8）
   16. Axis Probe：G3.5 四测（单调/正交/可逆/低比特行程）—— 装置本身先过对照样本
   17. 多视角配对：声明式单旋钮配对 / 拒绝猜测 / 缺口清单（角色卡数据线）
   18. Character Fitter 训练：配对驱动的对比目标 / 留出视角泛化 / 负对照（可证伪）
   19. ★ G2 通道分离：尺子双向校验 + 显式监督净收益（mix 对照）+ 退化输入报缺口
   20. ★ G3.5 **真探针**：接在真主干 + 真轴注入路径（domain→domain_embed→adaLN）；
       adaLN-Zero 不变量 / **非空性守卫**（死轴不许假通过 ①）/ ④ 真过量化器 / 负对照
   21. G3 **Sigmoid 注意力**（机制层装置）：one-hot v 反解精确权重 / 已知答案（等 logit ⇒
       α=1）/ **平移不变性负对照**（softmax 恒定 vs sigmoid 零点敏感）/ 3:1 构成 / L311 离群值
       ⛔ 只验算子层（随机权重），**不能**替代 G6 后的 benchmark 级验收
"""
from __future__ import annotations

import sys

# 🔴🔴 验收基线守卫（2026-10-03 全局审查发现 · 见 design v1.16 §勘误④）
#
# 本文件的 `check()` 靠**捕获 AssertionError** 判失败，而检查体内部大量使用裸 `assert`
# （实测 171 处裸 assert / 仅 3 处 raise）。Python 的 `-O` 会把 `assert` 语句**整体移除**
# ⇒ 检查函数退化成「只算不判」、返回空串 ⇒ 被记为 True。
#
# 实测已复现：`python -O -m kp.selftest` 仍打印「73/73 通过」，但 **71/73 项已不证明任何东西**。
# ⚠️ 这与「LoRA 在 4-bit 路径下静默失效」是**同一类病，只是这次发生在验收工具自身**：
#     任何带 `-O` 的 CI / PYTHONOPTIMIZE=1 / 打包器都会让基线静默失效且不报错。
#
# ⇒ 零成本修法：`__debug__` 在 `-O` 下恒为 `False`，是 python 层面唯一可靠的探针。
if not __debug__:
    raise RuntimeError(
        "selftest 不可在 -O / PYTHONOPTIMIZE 下运行：\n"
        "  本文件的断言机制是 `assert` + 捕获 AssertionError；`-O` 会把 assert 整体移除，\n"
        "  导致所有检查退化为「只算不判」，73/73 变成一个**不再证明任何东西**的数字。\n"
        "  请去掉 -O / PYTHONOPTIMIZE 后重跑。"
    )

import traceback
from typing import Callable, List, Tuple

import torch

from kp.paths import OUT

RESULTS: List[Tuple[str, bool, str]] = []


def _rm(path: str) -> None:
    """删临时产物。

    ⚠️ **不要用 `tempfile.gettempdir()`**：它在受沙箱限制的机器上（实测 viim）
    会静默退化成 cwd，于是临时 `.pt` 掉进**仓库根**、混进 `git status`
    （本机 kokona 上返回正常临时目录，所以这是**潜伏的、机器相关的**坑）。
    ⇒ 临时产物一律落 `KP_OUT` 并清理。
    """
    import os
    try:
        os.remove(path)
    except OSError:
        pass


def check(name: str, fn: Callable[[], str]) -> None:
    try:
        detail = fn() or ""
        RESULTS.append((name, True, detail))
        print(f"  ✅ {name}" + (f"  ·  {detail}" if detail else ""))
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  ❌ {name}  ·  {type(e).__name__}: {e}")
        traceback.print_exc()


def section(title: str) -> None:
    print(f"\n【{title}】")


def main() -> int:
    torch.manual_seed(0)
    print("=" * 68)
    print("KokonaPolaris-S4 骨架自检（CPU）")
    print("=" * 68)

    # ---------------- 1. latent ----------------
    section("1. 混合 latent")
    from kp.latent import (pack_latent, unpack_latent, split_channels,
                           join_channels, swap_channels, channel_mi_penalty,
                           check_shapes)
    from kp.config import LATENT

    def _roundtrip(dtype):
        x = torch.randn(2, LATENT.total_ch, 32, 32).to(dtype)
        blob = pack_latent(x, image_size=1024)
        y, meta = unpack_latent(blob)
        assert y.shape == x.shape, (y.shape, x.shape)
        assert torch.equal(y, x), "解包结果与原始张量不逐位相同"
        assert meta["image_size"] == 1024
        return f"{len(blob)} B, {meta['dtype']}"
    check("pack/unpack 往返（fp32）", lambda: _roundtrip(torch.float32))
    check("pack/unpack 往返（bf16）", lambda: _roundtrip(torch.bfloat16))

    def _channels():
        x = torch.randn(4, LATENT.total_ch, 32, 32)
        s, d = split_channels(x)
        assert s.shape[1] == LATENT.semantic_ch and d.shape[1] == LATENT.detail_ch
        assert torch.equal(join_channels(s, d), x)
        y = swap_channels(x)
        assert torch.equal(y[0, :LATENT.semantic_ch], x[1, :LATENT.semantic_ch])
        assert torch.equal(y[0, LATENT.semantic_ch:], x[0, LATENT.semantic_ch:])
        n = 3
        ind = torch.randn(200, LATENT.semantic_ch, 8, 8)
        det = torch.randn(200, LATENT.detail_ch, 8, 8)
        mi = float(channel_mi_penalty(ind, det, n_samples=200))
        return f"swap ✓ / MI(独立)={mi:.4f}"
    check("通道拆分·交换·互信息惩罚", _channels)

    def _shapes():
        rows = check_shapes()
        assert all(r["ok"] for r in rows), rows
        return " | ".join(f"{r['image']}→{r['tokens']}tok" for r in rows)
    check("token 数自检（256/512/1024）", _shapes)

    # ---------------- 2. 门控 bit-exact ----------------
    section("2. 能力总线：门控全 0 必须 bit-exact")
    from kp.capability import GatedLinear, DeltaPack, ParallelPack, load_adapter

    def _bitexact():
        w = torch.randn(64, 32)
        b = torch.randn(64)
        gl = GatedLinear(w, b)
        x = torch.randn(5, 32)
        before = gl(x)
        pack = DeltaPack("p", 32, 64, rank=4, seed=1)
        gl.add_pack(pack)                       # gate 默认 0
        after = gl(x)
        assert torch.equal(before, after), "挂包后（gate=0）输出不是逐位相同"
        assert pack.delta_weight().abs().max() == 0, "零初始化 B 应使 ΔW 恒为 0"
        return "挂包前后逐位相同，ΔW≡0"
    check("GatedLinear 挂包后 bit-exact", _bitexact)

    def _gate_on():
        w = torch.randn(64, 32)
        gl = GatedLinear(w)
        x = torch.randn(5, 32)
        before = gl(x)
        pack = DeltaPack("p", 32, 64, rank=4, seed=1)
        with torch.no_grad():
            pack.A.normal_()
            pack.B.normal_()
        gl.add_pack(pack)
        gl.set_gate("p", 1.0)
        after = gl(x)
        assert not torch.equal(before, after), "开后门控应当改变输出"
        assert pack.delta_weight().abs().max() > 0
        return f"L1 变化 {float((after-before).abs().mean().detach()):.4f}"
    check("开门控后确实生效", _gate_on)

    # ---------------- 3. 谱检查 ----------------
    section("3. Δ-Pack 谱检查（禁止入侵维度）")

    def _subspace_pass():
        w0 = torch.randn(128, 96)
        p = DeltaPack("d", 96, 128, rank=8, blocks=2, seed=3)
        err = p.init_in_subspace(w0, seed=3)
        rep = p.spectral_report(w0)
        assert rep.passed, f"子空间初始化却未通过：{rep}"
        return (f"cos_min={rep.min_cos_high_rank:.3f}（应≈1）, "
                f"正交能量={rep.energy_out_subspace:.2%}（应≈0）, 分解误差={err:.1e}")
    check("子空间初始化 → 合格", _subspace_pass)

    def _orthogonal_fail():
        m, n = 128, 96
        w0 = torch.randn(m, n)
        U, _, _ = torch.linalg.svd(w0, full_matrices=False)
        k0 = int(0.5 * min(m, n))                   # 与 spectral_check 默认 rank_ratio 对齐
        Utail = U[:, k0:]                           # 正交补：与 top-k0 子空间正交
        C = torch.randn(Utail.shape[1], n)
        dw = Utail @ C                              # 左奇异向量 ⊂ 正交补
        from kp.capability.delta_pack import spectral_check
        rep = spectral_check(dw, w0, target="orthogonal")
        assert not rep.passed, f"正交扰动竟判合格：{rep}"
        return (f"cos_min={rep.min_cos_high_rank:.3f}（应≈0）, "
                f"正交能量={rep.energy_out_subspace:.2%} → 正确判不合格")
    check("正交扰动 → 不合格", _orthogonal_fail)

    # ---------------- 4. 擦除可逆 ----------------
    section("4. 擦除算子 E 与 `E⁻¹∘E` 可逆性")
    from kp.capability import EraseOperator, EraseLedger, erasure_roundtrip

    def _erase():
        w0 = torch.randn(64, 48)
        op = EraseOperator.from_directions("e", "blocks.4.mlp.fc1",
                                           torch.randn(64), torch.randn(48), strength=0.6)
        x = torch.randn(8, 48)

        def fwd(w, xin):
            return torch.nn.functional.linear(xin, w)
        r = erasure_roundtrip(op, fwd, w0, x)
        assert r["w_roundtrip_err"] < 1e-5, r
        assert r["kl"] < 1e-6, r
        led = EraseLedger().add(op, stage="G6-pre", note="示例")
        assert len(led) == 1 and led.total_rank() == 1
        return (f"KL={r['kl']:.1e}, ΔW往返误差={r['w_roundtrip_err']:.1e}, "
                f"强度={op.strength:.3f}, 账本={len(led)}条")
    check("E⁻¹∘E 回到基线", _erase)

    # ---------------- 5. ∥-Pack ----------------
    section("5. ∥-Pack（并联包）")

    def _parallel():
        w = torch.randn(64, 32)
        gl = GatedLinear(w)
        x = torch.randn(5, 32)
        before = gl(x)
        pack = ParallelPack("pp", 32, 64, hidden=128)
        assert pack.delta_weight() is None, "∥-Pack 不应可表示为权重扰动"
        gl.add_pack(pack)
        assert torch.equal(before, gl(x)), "gate=0 时应 bit-exact"
        gl.set_gate("pp", 1.0)
        after = gl(x)
        assert torch.equal(before, after), "上投影零初始化 ⇒ 开启后初始贡献仍为 0"
        return "ΔW=None（物理上不覆盖底模）· 初始贡献 0 · gate=0 bit-exact"
    check("零初始化 + 短路 + 不可表示为 ΔW", _parallel)

    # ---------------- 6. 主干 ----------------
    section("6. 单流 DiT 主干")
    from kp.config import DiTCfg
    from kp.models import SingleStreamDiT, HybridVAE, TextTower, TextTowerCfg
    from kp.sample import sample

    tiny = DiTCfg(dim=64, layers=4, heads=4, mlp_ratio=2.0,
                  double_stream_blocks=1, matryoshka_tokens=(16, 64))
    model = SingleStreamDiT(tiny, latent_ch=40, identity_anchor_layers=[1])
    model.eval()

    def _dit_forward():
        x = torch.randn(1, 40, 8, 8)          # 8×8 = 64 token
        t = torch.full((1,), 500.0)
        ctx = torch.randn(1, 12, 64)
        ident = torch.randn(1, 16, 64)
        dom = torch.randn(1, 16)
        with torch.no_grad():
            v = model(x, t, text_ctx=ctx, identity_ctx=ident, domain=dom)
        assert v.shape == x.shape, (v.shape, x.shape)
        n_gl = len(model.gated_linears())
        return f"v{tuple(v.shape)}, 注入点 {n_gl} 个"
    check("前向形状", _dit_forward)

    def _dit_bitexact():
        x = torch.randn(1, 40, 8, 8)
        t = torch.full((1,), 300.0)
        with torch.no_grad():
            v0 = model(x, t)
        # 给每个注入点挂一个 Δ-Pack；随即打散 A/B（否则 B 的零初始化会让「开启」也无变化）
        n = 0
        for name, gl in model.gated_linears().items():
            pack = DeltaPack(f"d{n}", gl.in_features, gl.out_features, rank=2, seed=n)
            with torch.no_grad():
                pack.A.normal_()
                pack.B.normal_()
            gl.add_pack(pack)                       # gate 默认 0
            n += 1
        with torch.no_grad():
            v1 = model(x, t)
        assert torch.equal(v0, v1), "全部门控=0 时主干输出必须逐位相同"
        bus = model.capability_bus()
        assert bus.inventory(), "总线清单不应为空"
        bus.all_on()
        with torch.no_grad():
            v2 = model(x, t)
        assert not torch.equal(v0, v2), "全部开启后输出应改变"
        bus.all_off()
        with torch.no_grad():
            v3 = model(x, t)
        assert torch.equal(v0, v3), "全部关回后必须恢复 bit-exact"
        return f"{n} 个注入点：全关 bit-exact / 全开改变 / 关回复原"
    check("挂包·全关 bit-exact·全开生效·关回复原", _dit_bitexact)

    def _vae():
        vae = HybridVAE(base=16)
        img = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            z = vae.encode_latent(img)
            rec = vae.decode(z)
        assert z.shape == (1, 40, 8, 8), z.shape        # 256/32 = 8
        assert rec.shape == img.shape, (rec.shape, img.shape)
        return f"256²→latent{tuple(z.shape)}→256²（32×）"
    check("HybridVAE 32× 编解码形状", _vae)

    def _text():
        tt = TextTower(TextTowerCfg(vocab_size=512, dim=32, layers=2, heads=4, out_dim=64))
        ids = torch.randint(0, 512, (2, 10))
        with torch.no_grad():
            ctx = tt(ids)
        assert ctx.shape == (2, 10, 64), ctx.shape
        return f"ids(2,10)→ctx{tuple(ctx.shape)}"
    check("TextTower 接口形状（~220M 骨架）", _text)

    # ---------------- 7. 角色 ----------------
    section("7. 角色卡 + Character Fitter")
    from kp.character import CharacterCard, CharacterFitter, SEMANTIC_LAYERS

    def _fitter():
        f = CharacterFitter(dim=64, n_tokens=16, view_dim=32, heads=4)
        views = torch.randn(1, 2, 3, 64, 64)            # 2 视图（正 + 背）
        with torch.no_grad():
            tok = f(views)
        assert tok.shape == (1, 16, 64), tok.shape
        return f"2 视图 → 身份 token{tuple(tok.shape)}（走 cross-attention，不拼主序列）"
    check("Fitter 输出身份 token", _fitter)

    def _card():
        card = CharacterCard(
            name="Kokona",
            identity_token=torch.randn(16, 64),
            layers={SEMANTIC_LAYERS[1]: (torch.rand(64, 64, 4) * 255).to(torch.uint8).numpy()},
            depth_order=torch.rand(64, 64).numpy(),
            meta={"source": "selftest"},
        )
        warn = card.validate()
        # ⚠️ 不要用 tempfile.gettempdir()：本机实测它返回**工作区根目录**
        #    （受沙箱限制时 Python 会静默退化成 cwd），于是自检会在仓库根丢一个
        #    `_kp_card_selftest.pt`，混进 `git status`。临时产物一律落 KP_OUT 并清理。
        import os
        from kp.paths import OUT
        p = str(OUT / "_kp_card_selftest.pt")
        card.save(p)
        try:
            back = CharacterCard.load(p)
            assert back.name == card.name and back.num_tokens == card.num_tokens
            assert torch.equal(back.identity_token, card.identity_token)
        finally:
            try:
                os.remove(p)
            except OSError:
                pass
        return f"往返一致，{card.size_bytes()/1024:.0f}KB，警告 {len(warn)} 条"
    check("角色卡 保存/加载 往返", _card)

    # ---------------- 8. 采样 ----------------
    section("8. Rectified Flow + Matryoshka 采样")

    def _sample():
        model.eval()
        z = sample(model, (1, 40, 8, 8), steps=4, seed=0, matryoshka=(16, 64))
        assert z.shape == (1, 40, 8, 8), z.shape
        # 同 seed 可复现（G1 永久规范：判据是轨迹复现，不是像素对齐）
        z2 = sample(model, (1, 40, 8, 8), steps=4, seed=0, matryoshka=(16, 64))
        assert torch.equal(z, z2), "同 seed 采样不可复现"
        return f"4 步 Matryoshka 16→64 token，{tuple(z.shape)}，同 seed 可复现"
    check("采样可跑且可复现", _sample)

    # ---------------- 9. NVFP4 ----------------
    section("9. NVFP4 模拟量化（W4A8 · STE）")
    from kp.quant import (QuantSpec, quant_fp4, quant_fp8, quant_fp4_ste,
                          relative_error)

    def _ref_match():
        def ref_fp4(x, blk=16):
            shape = x.shape
            inn = shape[-1]
            if inn % blk:
                blk = inn
            xb = x.reshape(-1, inn // blk, blk)
            amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = (amax / 6.0).to(torch.float8_e4m3fn).to(x.dtype).clamp(min=1e-12)
            q = (xb / scale).round().clamp(-6.0, 6.0)
            return (q * scale).reshape(shape)
        x = torch.randn(64, 128)
        assert torch.equal(quant_fp4(x, 16, ste=False), ref_fp4(x, 16)), "与参考实现不一致"
        return "与 tools/e5b_qad.py 的 quant_fp4 逐位一致"
    check("与已有参考实现对拍", _ref_match)

    def _ste_grad():
        x = torch.randn(8, 64, requires_grad=True)
        quant_fp4_ste(x, 16).sum().backward()
        assert x.grad is not None and torch.equal(x.grad, torch.ones_like(x.grad)), x.grad
        return "STE 直通：梯度恒 1（可反传）"
    check("STE 可微", _ste_grad)

    def _block_err():
        w = torch.randn(256, 256)
        e16 = relative_error(quant_fp4(w, 16, ste=False), w)
        e32 = relative_error(quant_fp4(w, 32, ste=False), w)
        e8 = relative_error(quant_fp8(w, ste=False), w)
        assert e16 < e32, f"block16 应优于 block32：{e16} vs {e32}"
        return f"block16={e16:.2%} < block32={e32:.2%}（小 block 更优）· fp8={e8:.2%}"
    check("block 大小与误差（16 优于 32，复核 E5）", _block_err)

    def _gl_quant():
        w = torch.randn(64, 32)
        gl = GatedLinear(w)
        x = torch.randn(5, 32)
        base = gl(x)
        gl.set_quant(QuantSpec(weight="fp4", act="fp8"))
        q = gl(x)
        assert torch.isfinite(q).all() and not torch.equal(base, q), "量化应改变输出"
        gl.set_quant(None)
        assert torch.equal(gl(x), base), "关闭量化后必须恢复 bit-exact"
        return "量化改变输出；关闭后 bit-exact"
    check("GatedLinear 量化开关（关后 bit-exact）", _gl_quant)

    def _quant_cfg_is_live():
        """🔴 「`QUANT` 不是死旋钮」的守卫（2026-10-03 接线修复）。

        审查发现：`QuantCfg` 全 8 字段曾**只被 `arch_report` 打印**，改 config 零效果。
        修法：`nvfp4.spec_from_config()` + `DESIGN_SPEC` 由 config 实际构造 +
             `qad.SKIP_DEFAULT` 由 `quantize_proj_out` 推导。
        ⭐ 这条断言的作用是**持续证明"接线还活着"** ——
           任何人把 `DESIGN_SPEC` 改回写死字面量，这里立刻红。
        """
        import dataclasses
        from kp.config import QUANT
        from kp.quant.nvfp4 import DESIGN_SPEC, spec_from_config
        from kp.train.qad import SKIP_DEFAULT
        # ① 恒等：DESIGN_SPEC 必须等于「按 config 现算出来的那份」
        want = spec_from_config(QUANT)
        assert (DESIGN_SPEC.weight, DESIGN_SPEC.act, DESIGN_SPEC.block) == \
               (want.weight, want.act, want.block), (
            f"DESIGN_SPEC {DESIGN_SPEC.describe()} 与 config 算出的 "
            f"{want.describe()} 不一致 ⇒ 有人把它改回写死值了")
        # ② 变异性：改 config 的每个**可接线**字段，spec 必须跟着变（否则仍是死的）
        #    ⚠️ `act` 的映射是 位宽→档名（8→"fp8"、4→"fp4"），不是直接透传
        for field, val, attr, want in (("block_size", 8, "block", 8),
                                        ("act_bits", 4, "act", "fp4")):
            q2 = dataclasses.replace(QUANT, **{field: val})
            assert getattr(spec_from_config(q2), attr) == want, \
                (f"改 QUANT.{field}={val} 后 spec.{attr} 应为 {want!r}，"
                 f"实际 {getattr(spec_from_config(q2), attr)!r} ⇒ 该字段仍是死的")
        # ③ SKIP_DEFAULT 由 quantize_proj_out 推导
        want_skip = () if QUANT.quantize_proj_out else ("out_proj",)
        assert SKIP_DEFAULT == want_skip, \
            f"SKIP_DEFAULT {SKIP_DEFAULT} 与 quantize_proj_out={QUANT.quantize_proj_out} 不一致"
        return (f"DESIGN_SPEC ≡ spec_from_config(QUANT)（block={DESIGN_SPEC.block}）；"
                f"改 block_size/act_bits **确实会变**；SKIP_DEFAULT={SKIP_DEFAULT}")
    check("QUANT 配置真接线（改 config 行为跟着变）", _quant_cfg_is_live)

    def _quant_cfg_rejects_invalid():
        """⛔ 不可接线的字段必须**报错**而不是静默忽略。

        `weight_bits=8` / `scale_fmt=FP8` 这类改动会让格式**不再是 NVFP4**；
        若静默忽略，就回到了"死旋钮"的病根。⇒ 必须显式抛错。
        """
        import dataclasses
        from kp.config import QUANT
        from kp.quant.nvfp4 import spec_from_config
        for field, val, why in (("weight_bits", 8, "非 4-bit/E2M1"),
                                ("scale_fmt", "FP8", "非 E4M3")):
            try:
                spec_from_config(dataclasses.replace(QUANT, **{field: val}))
            except ValueError:
                pass
            else:
                raise AssertionError(f"QUANT.{field}={val}（{why}）被静默接受 ⇒ 应显式报错")
        return "weight_bits=8 / scale_fmt=FP8 均**显式报错**（不静默忽略）"
    check("不可接线字段显式报错（防回到死旋钮病根）", _quant_cfg_rejects_invalid)

    # ---------------- 10. 能力包落盘 / 加载 ----------------
    section("10. 能力包落盘 / 加载往返（四接口之一）")
    from kp.capability import save_adapter, load_adapter
    import os as _os

    def _adapter_roundtrip():
        w = torch.randn(48, 32)
        gl = GatedLinear(w)
        x = torch.randn(4, 32)
        base = gl(x)
        d = DeltaPack("d1", 32, 48, rank=4, seed=5)
        with torch.no_grad():
            d.A.normal_()
            d.B.normal_()
        gl.add_pack(d)
        pp = ParallelPack("p1", 32, 48, hidden=64)
        with torch.no_grad():
            pp.down.weight.normal_()
            pp.up.weight.normal_()
            pp.up.bias.normal_()
        gl.add_pack(pp)
        gl.set_gate("d1", 0.7)
        gl.set_gate("p1", 0.3)
        y = gl(x)
        path = str(OUT / "_kp_adapter_selftest.pt")
        n = save_adapter(gl, path)
        try:
            gl2 = GatedLinear(w)
            mounted = load_adapter(gl2, path)
        finally:
            _rm(path)
        gl2.set_gate("d1", 0.7)
        gl2.set_gate("p1", 0.3)
        assert torch.allclose(gl2(x), y, atol=1e-6), float((gl2(x) - y).abs().max())
        gl2.set_gate("d1", 0.0)
        gl2.set_gate("p1", 0.0)
        assert torch.equal(gl2(x), base), "全关后应恢复 bit-exact"
        return f"导出 {n} 个包 / 挂载 {len(mounted)} 个；数值一致，全关后 bit-exact"
    check("save_adapter → load_adapter 往返", _adapter_roundtrip)

    def _adapter_dit():
        torch.manual_seed(7)
        m1 = SingleStreamDiT(tiny, latent_ch=40, identity_anchor_layers=[1])
        torch.manual_seed(7)
        m2 = SingleStreamDiT(tiny, latent_ch=40, identity_anchor_layers=[1])
        for i, (name, glmod) in enumerate(m1.gated_linears().items()):
            d = DeltaPack(f"d{i}", glmod.in_features, glmod.out_features, rank=2, seed=i)
            with torch.no_grad():
                d.A.normal_()
                d.B.normal_()
            glmod.add_pack(d)
            glmod.set_gate(d.name, 1.0)
        path = str(OUT / "_kp_adapter_dit.pt")
        n = save_adapter(m1, path)
        try:
            mounted = load_adapter(m2, path)
        finally:
            _rm(path)
        x = torch.randn(1, 40, 8, 8)
        t = torch.full((1,), 300.0)
        with torch.no_grad():
            assert torch.allclose(m1(x, t), m2(x, t), atol=1e-6), "整模型往返不一致"
        return f"{n} 个包跨模型往返一致（挂载 {len(mounted)}）"
    check("整模型级 落盘/加载 往返", _adapter_dit)

    # ---------------- 11. QAD ----------------
    section("11. QAD：冻结主干 + 只训 Δ-Pack")
    from kp.train import set_quant as qad_set_quant, run_qad, grad_health, budget_projection

    def _qad():
        torch.manual_seed(11)
        mm = SingleStreamDiT(tiny, latent_ch=40, identity_anchor_layers=[1])
        nq = qad_set_quant(mm)
        x = torch.randn(1, 40, 8, 8)
        t = torch.full((1,), 400.0)
        res = run_qad(mm, x, t, steps=20, seed=11)
        assert res.loss_drop > 0.5, (f"loss 未显著下降："
                                    f"{res.history[0]['loss']} → {res.history[-1]['loss']}")
        gh = grad_health(mm)
        assert gh["base_max_grad"] == 0.0, "冻结主干不应有梯度"
        assert gh["pack_max_grad"] > 0, "包参数应有梯度"
        b = res.budget
        return (f"{nq} 注入点量化；loss {res.history[0]['loss']:.5f}→"
                f"{res.history[-1]['loss']:.5f}（降 {res.loss_drop:.0%}）"
                f"｜主干梯度 0，包梯度 {gh['pack_max_grad']:.2e}")
    check("QAD loss 下降 + 梯度只进包", _qad)

    def _qad_budget():
        proj = budget_projection(rank=4)
        s, m = proj["KP-S"], proj["KP-M"]
        assert s["trainable_ratio"] < 0.01, s
        assert m["trainable_ratio"] < 0.01, m
        return (f"真实规模投影：KP-S 可训 {s['trainable']/1e6:.2f}M"
                f"（{s['trainable_ratio']:.2%}，AdamW ≈{s['adamw_state_gb']:.2f}GB）｜"
                f"KP-M {m['trainable']/1e6:.2f}M（{m['trainable_ratio']:.2%}）"
                f" —— 对照 E5b 实测 5.99M/0.37%")
    check("adapter 预算投影（真实 KP-S/KP-M）", _qad_budget)

    # ---------------- 12. SVDPack ----------------
    section("12. SVDPack（子空间约束 / 跨版本可迁移）")
    from kp.capability import SVDPack

    def _svd_pack():
        w0 = torch.randn(96, 64)
        p = SVDPack.init_from("s1", w0, rank=8)
        assert float(p.delta_weight().abs().max()) == 0.0, "σ=0 时 ΔW 必须恒为 0"
        rep = p.spectral_report(w0)
        assert rep.passed and rep.min_cos_high_rank > 0.99, rep
        with torch.no_grad():
            p.sigma.normal_()
        rep2 = p.spectral_report(w0)
        assert rep2.passed and rep2.min_cos_high_rank > 0.99, rep2
        return (f"可训参数仅 {p.trainable_params}（存储 {p.stored_params}）｜"
                f"cos_min={rep2.min_cos_high_rank:.4f}（按构造≈1）")
    check("按构造通过谱检查（不可能产生入侵维度）", _svd_pack)

    def _svd_portable():
        w0 = torch.randn(96, 64)
        p = SVDPack.init_from("s2", w0, rank=8)
        with torch.no_grad():
            p.sigma.normal_()
        dw = p.delta_weight().clone()
        spec = p.to_spec("blocks.0.attn.qkv")
        w0_new = torch.randn(96, 64)                 # 模拟「主干换了一个版本」
        q = SVDPack.from_spec(w0_new, spec)
        assert torch.allclose(q.delta_weight(), dw, atol=1e-6), "跨版本 ΔW 应逐值不变"
        return "换 W0 后 ΔW 逐值不变 ⇒ 跨主干版本直接可用"
    check("跨版本可迁移（基被存储，不从 W0 重算）", _svd_portable)

    def _svd_in_layer():
        w = torch.randn(48, 32)
        gl = GatedLinear(w)
        x = torch.randn(4, 32)
        base = gl(x)
        p = SVDPack.init_from("s3", w, rank=6)      # gate 默认 0（CAP.gate_init）
        gl.add_pack(p)
        assert torch.equal(gl(x), base), "σ=0 时挂包应 bit-exact"
        with torch.no_grad():
            p.sigma.normal_()
        # ★ 两条不变量要**分开**验：门控=0 时 σ≠0 也必须 bit-exact（整条旁路短路）
        assert torch.equal(gl(x), base), "门控=0 时即使 σ≠0 也必须 bit-exact"
        gl.set_gate("s3", 1.0)
        assert not torch.equal(gl(x), base), "开门控后应生效"
        gl.set_gate("s3", 0.0)
        assert torch.equal(gl(x), base), "关断后应 bit-exact"
        return "挂包(σ=0) bit-exact → 门控=0·σ≠0 仍 bit-exact → 开门生效 → 关断复原"
    check("接入 GatedLinear（可关断）", _svd_in_layer)

    # ---------------- 13. CharaBridge ----------------
    section("13. CharaBridge（身份 + 几何双分支）")
    from kp.models import CharaBridge

    def _cb_off():
        cb = CharaBridge(dim=64, n_tokens=16, view_dim=32, heads=4)
        refs = torch.randn(1, 2, 3, 64, 64)
        assert cb(refs) is None, "门控=0 时应返回 None（整条分支跳过）"
        cb.set_gate(1.0)
        tok = cb(refs)
        assert tok is not None and tok.shape == (1, 16, 64), getattr(tok, "shape", None)
        return "gate=0 → None；gate=1 → (1,16,64)"
    check("关断返回 None（而非零向量）", _cb_off)

    def _cb_geo():
        cb = CharaBridge(dim=64, n_tokens=16, view_dim=32, heads=4, gate=1.0)
        refs = torch.randn(1, 2, 3, 64, 64)
        geo = torch.randn(1, 2, 3, 64, 64)
        a, b = cb(refs, geo=None), cb(refs, geo=geo)
        assert a.shape == b.shape == (1, 16, 64)
        assert not torch.allclose(a, b), "几何分支应当改变输出"
        return "几何分支（normal/depth/ray-pose）确实参与融合"
    check("几何分支生效", _cb_geo)

    def _cb_dit():
        torch.manual_seed(3)
        m = SingleStreamDiT(tiny, latent_ch=40, identity_anchor_layers=[1])
        x = torch.randn(1, 40, 8, 8)
        t = torch.full((1,), 300.0)
        with torch.no_grad():
            v_off = m(x, t, identity_ctx=None)
        cb = CharaBridge(dim=64, n_tokens=16, view_dim=32, heads=4, gate=0.0)
        assert cb(torch.randn(1, 2, 3, 64, 64)) is None
        with torch.no_grad():
            assert torch.equal(v_off, m(x, t, identity_ctx=None))
        tok = CharaBridge(dim=64, n_tokens=16, view_dim=32, heads=4,
                          gate=1.0)(torch.randn(1, 2, 3, 64, 64))
        # ① adaLN-Zero 保护：身份门控初始为 0 ⇒ 此时注入身份 token 也不改变输出
        with torch.no_grad():
            assert torch.equal(v_off, m(x, t, identity_ctx=tok)), \
                "adaLN-Zero 下身份门控为 0，注入 token 不应改变输出"
        # ② 显式打开身份门（adaLN 输出的第 8 段 g_id）后，注入才生效
        d = m.cfg.dim
        with torch.no_grad():
            for blk in m.blocks:
                if blk.identity_cross is not None:
                    # ⚠️ 切片必须非空：adaLN bias 长度 = 8d，若段数与本行不一致会得到
                    #    **空切片**，`.fill_()` 静默 no-op **不抛异常** ⇒ 门根本没开，
                    #    却要晚一步炸在下面那句断言上，报一句完全指不到根因的错。
                    #    这里显式拦住（通用教训 #2：报错文本 ≠ 根因）。
                    _sl = blk.adaLN[-1].bias[7 * d:8 * d]
                    assert _sl.numel() == d, \
                        f"adaLN 段数与 selftest 不符：得到空切片 {tuple(_sl.shape)}"
                    _sl.fill_(0.5)
            v_off2 = m(x, t, identity_ctx=None)
            v_on = m(x, t, identity_ctx=tok)
        assert torch.equal(v_off2, v_off), "仅开门、不给身份 token ⇒ 仍逐位不变"
        assert not torch.equal(v_on, v_off2), "开门后注入身份 token 应改变输出"
        return "关断时主干逐位不变；adaLN-Zero 下注入无效；开门后注入生效"
    check("与主干联动：关断 bit-exact", _cb_dit)

    # ---------------- 14. Layout Planner ----------------
    section("14. Layout Planner（确定性排版 · T0）")
    from kp.typography import TextBlock, LayoutSpec, plan, roi_boxes, NO_LINE_START, ROIBranch

    def _layout_basic():
        L = plan(LayoutSpec(canvas=(1024, 1024), margin=64,
                            blocks=[TextBlock("心夏北极星", size=96, align="center")]))
        assert L["valid"], (L["boxes"], L["warnings"])
        b = L["boxes"][0]
        assert len(b["lines"]) == 1, b["lines"]
        assert len(roi_boxes(L)) == 1
        return f"5 字一行放下：框 {b['w']:.0f}×{b['h']:.0f}，valid={L['valid']}"
    check("中文短句一行放下", _layout_basic)

    def _layout_wrap():
        txt = "心夏北极星，" * 12
        L = plan(LayoutSpec(canvas=(1024, 1024), margin=64,
                            blocks=[TextBlock(txt, size=64)]))
        b = L["boxes"][0]
        assert len(b["lines"]) > 1, b["lines"]
        assert all(not ln or ln[0] not in NO_LINE_START for ln in b["lines"]), b["lines"]
        return f"{len(txt)} 字 → {len(b['lines'])} 行（避头尾生效）"
    check("长文本折行 + 避头尾", _layout_wrap)

    def _layout_multi():
        L = plan(LayoutSpec(canvas=(1024, 1024), margin=64, blocks=[
            TextBlock("第一章", size=72, align="center"),
            TextBlock("心夏北极星是一个自研文生图架构。" * 2, size=40),
            TextBlock("竖排测试", size=64, direction="v", align="right"),
        ]))
        assert len(L["boxes"]) == 3
        assert L["boxes"][2]["direction"] == "v"
        assert L["valid"], (L["boxes"], L["warnings"])
        return f"3 块（含竖排）全部合法，valid={L['valid']}"
    check("多块 + 竖排 + 合法性校验", _layout_multi)

    def _roi_branch():
        br = ROIBranch(compression=4, dim=64, base=16, tokens_per_roi=16)
        img = torch.randn(1, 3, 256, 256)
        rois = [{"x0": 0, "y0": 0, "x1": 64, "y1": 64}]
        assert br(img, rois) is None, "关断时应返回 None"
        br.set_gate(1.0)
        out = br(img, rois)
        assert out.shape == (1, 16, 64), out.shape
        return f"4× 压缩 ROI → {tuple(out.shape)}；关断时 None"
    check("ROIBranch 低压缩分支", _roi_branch)

    # ---------------- 15. caption 语料审计（P1.8） ----------------
    section("15. caption 语料审计（P1.8 中文自然语言支持）")
    from kp.data import (audit_captions, detect_language, tag_soup_score,
                         suggest_language_plan)

    def _lang_detect():
        cases = {
            "一位少女站在海边。": "zh",
            "海辺に立つ少女。": "ja",                       # 有假名 ⇒ 日文，不能算中文
            "A girl stands on the beach.": "en",
            "少女穿着白色 dress 站在海边。": "zh",           # 夹 1 个英文词 ⇒ 仍是中文
            "少女 wearing a white dress 在海边。": "mixed",   # 英文词多 ⇒ 混排
            "12345 !!!": "other",
        }
        bad = {t: (l, detect_language(t)) for t, l in cases.items()
               if detect_language(t) != l}
        assert not bad, f"语言判定错误：{bad}"
        return f"{len(cases)} 例全对（含假名→ja、夹词→zh、多词→mixed）"
    check("语言判定（zh/ja/en/mixed/other）", _lang_detect)

    def _tag_soup():
        good = [
            "一位长发少女站在夏日的海边，微笑着看向镜头。",
            "少女回眸，海风吹乱了她的发丝。",
            "A girl with long silver hair stands on a rooftop at dusk.",
            "夕阳把整片天空染成橘红色，少女的影子被拉得很长。",
        ]
        bad = [
            "1girl, solo, long hair, blue eyes, smile, white shirt, outdoors",
            "masterpiece, best quality, ultra detailed, 8k, anime style",
            "长发, 蓝眼睛, 微笑, 白衬衫, 户外, 白天",
        ]
        fp = [(t, tag_soup_score(t)) for t in good if tag_soup_score(t) >= 0.6]
        fn = [(t, tag_soup_score(t)) for t in bad if tag_soup_score(t) < 0.6]
        assert not fp, f"自然语言被误判为标签串：{fp}"
        assert not fn, f"标签串漏检：{fn}"
        return (f"自然语言 {len(good)}/{len(good)} 不误报，"
                f"标签串 {len(bad)}/{len(bad)} 全命中")
    check("标签串（tag soup）检测", _tag_soup)

    def _audit_mix():
        zh = ["一位少女站在夏日的海边，微笑着看向镜头。" * 1] * 8
        en = ["A girl stands on a beach at summer."] * 2
        ok = audit_captions(zh + en)
        assert ok.passed, (ok.lang_share, ok.issues)
        # 缺英文对齐 → 必须给出 issue 且判失败
        no_en = audit_captions(zh)
        assert not no_en.passed and no_en.issues, no_en.lang_share
        # 标签串 → 必须判违规
        soup = audit_captions(["1girl, solo, long hair, blue eyes, smile, white shirt"])
        assert not soup.passed and any(k == "标签串" for _, k, _ in soup.violations)
        return (f"合规语料通过（zh {ok.lang_share['zh']:.0%}/en {ok.lang_share['en']:.0%}）；"
                f"缺英文/含标签串均被判失败")
    check("语言配比达标判定 + 违规定位", _audit_mix)

    def _coverage():
        caps = ["心夏北极星站在海边。", "北极星的光芒落在海面。",
                "心夏望向远方。"]
        a = audit_captions(caps, zh_band=(0.0, 1.0), en_band=(0.0, 0.0))
        # 汉字集合 = 三句汉字去重后的并集（独立写出期望值，不用同一套区间判定）
        exp = {"心", "夏", "北", "极", "星", "站", "在", "海", "边",
               "的", "光", "芒", "落", "面", "望", "向", "远", "方"}
        assert set(a.char_set) == exp, sorted(a.char_set)
        assert len(a.char_set) == 18, len(a.char_set)
        plan = suggest_language_plan(1000)
        assert plan["total"] == 1000 and plan["zh"] > plan["en"], plan
        return (f"汉字覆盖 {len(a.char_set)} 个（去重并集）；"
                f"1000 条配额建议 zh={plan['zh']}/en={plan['en']}/ja={plan['ja']}")
    check("汉字覆盖统计 + 配额反推", _coverage)

    # ---------------- 16. Axis Probe（G3.5 四测） ----------------
    section("16. Axis Probe（G3.5 · L1 条件轴真实性四测）")
    from kp.probe import AxisProbe, axis_report_text
    from kp.probe.synthetic import Synthetic

    def _axis_probe():
        S = Synthetic(n_axes=6, dim=256)
        fns = S.all()
        exp = S.expected_failures()
        got = {}
        for name, f in fns.items():
            rep = AxisProbe(f, n_axes=6, seed=0).run()
            got[name] = rep.results[0].failures
        # ① 干净正交轴必须四测全过
        assert got["clean"] == [], f"clean 不该有失败项：{got['clean']}"
        # ②③④ 各样本恰好触发其对应的那一测
        for name in ("coupled", "suppressed", "nonlinear"):
            assert set(exp[name]) <= set(got[name]), \
                f"{name} 应触发 {exp[name]}，实测 {got[name]}"
        return (f"4 类样本各司其职：clean 全过；"
                f"coupled→{got['coupled']}；suppressed→{got['suppressed']}；"
                f"nonlinear→{got['nonlinear']}")
    check("四测能分别抓出串扰 / 非单调 / 低比特抹平", _axis_probe)

    def _axis_lowbit_is_essential():
        """⭐ 关键：`suppressed` 必须**只**在低比特行程一测上失败。

        如果它同时也挂了单调/正交/可逆，那说明这一测是可被替代的 ——
        而设计稿的论点是「**低比特行程在 bf16 上调参时完全看不出来**，
        只有把量化器接进回路才暴露」。这条断言就是在守这个论点。
        """
        S = Synthetic(n_axes=6, dim=256)
        r = AxisProbe(S.suppressed(), n_axes=6, seed=0).run().results[0]
        assert r.failures == ["低比特行程"], f"实际失败项 {r.failures}"
        assert r.mono >= 0.9 and r.ortho <= 0.3 and r.rev >= 0.9, r.as_dict()
        assert r.travel_range_keep < 0.5, r.travel_range_keep
        return (f"被抑制的轴：单调 {r.mono:.2f} / 串扰 {r.ortho:.2f} / 可逆 {r.rev:.2f} "
                f"全过，**只有**行程 {r.travel_keep:.2f} 挂 —— 量化回路之外看不见")
    check("低比特行程不可被前三测替代（G3.5 的要害）", _axis_lowbit_is_essential)

    def _axis_verdict():
        """未过门的轴必须被明确判为「归 L3」。"""
        S = Synthetic(n_axes=5, dim=256)          # n_axes 必须与探针一致
        rep = AxisProbe(S.suppressed(), n_axes=5, seed=0).run()
        assert rep.n_pass == 0 and len(rep.l3_axes) == 5, rep.as_dict()
        assert all("L3" in r.verdict for r in rep.results)
        txt = axis_report_text(rep)
        assert "未过门的轴默认归" in txt
        rep2 = AxisProbe(S.clean(), n_axes=5, seed=0).run()
        assert rep2.n_pass == 5 and rep2.l3_axes == []
        return "不过 → 0/5 归 L3；通过 → 5/5 留在 L1（报告含归 L3 提示）"
    check("判定律：不过的轴默认归 L3", _axis_verdict)

    # ---------------- 17. 多视角配对（角色卡数据线） ----------------
    section("17. 多视角配对（声明式 · 单旋钮变化）")
    from kp.character.pairing import (Record, PairSpec, build_pairs,
                                      format_pair_report)

    def _pair_single_knob():
        """标准情形：同角色、同姿势，只有 view 不同 ⇒ 恰好配出单旋钮对。"""
        rs = [
            Record("f.png", "kokona", {"view": "front", "pose": "stand"}),
            Record("b.png", "kokona", {"view": "back", "pose": "stand"}),
            Record("l.png", "kokona", {"view": "left", "pose": "stand"}),
        ]
        rep = build_pairs(rs, PairSpec(vary=("view",), match=("pose",)))
        # 3 视图两两配对 = C(3,2) = 3 对
        assert rep.n_pairs == 3, rep.n_pairs
        assert len(rep.paired_keys) == 3
        assert not rep.undeclared and not rep.gaps
        return f"3 视图 → {rep.n_pairs} 对（组合式扩增）"
    check("同角色 + 只动一个旋钮 → 配对", _pair_single_knob)

    def _pair_respects_match():
        """`match` 轴不同 ⇒ **不许**配对（否则学到的不是"只动一个旋钮"）。"""
        rs = [
            Record("a.png", "kokona", {"view": "front", "pose": "stand"}),
            Record("b.png", "kokona", {"view": "back", "pose": "sit"}),   # 姿势也变了
        ]
        rep = build_pairs(rs, PairSpec(vary=("view",), match=("pose",)))
        assert rep.n_pairs == 0, rep.n_pairs
        assert set(rep.isolated) == {"a.png", "b.png"}, rep.isolated
        return "姿势同时变了 → 0 对，两条都记为孤立（不污染训练目标）"
    check("match 轴不同则拒绝配对", _pair_respects_match)

    def _pair_never_guesses():
        """⭐ 没有声明就**明说没有**，绝不用文件名去猜（这是本模块存在的理由）。"""
        rs = [
            Record("front.png", "kokona", {}),      # 名字里带 front，但**没声明**
            Record("back.png", "kokona", {}),
        ]
        rep = build_pairs(rs, PairSpec(vary=("view",), match=()))
        assert rep.n_pairs == 0, "不声明就配对 = 猜，必须拒绝"
        assert len(rep.undeclared) == 2, rep.undeclared
        assert any("未声明" in g for g in rep.gaps), rep.gaps
        return f"两者都缺 view 列 → 0 对 + {len(rep.undeclared)} 条 undeclared + 缺口说明"
    check("未声明 view → 拒绝配对并报缺口（不猜）", _pair_never_guesses)

    def _pair_gap_report():
        """给定期望取值时应报出「还缺哪些视角」—— 这是给用户看的「该补拍什么」。"""
        rs = [Record("f.png", "kokona", {"view": "front"})]
        rep = build_pairs(rs, PairSpec(vary=("view",), match=()),
                          target_values={"view": ("front", "back", "left", "right")})
        txt = format_pair_report(rep)
        assert any("back" in g and "left" in g for g in rep.gaps), rep.gaps
        assert "该补拍什么" in txt
        return "缺 back/left/right 被明确列出（用户据此补拍）"
    check("缺口清单（该补拍什么）", _pair_gap_report)

    # ---------------- 18. Character Fitter 训练（配对驱动） ----------------
    section("18. Character Fitter 训练（配对驱动 · 留出视角 + 负对照）")
    from kp.character.dataset import PairViewLoader
    from kp.train.fitter import (train_fitter, run_closed_loop, fitter_loss,
                                 shuffle_positives)

    def _loader_synthetic():
        """合成加载器：n 身份 × C(v,2) 对，张量形状/轴标注都对得上。"""
        ld = PairViewLoader.synthetic(n_identities=4, n_views=3, size=24, seed=0)
        assert ld.n_pairs == 12, ld.n_pairs                 # 4 × C(3,2)
        assert len(ld.identities()) == 4
        assert all(tuple(p.anchor.shape) == (3, 24, 24) for p in ld.pairs)
        assert ld.pairs[0].varied_axis == "view"
        assert len(ld.group_by_identity()) == 4
        return f"4 身份 × 3 视角 → {ld.n_pairs} 对（组合式扩增）"
    check("合成配对加载器（身份 × 视角）", _loader_synthetic)

    def _loader_split_leakage():
        """⭐ 留出视角**绝不能**出现在训练对里 —— 否则"泛化"是假的。"""
        ld, holdout, refs = PairViewLoader.synthetic_split(
            n_identities=3, n_train_views=3, n_holdout=1, size=24, seed=0)
        train_keys = {p.anchor_key for p in ld.pairs} | {p.positive_key for p in ld.pairs}
        ho_keys = {f"{ident}_v{v}" for ident in holdout
                   for v in range(3, 4)}
        assert not (train_keys & ho_keys), train_keys & ho_keys
        assert set(holdout) == set(refs) == set(ld.identities())
        assert all(len(v) == 1 for v in holdout.values()), holdout
        return (f"训练 {len(train_keys)} 视图 vs 留出 {len(ho_keys)} 视图，**无交集**")
    check("留出视角不泄漏进训练集", _loader_split_leakage)

    def _loss_semantics():
        """单身份 ⇒ **无**负样本：has_negatives=False、gap 记 0（未知 ≠ 好）。"""
        torch.manual_seed(0)
        ta = torch.randn(2, 4, 8)
        tb = ta + 0.01 * torch.randn(2, 4, 8)
        _, inv, gap, has_neg = fitter_loss(ta, tb, ["a", "a"])
        assert has_neg is False, "单身份不该有负样本"
        assert float(gap) == 0.0, gap
        assert float(inv) < 0.05, inv
        _, _, gap2, has_neg2 = fitter_loss(ta, -tb, ["a", "b"])
        assert has_neg2 is True, "双身份必须启用负样本"
        # ⚠️ gap 可以为负（正对相似度低于负对）—— 它只是**诊断量**，不是"必须为正"
        assert float(gap2) < 0.0, "这里正对是反相的 ⇒ gap 应为负，正好证明它不作弊"
        return (f"单身份 has_neg=False/gap=0；双身份 has_neg=True/gap={float(gap2):+.3f}"
                f"（可为负 ⇒ 是诊断量而非损失项）")
    check("损失语义：无负样本必须显式标记", _loss_semantics)

    def _shuffle_control_breaks_pairs():
        ld = PairViewLoader.synthetic(n_identities=4, n_views=3, size=24, seed=0)
        bad = shuffle_positives(ld, seed=0)
        assert bad.n_pairs == ld.n_pairs
        same = sum(1 for a, b in zip(ld.pairs, bad.pairs)
                   if a.identity == b.identity and a.positive_key == b.positive_key)
        assert same < bad.n_pairs, "打乱后不应与原始配对完全相同"
        return f"{bad.n_pairs} 对中 {same} 对未变（其余 positive 已换成别身份）"
    check("负对照：打乱配对确实破坏了配对", _shuffle_control_breaks_pairs)

    def _closed_loop():
        """⭐ 四判据：不变性 / 分离性 / **留出泛化** / **负对照**。"""
        r = run_closed_loop(n_identities=6, n_train_views=3, n_holdout=1,
                            size=32, steps=100, seed=0)
        assert r.ok_invariance, f"不变性未过：inv_end={r.fit.inv_end:.4f}"
        assert r.ok_separation, f"分离未过：gap_end={r.fit.gap_end:.4f}"
        assert r.ok_generalization, f"留出泛化未过：ho={r.holdout_inv:.4f}"
        assert r.ok_control, f"负对照未过：ctrl_ho={r.control_holdout_inv:.4f}"
        return (f"不变性 {r.fit.inv_start:.3f}→{r.fit.inv_end:.3f}｜"
                f"间隔 {r.fit.gap_start:+.3f}→{r.fit.gap_end:+.3f}｜"
                f"留出 {r.holdout_inv:.4f} vs 负对照 {r.control_holdout_inv:.4f}")
    check("闭式自检四判据（留出泛化 + 负对照）", _closed_loop)

    def _real_batch_smoke():
        """真实批次 kokona：能加载成对、能训（无负样本必须明说，不算通过）。"""
        try:
            ld = PairViewLoader.from_batch("kokona", size=32)
        except FileNotFoundError:
            return "跳过（无 data/characters/kokona）—— 非失败"
        if ld.n_pairs == 0:
            return "批次存在但 0 对（需 ≥2 视图 + 声明的 view）—— 非失败"
        res = train_fitter(ld, steps=10, seed=0, batch_size=min(4, ld.n_pairs))
        assert res.inv_end <= res.inv_start + 1e-6
        note = "" if res.has_negatives else "｜**无负样本**（覆盖缺口，非通过）"
        return (f"kokona {ld.n_pairs} 对 / {res.n_identities} 身份，训 10 步 "
                f"inv {res.inv_end:.4f}{note}")
    check("真实批次冒烟（kokona）", _real_batch_smoke)

    # ---------------- 19. G2 通道分离（结构性门） ----------------
    section("19. G2 通道分离（交叉扰动 + 尺子双向校验）")
    from kp.latent.separation import (
        synthetic_batch, train_separation, evaluate, _Oracle, _Leaky,
        cross_perturb, shuffle_within_batch,
    )
    from kp.latent.hybrid import channel_mi_penalty as _mi
    from kp.latent.hybrid import split_channels as _split

    def _g2_ruler():
        """⭐ 尺子必须先自证：oracle 必须 PASS、leaky 必须 FAIL。

        没有这一条，G2 后面所有数字都不可信 —— 装置本身坏掉时会**静默给出
        看起来合理的结论**（本项目第一版就踩过：oracle 报 nan、leaky 报 1.000）。
        """
        x, _ = synthetic_batch(n=16, side=32, seed=0)
        r_or = evaluate(_Oracle(), x)
        r_lk = evaluate(_Leaky(), x)
        assert r_or.overall, f"oracle（构造上完全分离）竟未通过：{r_or.to_dict()}"
        assert not r_lk.overall, f"leaky（语义分支读细节块）竟通过：{r_lk.to_dict()}"
        dep_or = max(max(r["dep_semantic"], r["dep_detail"]) for r in r_or.rows)
        dep_lk = max(max(r["dep_semantic"], r["dep_detail"]) for r in r_lk.rows)
        return f"oracle 依赖 {dep_or:.4f}（PASS）/ leaky 依赖 {dep_lk:.4f}（FAIL）→ 尺子有分辨力"
    check("尺子双向校验：oracle 必过 + leaky 必挂", _g2_ruler)

    def _g2_supervision_matters():
        """⭐ G2 的核心断言：**显式监督买到了分离**，而且必须用 mix 来证明。

        `mix=0` 时分离是白送的 ⇒ 负对照也该过（证明装置**不误报**）；
        `mix>0` 时存在「语义分支顺手读细节块」的捷径 ⇒ 负对照 FAIL、
        有监督 PASS。**只测一个 mix 会得出错误结论**（第一版就栽在这里：
        只跑 mix=0，于是 w_inv=0 也 PASS，看起来「监督没用」）。
        """
        x0, t0 = synthetic_batch(n=16, side=32, seed=0, mix=0.0)
        neg0 = evaluate(train_separation(x0, t0, steps=250, w_inv=0.0, seed=0), x0)
        assert neg0.overall, f"mix=0 时负对照本应也通过（分离白送）：{neg0.to_dict()}"

        x1, t1 = synthetic_batch(n=16, side=32, seed=0, mix=0.6)
        sup = evaluate(train_separation(x1, t1, steps=250, w_inv=1.0, seed=0), x1)
        neg = evaluate(train_separation(x1, t1, steps=250, w_inv=0.0, seed=0), x1)
        assert sup.overall, f"有监督未通过：{sup.to_dict()}"
        assert not neg.overall, f"mix=0.6 时负对照本应走捷径而失败：{neg.to_dict()}"
        w = lambda r: max(max(q["dep_semantic"], q["dep_detail"]) for q in r.rows)
        return (f"mix=0 负对照也过（无假阴性）；mix=0.6 负对照 {w(neg):.4f} FAIL "
                f"vs 监督 {w(sup):.4f} PASS（{w(neg)/max(w(sup),1e-9):.1f}×）")
    check("显式监督确有净收益（mix 对照，非单点）", _g2_supervision_matters)

    def _g2_degenerate():
        """退化输入必须显式报缺口 —— 不许静默返回「全过」的假报告。"""
        x1, _ = synthetic_batch(n=1, side=32, seed=0)
        try:
            shuffle_within_batch(x1)
        except ValueError as e:
            assert "batch ≥ 2" in str(e)
        else:
            raise AssertionError("batch=1 时打乱竟未报错（会静默给出假结论）")
        try:
            evaluate(_Oracle(), x1)
        except ValueError as e:
            assert "batch ≥ 2" in str(e)
        else:
            raise AssertionError("batch=1 时验收竟未报错")
        return "batch=1 → 明确报缺口（不静默放假通过）"
    check("退化输入（batch=1）显式报缺口", _g2_degenerate)

    def _g2_mi_penalty():
        """MI 惩罚：独立通道应低于耦合通道（可微，训练侧可直接用）。"""
        torch.manual_seed(0)
        s = torch.randn(64, LATENT.semantic_ch, 8, 8)
        d_ind = torch.randn(64, LATENT.detail_ch, 8, 8)
        d_coupled = torch.randn(64, LATENT.detail_ch, 8, 8) + \
            s.repeat(1, LATENT.detail_ch // LATENT.semantic_ch, 1, 1)
        m_ind = float(_mi(*_split(torch.cat([s, d_ind], 1)), n_samples=512))
        m_cou = float(_mi(*_split(torch.cat([s, d_coupled], 1)), n_samples=512))
        assert m_cou > m_ind * 1.5, (m_ind, m_cou)
        return f"独立 {m_ind:.5f} < 耦合 {m_cou:.5f}（惩罚能分辨）"
    check("通道互信息惩罚能分辨独立/耦合", _g2_mi_penalty)

    def _g2_perturb_ops():
        """三种扰动算子的语义：只动指定通道块，其它块逐位不变。"""
        x, _ = synthetic_batch(n=8, side=32, seed=0)
        sc = LATENT.semantic_ch
        d = cross_perturb(x, "shuffle_detail")
        assert torch.equal(d[:, :sc], x[:, :sc]), "打乱细节竟动了语义块"
        assert not torch.equal(d[:, sc:], x[:, sc:]), "打乱细节后细节块没变"
        s = cross_perturb(x, "shuffle_semantic")
        assert torch.equal(s[:, sc:], x[:, sc:]), "打乱语义竟动了细节块"
        assert not torch.equal(s[:, :sc], x[:, :sc]), "打乱语义后语义块没变"
        return "shuffle_detail / shuffle_semantic / reverse 各自动对了该动的块"
    check("交叉扰动算子只动目标通道块", _g2_perturb_ops)

    def _g2_content_redundancy_ruler():
        """🔴 缺陷 ③ 的尺子校验：内容冗余测项**必须先过已知答案**。

        ⚠️ 这条自检的存在意义：本项目**同一个量被用过两种口径**——
          ①「按图展平」⇒ N=16 vs D=1152，**D≫N 欠定** ⇒ 最小二乘恒报 R²=1.0000；
          ②「`N·h·w` 个空间位置当样本」⇒ 良态，能分辨。
        ⚠️ 我曾用 ① 去质疑真图上 ② 得到的 0.988，**错误地撤回了正确结论**。
        ⇒ 这条断言的作用是**把「口径正确」这件事本身钉住**，防止再犯。
        """
        from kp.latent.separation import cross_predictability
        sc, dc = LATENT.semantic_ch, LATENT.detail_ch
        g = torch.Generator().manual_seed(0)
        n, s = 16, 16
        # 已知答案①：两块完全独立 ⇒ 应 ≈0
        sem = torch.randn(n, sc, s, s, generator=g)
        det = torch.randn(n, dc, s, s, generator=g)
        r_ind = cross_predictability(torch.cat([sem, det], 1))
        # 已知答案②：语义块被复制进细节块 ⇒ 应显著 >0
        det_cp = det.clone()
        det_cp[:, :sc, :, :] = sem
        r_cp = cross_predictability(torch.cat([sem, det_cp], 1))
        # 已知答案③：弱冗余 0.6 ⇒ 应居中
        det_wk = det.clone()
        det_wk[:, :sc, :, :] = 0.6 * sem + 0.4 * det_wk[:, :sc, :, :]
        r_wk = cross_predictability(torch.cat([sem, det_wk], 1))
        a, b, c = (r_ind["cross_r2_detail_from_sem"],
                   r_wk["cross_r2_detail_from_sem"],
                   r_cp["cross_r2_detail_from_sem"])
        assert a < 0.02, f"完全独立竟报 {a:.4f}（口径错，欠定最小二乘？）"
        assert c > 0.15, f"精确拷贝竟只报 {c:.4f}（量没分辨力）"
        assert a < b < c, f"三档不单调：独立 {a:.4f} / 弱冗余 {b:.4f} / 拷贝 {c:.4f}"
        return f"独立 {a:.4f} < 弱冗余 {b:.4f} < 拷贝 {c:.4f}（位置级口径，三档单调）"
    check("内容冗余测项先过已知答案（口径守卫）", _g2_content_redundancy_ruler)

    def _g2_redundancy_cannot_judge_encoder():
        """⚠️ 反向断言：`cross_r2` **不能**当「编码器有没有分开」的判据。

        `_Oracle` 的 `sem_path`/`det_path` 就是输入的两块切片 ⇒ 它的输出两块
        **恒等于**输入 ⇒ 测 oracle 等于测输入 ⇒ **在这个尺子上无任何分辨力**。
        ⇒ 写死这条断言，避免以后又有人拿 `cross_r2` 当编码器判据。
        """
        from kp.latent.separation import cross_predictability
        x, _ = synthetic_batch(n=16, side=16, seed=0, mix=0.6)
        o = _Oracle()
        with torch.no_grad():
            s, d = o.sem_path(x), o.det_path(x)
        r_out = cross_predictability(torch.cat([s, d], 1))
        r_in = cross_predictability(x)
        assert abs(r_out["cross_r2_sem_from_detail"]
                   - r_in["cross_r2_sem_from_detail"]) < 1e-6, (
            "oracle 输出不再等于输入切片 —— 若这是真的，说明 _Oracle 的构造变了，"
            "本断言与文档需要一起更新")
        return ("oracle 输出恒等于输入切片 ⇒ cross_r2 测的是 latent 内容、"
                "不是编码器行为（0.988 不能这么解读）")
    check("冗余量不可当编码器判据（防止再次误用）", _g2_redundancy_cannot_judge_encoder)

    def _g2_real_separation_smoke():
        """🔴 真图版 G2 装置的**冒烟**（997 行此前**零自检覆盖**）。

        ⚠️ **断言的是「结构完整 + 缺口被报出」，不是 verdict 本身** ——
            `verdict` 会随分辨率/数据波动（本 smoke 实测 FAIL，384px 全量跑则 PASS），
            拿它当断言就是**制造假失败**。
        ⭐ 三个必须恒真的不变量（它们才说明装置真的跑起来了）：
            ① `sanity_ok` True  ⇒ 尺子双向校验过了（oracle 过 / leaky 挂）
            ② `len(arms) == 5`  ⇒ 五个对照臂都在（含负对照与塌缩体检）
            ③ `conclusion_strength` 非空 + **低分辨率时必须报出分辨率缺口**
        """
        from kp.latent.real_separation import run_real_g2
        rep = run_real_g2(size=128, steps=20, n_perm=3, max_views=8,
                          synth_steps=30)
        assert rep.get("sanity_ok") is True, f"尺子双向校验没过：{rep.get('rulers')}"
        arms = rep.get("arms") or {}
        expected = {"frozen_random_encoder", "joint_supervised", "joint_negative",
                    "joint_no_anchor", "semantic_routed"}
        assert set(arms) == expected, f"实验臂不全：缺 {expected - set(arms)}"
        assert rep.get("conclusion_strength") not in (None, ""), "结论强度未标注（不许裸给数字）"
        gaps = rep.get("gaps") or []
        assert any("分辨率" in g for g in gaps), (
            f"128² 低分辨率**必须**报出分辨率缺口，否则这个数字会被当真：gaps={gaps}")
        gap = rep.get("gap") or {}
        return (f"5 臂全跑通 · sanity_ok ✓ · strength={rep.get('conclusion_strength')} · "
                f"缺口已报「{gaps[0][:24]}…」· 监督净收益 {gap.get('improvement_x', 0):.2f}×")
    check("真图 G2 装置冒烟（997 行纳入覆盖）", _g2_real_separation_smoke)

    # ---------------- 20. G3.5 真探针（真主干 + 真轴注入路径） ----------------
    section("20. G3.5 真探针（真主干 + 真轴注入路径 · 非空性守卫 + 真量化 ④）")
    from kp.config import AXIS as _AXIS
    from kp.probe import real as _RPB
    from kp.probe.axis import _spearman as _rho
    # ⚠️ 本节自己要用 AxisProbe 验「死轴不许假通过 ①」——
    #    若只依赖第 16 节的 import，则**单独跑本节会 NameError**（节序侥幸能用 ≠ 正确）。
    from kp.probe.axis import AxisProbe as _AxisProbe

    # 小测试形状：纯 CPU 可跑是硬要求（不跑 1024² 全尺寸）
    _RP = dict(dim=64, layers=4, heads=4, tokens=6, n_random=32)

    def _rp_adaln_zero():
        """⭐ 设计不变量：adaLN-Zero ⇒ **初值时域条件向量进不去输出**。

        守的不是「代码现在这样」，而是设计稿写死的 adaLN-Zero 语义
        （`dit.py`：adaLN 末层 weight 全 0、只把 attn/mlp/txt 三个 gate 的 bias 置 1；
        身份门控第 8 段保持 0）。⇒ 推论：**未训练主干上 G3.5 根本无从谈起** ——
        这不是「轴不成立」，而是「门还没开」。

        ⚠️ 必须在**同一 batch 形状**下比对：跨 batch 会因 GEMM 分块差异产生
           ~1e-8 的假差异（实测根因，不是猜测）。
        """
        m = _RPB.build_test_backbone(seed=0)
        g = torch.Generator().manual_seed(7)
        x = torch.randn(3, 40, 6, 6, generator=g)
        t = torch.full((3,), 500.0)
        with torch.no_grad():
            a = m(x, t, domain=torch.zeros(3, 16))
            b = m(x, t, domain=torch.randn(3, 16, generator=g))
            c = m(x, t)
        assert torch.equal(a, b), "adaLN-Zero 下域向量竟改变了输出"
        assert torch.equal(a, c), "domain=None 与 domain=zeros 竟不同"
        return "domain=0 / domain=随机 / domain=None 三者**逐位相同** ⇒ 门是死的"
    check("adaLN-Zero：初值时域条件向量进不去输出（逐位）", _rp_adaln_zero)

    def _rp_nonvacuous():
        """⭐ 死轴**必须**判 ①不成立 —— 两道防线都要验。

        ① 的语义是「轴值单调 ⇒ 输出沿该维单调变化」。一条**完全没有响应**的轴
        必须判 ① 不成立。但秩相关对常量没有定义，**原口径实测给出 ρ=1.00**
        （根因：`torch.argsort` 对常量张量返回原序索引 `[0,1,2,…]`，排完秩后
        与轴值的等距秩完全相关），② 的零方向余弦 0、③ 分母为 0 被跳过
        ⇒ **一条死轴会拿到「3/4 通过」**，即设计稿要害处的静默假通过。

        现在两道防线都在（2026-10-03 补）：
          · **装置层**：`axis._spearman` 对常量输入返回 0（不再是 1.0）；
          · **流程层**：真探针在四测前做非空性预检（判据 = 响应是否显著高于
            实测数值噪声底线；CPU oneDNN GEMM 行间噪声 ~1e-8）。
        ⚠️ 本断言**故意同时钉两条**：任何一条单独被削弱，死轴都可能再次假通过。
        """
        c = torch.zeros(9)
        r = _rho(torch.linspace(-1.0, 1.0, 9), c)
        # 装置层：退化输入必须判 0（**设计承诺**：无变化 ⇒ 无相关性可言）
        assert r == 0.0, f"装置层守卫失效：ρ(常量) 应为 0，实测 {r}"
        # 流程层：只有装置层修好还不够，必须端到端确认死轴不通过 ①
        dead = _AxisProbe(lambda V: torch.ones(V.shape[0], 8), n_axes=4, seed=0).run()
        assert dead.n_pass == 0, f"死 responder 竟通过 {dead.n_pass}/4"
        assert all(q.mono == 0.0 for q in dead.results), "死轴的 ① 必须判 0"

        m = _RPB.build_test_backbone(seed=0)                 # 门关（主干初值）
        rep = _RPB.run_real_probe(model=m, door=False, **_RP)
        assert rep.n_inert == 16, f"门关时 16 条轴应全判 inert，实测 {rep.n_inert}"
        assert all(q.mono == 0.0 for q in rep.probe.results), "inert 轴的 ① 必须判 0"
        assert rep.n_pass == 0 and len(rep.l3_axes) == 16, rep.l3_axes
        return (f"装置层 ρ(常量)={r:.2f}；流程层：门关 16/16 inert、① 全 0、全部归 L3")
    check("非空性守卫：无响应的轴不许假通过 ①", _rp_nonvacuous)

    def _rp_door_and_reproducible():
        """开门后注入路径必须**可观测**、且必须**真的接在 domain_embed 上**。"""
        kw = dict(door=True, door_scale=0.02, **_RP)
        r1 = _RPB.run_real_probe(seed=0, **kw)
        r2 = _RPB.run_real_probe(seed=0, **kw)
        assert r1.n_inert == 0, "开门后仍判 inert ⇒ 门没开、或没接在真路径上"
        for a, b in zip(r1.probe.results, r2.probe.results):
            assert (a.mono, a.ortho, a.rev, a.travel_keep) == \
                   (b.mono, b.ortho, b.rev, b.travel_keep), f"不同种子不可复现：{a.name}"
        # 响应必须真的经过 domain_embed：清零它的权重 ⇒ 响应必须塌回噪声
        m = _RPB.build_test_backbone(seed=0)
        _RPB.open_domain_door(m, scale=0.02, seed=0)
        resp = _RPB.make_responder(m, tokens=6, seed=0)
        dev0, floor = _RPB.domain_response_dev(m, resp, [0])
        with torch.no_grad():
            m.domain_embed.weight.zero_()
        dev1, _ = _RPB.domain_response_dev(m, resp, [0])
        d0, d1 = float(dev0[0]), float(dev1[0])
        assert d1 < max(d0 * 0.01, 10.0 * floor), (
            f"清零 domain_embed 后响应仍有 {d1:.2e}（原 {d0:.2e}）⇒ 未接在该路径上")
        return (f"16/16 有响应；同种子两次四测数字完全一致；"
                f"清零 domain_embed ⇒ 响应 {d0:.2e}→{d1:.2e}（噪声底 {floor:.1e}）")
    check("真探针确实接在 domain_embed 注入路径上（且可复现）", _rp_door_and_reproducible)

    def _rp_quantizer_in_loop():
        """⭐ ④ 必须**真的过一遍量化器**（设计稿点名「不可被前三测替代」）。

        守两条不变量：
          ⓐ 量化回路真的接进了主干 —— W4A8 下响应必须与 bf16 **不同**
             （若逐位相同，说明量化器根本没挂上，④ 就是假的）；
          ⓑ 「信号弱」必须由 ④ 单独抓 —— 在真路径上把一条轴压到量化台阶以下：
             该轴 ④ 挂，而 ① 仍过 ⇒ **④ 不可被 ① 替代**（否则设计稿的要害论点不成立）。
        """
        m = _RPB.build_test_backbone(seed=0)
        _RPB.open_domain_door(m, scale=0.02, seed=0)
        rep = _RPB.run_real_probe(model=m, door=False, **_RP)
        qerr = [rep.aux[i]["quant_relative_error"] for i in range(16)]
        assert min(qerr) > 1e-4, f"量化后响应与 bf16 相同 ⇒ 量化器没进回路：{min(qerr)}"

        m2 = _RPB.build_test_backbone(seed=0)
        _RPB.open_domain_door(m2, scale=0.02, seed=0)
        # ⭐ 必须是**配对对照**：选一条在参考状态下 ④ 本来**能过**的轴来压，
        #    否则「压了之后 ④ 挂」可能只是它本来就挂（假对照）。
        best = max(range(16), key=lambda i: rep.probe.results[i].travel_keep)
        assert rep.probe.results[best].travel_keep >= _AXIS.travel_threshold, \
            f"参考状态下没有一条轴能过 ④（最好 {rep.probe.results[best].travel_keep:.2f}）⇒ 对照无效"
        _RPB.corrupt_domain_embed(m2, kind="suppressed", a=best, factor=0.02)
        rep2 = _RPB.run_real_probe(model=m2, door=False, **_RP)
        sup = rep2.probe.results[best]
        ref_t = rep.probe.results[best].travel_keep
        assert "低比特行程" in sup.failures, \
            f"被压到量化台阶下的轴未挂 ④：行程 {ref_t:.2f}→{sup.travel_keep:.2f}" \
            f"（失败项 {sup.failures}）"
        assert sup.mono >= 0.9, f"① 不该抓这个（那是 ④ 的职责）：mono={sup.mono}"
        assert not rep2.aux[best]["inert"], "应落在「弱响应」而不是「无响应」"
        return (f"量化相对误差 ≥{min(qerr):.1e}（器真在回路里）；"
                f"配对压下轴{best}：④ 行程 {ref_t:.2f}→{sup.travel_keep:.2f}（必挂），"
                f"而 ① 仍 {sup.mono:.2f}")
    check("④ 真的过量化器，且不可被 ① 替代", _rp_quantizer_in_loop)

    def _rp_coupled_control():
        """负对照（真路径可证伪）：两轴**共线注入** ⇒ ② 必须爆掉。"""
        m = _RPB.build_test_backbone(seed=0)
        _RPB.open_domain_door(m, scale=0.02, seed=0)
        _RPB.corrupt_domain_embed(m, kind="coupled", a=2, b=3)
        rep = _RPB.run_real_probe(model=m, door=False, **_RP)
        o2 = rep.probe.results[2].ortho
        o3 = rep.probe.results[3].ortho
        assert o2 > 0.95 and o3 > 0.95, (o2, o3)
        assert "正交" in rep.probe.results[2].failures
        return f"共线轴 ② 串扰 → {o2:.3f} / {o3:.3f}（远超门线，必挂）"
    check("负对照：真路径上的共线注入被 ② 抓住", _rp_coupled_control)

    def _rp_ortho_power():
        """⭐ ② 的**分辨力前提**：门线必须高于「随机方向基线」，否则测的是维度不是解耦。

        16 个轴的方向都住在同一个读出空间里：D 维中 n 个随机方向的 max|cos| 有基线。
        D=48（结构化读出）时基线已 **高于** 门线 0.30 ⇒ ② 在该读出下**没有分辨力**
        （实测未训练主干上 16 条轴全在基线附近）；D=1440（全输出场）时基线远低于门线
        ⇒ ② 才谈得上判别。这条断言把「读出维选择」钉成了 ② 结论的前置条件。
        """
        b48 = _RPB.orthogonality_null_baseline(16, 48, trials=100, seed=0)
        bhi = _RPB.orthogonality_null_baseline(16, 1440, trials=100, seed=0)
        assert b48["mean_max_cos"] > _AXIS.ortho_threshold, \
            f"前提变了：D=48 的随机基线本应高于门线，实测 {b48}"
        assert bhi["mean_max_cos"] < _AXIS.ortho_threshold, bhi
        return (f"D=48 基线 {b48['mean_max_cos']:.3f} > 门线 {_AXIS.ortho_threshold}"
                f"（无分辨力）｜D=1440 基线 {bhi['mean_max_cos']:.3f}（有分辨力）")
    check("② 的分辨力基线：低维读出下门线不可达", _rp_ortho_power)

    # ---------------- 21. G3 Sigmoid 注意力（机制层装置） ----------------
    section("21. G3 Sigmoid 注意力装置（机制层 · ⛔ 非过门依据）")
    from kp.probe import attn as _G3
    from kp.models.dit import SIGMOID as _SIG, SOFTMAX as _SM

    def _g3_estimator_known_answer():
        """⭐ **已知答案（层①：估计器本身）**：α 拟合器必须在**构造样本**上给出 1 / 0。

        ⚠️ 为什么单独验这一层：下面三项走的是真实代码路径，验的是**设计主张**；
        但如果 α 的拟合逻辑本身算错了，那三项的数字全无意义。
        ⇒ 两层互补（照 `known_answer_samples` 的 docstring）：
           层①验「尺子准不准」，层②验「测的是不是真东西」。
        """
        ka = _G3.known_answer_samples()
        assert abs(ka["flat_alpha"] - 1.0) < 1e-6, ka
        assert abs(ka["concentrated_alpha"] - 0.0) < 1e-6, ka
        return (f"flat α={ka['flat_alpha']:.4f}（理论 1）/ "
                f"concentrated α={ka['concentrated_alpha']:.4f}（理论 0）")
    check("已知答案：α 估计器在构造样本上给出 1 / 0（尺子本身准）", _g3_estimator_known_answer)

    def _g3_identity():
        """装置地基：one-hot v 恒等式 ⇒ out[...,j] 必须**逐位**等于真实权重。"""
        w = _G3.attention_weights(_SIG, [3.0, 0.5, -1.0, 2.0, 0.0])
        # 真算一遍权重，与「从 out 反解出来的」对拍
        lg = w.logits[0, 0, 0]
        ref = torch.sigmoid(lg)
        assert torch.allclose(w.weights[0, 0, 0], ref, atol=1e-6), \
            float((w.weights[0, 0, 0] - ref).abs().max())
        s = w.shares()[0, 0, 0]
        assert abs(float(s.sum()) - 1.0) < 1e-5, float(s.sum())
        # ⭐ sigmoid 权重**不和为 1**（这正是它与 softmax 的分水岭）
        assert abs(float(w.total_mass()[0, 0, 0]) - 1.0) > 1e-3, \
            "sigmoid 权重总和居然≈1，装置可能读错了"
        return f"sigmoid 总质量 {float(w.total_mass()[0,0,0]):.3f}（≠1 ✅ 非归一化确认）"
    check("装置能反解真实代码路径上的精确权重", _g3_identity)

    def _g3_known_answer():
        """⭐ **已知答案**：等 logit ⇒ 份额恒 1/N ⇒ α=1，两种机制都必须给出 1.0。"""
        r = {}
        for k in (_SM, _SIG):
            a = _G3.dilution_exponent(k, sig_logit=0.0, bg_logit=0.0)
            assert abs(a - 1.0) < 1e-3, f"{k} 等 logit 的 α 应恒为 1.0，实测 {a:.4f}"
            r[k] = a
        return f"softmax α={r[_SM]:.4f} / sigmoid α={r[_SIG]:.4f}（理论 1.0000 ✅）"
    check("已知答案：等 logit ⇒ α 必为 1（尺子有分辨力）", _g3_known_answer)

    def _g3_softmax_shift_invariant():
        """⭐ 已知答案：**softmax 可用抬高信号**把 α 压到 ~0 —— 它只看间距（平移不变）。"""
        a_far = _G3.dilution_exponent(_SM, sig_logit=50.0, bg_logit=0.0)
        a_near = _G3.dilution_exponent(_SM, sig_logit=8.0, bg_logit=0.0)
        assert a_far < 0.05, f"间距 50 时 softmax 应当几乎不稀释，α={a_far:.4f}"
        assert a_far < a_near, (a_far, a_near)
        return f"间距50 α={a_far:.4f} ≤ 间距8 α={a_near:.4f}（间距越大越不稀释）"
    check("已知答案：softmax 靠「抬信号」不稀释（平移不变）", _g3_softmax_shift_invariant)

    def _g3_sigmoid_needs_negative_bg():
        """⭐ 已知答案：**sigmoid 只能靠把背景压到负侧**才不稀释 —— 它看绝对零点。"""
        a_bg0 = _G3.dilution_exponent(_SIG, sig_logit=4.0, bg_logit=0.0)
        a_bgneg = _G3.dilution_exponent(_SIG, sig_logit=0.0, bg_logit=-50.0)
        assert a_bg0 > 0.9, f"背景在 0 时 sigmoid 应当强稀释，α={a_bg0:.4f}"
        assert a_bgneg < 0.05, f"背景压到 -50 时不应稀释，α={a_bgneg:.4f}"
        return (f"背景=0 → α={a_bg0:.4f}（强稀释）｜背景=-50 → α={a_bgneg:.4f}（不稀释）"
                f" ⇒ sigmoid **不是**天然不稀释，取决于绝对零点")
    check("已知答案：sigmoid 靠「压背景到负侧」不稀释", _g3_sigmoid_needs_negative_bg)

    def _g3_negative_control():
        """⭐ **负对照**：零点敏感性 —— softmax 恒定，sigmoid 跨数量级塌陷。

        这是本轮最关键的发现：sigmoid **没有平移不变性**。
        """
        offs = (-12.0, -8.0, -4.0, 0.0, 4.0)
        sm = _G3.contrast_vs_offset(_SM, 8.0, offs)
        sg = _G3.contrast_vs_offset(_SIG, 8.0, offs)
        # softmax：对比度必须与零点无关（数学恒等式）
        rel = (max(sm) - min(sm)) / max(sm)
        assert rel < 1e-9, f"softmax 对比度不该随零点变，实测变化 {rel:.2%}"
        # sigmoid：零点一漂就必须塌（否则这条负对照没有分辨力）
        drop = max(sg) / max(min(sg), 1e-30)
        assert drop > 1e3, f"sigmoid 对零点应极敏感，实测只差 {drop:.1f}×（无分辨力？）"
        return (f"softmax 恒 {sm[0]:,.0f}×（变化 {rel:.1%}）｜"
                f"sigmoid {max(sg):,.0f}× → {min(sg):,.1f}×（塌 {drop:,.0f}×）")
    check("负对照：softmax 平移不变 vs sigmoid 零点敏感", _g3_negative_control)

    def _g3_plan():
        """3:1 混合注意力构成 + 第 0 层 softmax 锚点（L308 / L313）。

        🔴🔴 **这条断言在 2026-10-03 被重写（原版违反本项目教训 #5）**：

        **原版**：`assert 0.2 <= ratio <= 0.35`，实测 0.2917。
        ⛔ **问题**：那个容差带是围绕「实际发生的 1:3.43」设计的，**不是围绕设计目标 1:3**。
        它无法区分「符合设计」与「偏离设计但落在宽容带内」⇒ **把偏离固化成了合格**。

        ⭐ **结构性根因**（`build_attn_plan`）：softmax 锚点占掉第 0 层后，剩余 `L−1` 个槽位，
        而 **只有 `L ≡ 1 (mod 4)` 时 `[L,L,L,S]` 才能整除**。实测：
            L=24 → 1:3.60（L−1=23 ≡ 3 mod 4）
            L=32 → 1:3.43（L−1=31 ≡ 3 mod 4）  ← KP-S / KP-M 都是这个
            L=25 → **1:3.00 ✓**（25 ≡ 1 mod 4）
            L=33 → **1:3.00 ✓**
        ⇒ KP-S=24 / KP-M=32 **在结构上就不可能精确 3:1**（除非改层数）。

        ✅ **现在断言的是「设计意图」而非「当前行为」**：
            ① 锚点恰好 1 次、第 0 层（硬不变量）
            ② **每 4 层窗口内严格 3 linear + 1 sigmoid**（这是设计真正的承诺）
            ③ 报告**如实给出**全局比例与「差多少」，不假装它是 3:1
        """
        from kp.models.dit import build_attn_plan, SOFTMAX, LINEAR, SIGMOID
        c = _G3.plan_composition(32)
        assert c["softmax"] == 1, f"softmax 锚点应恰好 1 次，实测 {c}"
        assert c["layers"] == 32, c
        # ② 逐 4 层窗口校验（锚点之后）：这是「每 4 层 3+1」的字面承诺
        plan = build_attn_plan(32, 3, 1, True)
        assert plan[0] == SOFTMAX, f"第 0 层应为 softmax 锚点，实测 {plan[0]}"
        body = plan[1:]
        for s in range(0, len(body) - 3, 4):
            win = body[s:s + 4]
            assert win.count(LINEAR) == 3 and win.count(SIGMOID) == 1, (
                f"第 {s + 1}–{s + 4} 层窗口应严格 3 linear + 1 sigmoid，实测 {win}")
        # ③ 如实报告偏差（**不 assert 它等于 1/3** —— 结构上就不可能）
        ratio = c["sigmoid"] / max(c["linear"], 1)
        # 顺带证明「L ≡ 1 (mod 4) 才行」这个根因
        ok = _G3.plan_composition(25)
        assert ok["sigmoid"] / max(ok["linear"], 1) == 1 / 3, (
            f"L=25 应能整除到精确 1:3，实测 {ok} ⇒ 根因判断有误，需重新理解 build_attn_plan")
        b = _G3.qknorm_logit_bound(1792, 16)
        return (f"{c} ｜ 每 4 层窗口严格 3+1 ✓ ｜ 全局 1:{1/ratio:.1f}"
                f"（L=32 时结构上无法整除；L=25 时精确 1:3 ✓）｜ QK-Norm 上界 {b:.2f}")
    check("3:1 每 4 层窗口严格 3+1（不把偏离固化成合格）", _g3_plan)

    def _g3_quant_friendly():
        """🔴 L311「减少激活离群值」—— **已有通过门线**（2026-10-03 用户拍板）。

        ⭐ **判据缺口已闭合**：设计稿 §9.2 的 G3 行把「量化友好性」写在「验证什么」列，
        但「**通过判据**」列**没有条目** ⇒ 此前只能报方向、不敢作门。
        ⇒ 用户拍板补：**离群比 ≤ 2× softmax**（实测 4.62 vs 13.18 = **2.85× 更低**）。

        ⭐ **为什么门线取 2.0 而不是「越低越好」**：
        · 实测增益是 **2.85×** ⇒ 门线 2.0 留了 ~30% 余量，**不会因随机波动假红**；
        · 门线 <1.0 意味着 sigmoid 必须比 softmax **还好**，那是不可证伪的强主张。
        ⚠️ 门线**只约束方向 + 幅度**（量化友好性），**不**替代 L310（长提示收益）——
           那一半实测**未获支持**（见 §21 的平移不变性负对照）。
        """
        outs = {}
        for n in (128, 512):
            a = _G3._act_stats_for(_SM, n_tokens=n, seed=0)
            b = _G3._act_stats_for(_SIG, n_tokens=n, seed=0)
            outs[n] = (a, b)
        sm512, sg512 = outs[512]
        assert sg512["outlier_ratio"] < sm512["outlier_ratio"], \
            f"N=512 时 sigmoid 离群比应更低：sm={sm512} sg={sg512}"
        gain = sm512["outlier_ratio"] / max(sg512["outlier_ratio"], 1e-9)
        # 🔴 正式门线（用户拍板）：离群比不得高于 softmax 的 2 倍
        GATE = 2.0
        assert gain >= GATE, (
            f"量化友好性未达门线：实测 {gain:.2f}× < 要求的 {GATE}× "
            f"（sm {sm512['outlier_ratio']:.2f} / sg {sg512['outlier_ratio']:.2f}）"
            f" ⇒ 若真如此，**Sigmoid 层对 NVFP4 不再是优势**，3:1 的立论要重估")
        return (f"N=128 离群比 sm {outs[128][0]['outlier_ratio']:.2f} / sg {outs[128][1]['outlier_ratio']:.2f}"
                f"｜N=512 sm {sm512['outlier_ratio']:.2f} / sg {sg512['outlier_ratio']:.2f}"
                f"（**{gain:.2f}× 更低 ≥ 门线 {GATE}× ✅**）")
    check("🔴 L311 量化友好性：离群比 ≤2× softmax（门线已补）", _g3_quant_friendly)

    # ---------------- 22. 排版链路闭环（plan → ROIBranch → composite） ----------------
    section("22. 排版链路闭环（拼回 latent · 判据可证伪）")
    from kp.typography.composite import (  # noqa: E402
        composite_latent,
        run_acceptance as _tc_accept,
        DegenerateLayoutError as _TCL,
    )

    def _tc_acceptance():
        """⭐ 三件套闭环的 12 条用例：正样本 1 + **负对照 3** + 退化 5 + 重叠 2。

        ⚠️ 负对照是关键：全零 / 随机噪声 / 平移一格都必须**判不合格**。
        只有正样本能过的判据是摆设（照 G2 / G3.5 的老规矩）。
        """
        r = _tc_accept()
        assert r["ok"], f"排版闭环验收失败 {r['n_failed']}/{r['n_cases']}：" + \
                        "; ".join(c["name"] for c in r["cases"] if not c["ok"])
        pos = [c for c in r["cases"] if c["name"].startswith("正样本")][0]
        neg = [c for c in r["cases"] if c["name"].startswith("负对照")]
        deg = [c for c in r["cases"] if c["name"].startswith("退化")]
        ovl = [c for c in r["cases"] if c["name"].startswith("重叠")]
        return (f"{r['n_cases']}/{r['n_cases']} 通过（正样本 1 / 负对照 {len(neg)} / "
                f"退化 {len(deg)} / 重叠 {len(ovl)}）")
    check("12 条用例全过（含 3 条负对照 + 5 条退化）", _tc_acceptance)

    def _tc_degenerate_raises():
        """退化输入必须**抛** `DegenerateLayoutError`，不能静默返回。

        ⛔ 空 layout 直接传给 `windows_from_layout` —— 这是唯一入口，
        所以断言它在这里抛出，而不是去测别的包装函数。
        """
        from kp.typography.composite import windows_from_layout as _wf
        try:
            _wf({}, scale=32, shape=(32, 32))
        except _TCL as e:
            return f"空 layout 正确抛 DegenerateLayoutError：{str(e)[:44]}"
        raise AssertionError("空 layout 竟然没抛异常 ⇒ 退化输入被静默吞掉了")
    check("退化输入：空 layout 抛错而非静默", _tc_degenerate_raises)

    # ---------------- 23. M3 修复预实验（真图冗余 0.988 的三条修法）----------------
    section("23. M3 Latent 解耦修复（构造级 · 真图待验）")

    def _m3_split_param_equality():
        """🔴 修法② 的**参数量恒等**必须精确成立（不是"约等于"，是 **0 差**）。

        恒等式：`(8+32)·d ≡ 40·d`（单路 `40→d`）⇒ 拆两路只多一次加法。
        ⭐ 这是「拆路免费」这个结论的**唯一硬证据**。
        ⚠️ 顺带守一条：本项目教训「『预留接口』不该用『每层都付钱』的方式存在」——
           若有人把 `self.patch_embed` 与 split 两路**同时建**，会多出 `40·d` **死参数**。
        """
        import dataclasses
        from kp.config import DIT_S, DIT_M, LATENT
        from kp.models.dit import SingleStreamDiT

        def n_params(cfg, split):
            c = dataclasses.replace(cfg, split_patch_embed=split)
            torch.manual_seed(0)
            m = SingleStreamDiT(c, latent_ch=LATENT.total_ch,
                                identity_anchor_layers=[1, c.layers // 2])
            return (sum(p.numel() for p in m.parameters())
                    + sum(b.numel() for b in m.buffers() if b.is_floating_point()))

        diffs = []
        for name, cfg in (("KP-S", DIT_S), ("KP-M", DIT_M)):
            a, b = n_params(cfg, False), n_params(cfg, True)
            assert a == b, f"{name} 拆两路后参数量变了：{a} -> {b}（差 {b - a}）"
            diffs.append(f"{name} 0")
        return f"共享 ≡ 拆两路，参数量**精确相等**（{'/'.join(diffs)}）"
    check("修法② 参数量精确恒等（无死参数）", _m3_split_param_equality)

    def _m3_split_default_off():
        """⛔ 修法② **默认关闭** —— 它改变架构行为，须在 P2 换主干时显式拍板。"""
        from kp.config import DIT_S, DiTCfg
        assert DIT_S.split_patch_embed is False, "split_patch_embed 不该默认开启"
        assert DiTCfg().split_patch_embed is False, "DiTCfg 默认值不该是 True"
        return "split_patch_embed 默认 False（架构变更须显式拍板）"
    check("修法② 默认关闭（不在主干里偷做架构变更）", _m3_split_default_off)

    def _m3_ruler():
        """预实验的**尺子**：起点必须有分辨力，否则"压下来了"是假象。"""
        from kp.probe.m3_fix import make_high_redundancy, cross_r2
        vals = []
        for noise in (0.20, 0.50, 1.00):
            r = cross_r2(make_high_redundancy(noise=noise))
            vals.append(r["det_from_sem"])
        assert vals[0] > 0.9, f"强冗余起点只报 {vals[0]:.4f}（造的数据不够冗余）"
        assert vals[0] > vals[1] > vals[2], f"三档不单调：{vals}"
        return f"强 {vals[0]:.3f} > 中 {vals[1]:.3f} > 弱 {vals[2]:.3f}（起点有分辨力）"
    check("起点可分性（造的数据真有冗余）", _m3_ruler)

    def _m3_fix_grad():
        """修法①：可微跨块去相关惩罚 ⇒ 能把冗余压下来。"""
        from kp.probe.m3_fix import make_high_redundancy, cross_r2, apply_fix_grad
        z0 = make_high_redundancy(noise=0.20)
        before = cross_r2(z0)["det_from_sem"]
        after = cross_r2(apply_fix_grad(z0))["det_from_sem"]
        assert after < before * 0.5, f"修法① 没压下来：{before:.4f} -> {after:.4f}"
        return f"det|sem {before:.4f} -> {after:.4f}（压到门线 0.10 以下）"
    check("修法① 可微去相关惩罚能压冗余", _m3_fix_grad)

    def _m3_split_patch():
        """修法②：patch_embed 拆两路、不共享权重 ⇒ 参数量零代价。"""
        from kp.probe.m3_fix import make_high_redundancy, cross_r2, SplitPatchEmbed
        z0 = make_high_redundancy(noise=0.20)
        before = cross_r2(z0)["det_from_sem"]
        after = cross_r2(SplitPatchEmbed.trainable(z0))["det_from_sem"]
        assert after < before * 0.5, f"修法② 没压下来：{before:.4f} -> {after:.4f}"
        # 参数量恒等：(8+32)·d == 40·d，与单路 40→d 完全相同
        d = 1152
        assert (LATENT.semantic_ch + LATENT.detail_ch) * d == LATENT.total_ch * d
        return f"det|sem {before:.4f} -> {after:.4f}；参数量 (8+32)·d = 40·d **零代价**"
    check("修法② 拆两路能压冗余且零参数量代价", _m3_split_patch)

    def _m3_no_fake_claim():
        """⚠️ 防过度外推守卫：本预实验**只在构造数据上**做过。

        真图（cross_r2=0.988）上能不能压下来、会不会损失重建质量，
        **本实验都没有测** ⇒ 任何"已证明 M3 可修"的说法都是错的。
        """
        from kp.probe.m3_fix import make_high_redundancy
        z = make_high_redundancy(noise=0.20)
        assert z.shape[1] == LATENT.total_ch
        return ("预实验仅覆盖构造级；真图 0.988 的可修性 + 重建代价**均未测**（勿宣称已修）")
    check("预实验边界声明（防把构造结论当真图结论）", _m3_no_fake_claim)

    # ---------------- 24. P1 VAE 训练器（过拟合判据 + 固定评估集）----------------
    section("24. P1 VAE 训练器（⭐ 小数据量下不许用随机 batch 判收敛）")

    def _p1_fixed_eval_set_exists():
        """🔴 守卫：训练器**必须有固定评估集**。

        背景（2026-10-03 实测打脸）：
          11 张图 + `batch=4` 随机抽 ⇒ 训练日志呈「0.321 → **0.386 反弹**」
          ⇒ 我据此判断「过拟合」。⛔ **判断错了** ——
          同一 checkpoint 用**固定全量 11 张**评估，l1 = 0.1378（比日志的 0.1719 更低）
          ⇒ **所谓反弹是采样噪声，不是过拟合。**
        ⇒ 这条断言的作用：防止有人把「随机 batch 的 loss 曲线」当收敛判据。
        """
        from kp.train.vae_pretrain import fixed_eval_set, evaluate
        import inspect
        assert callable(fixed_eval_set), "缺少 fixed_eval_set"
        assert callable(evaluate), "缺少 evaluate"
        # 固定集必须**确定**：同一输入两次调用结果逐位相同
        import torch as _t
        from kp.train.vae_pretrain import list_images
        from kp.paths import DATA
        paths = list_images([DATA / "characters" / "kokona"])
        if not paths:
            return "⚠️ 无 kokona 图，跳过逐位校验（仅断言函数存在）"
        a = fixed_eval_set(paths, 32, _t.device("cpu"))
        b = fixed_eval_set(paths, 32, _t.device("cpu"))
        assert _t.equal(a, b), "固定评估集两次调用结果不同 ⇒ 不是确定性固定集"
        return f"固定集确定性 ✓（{a.shape[0]} 张 · 两次调用逐位相同）"
    check("训练器有固定评估集（不许用随机 batch 判收敛）", _p1_fixed_eval_set_exists)

    def _p1_multiscale_is_cheap_and_differentiable():
        """多尺度感知损失：可回传 + **真的看重结构**。

        ⚠️ **这个断言的样本选择踩过两次坑（留档）**：
          ① 先用 `torch.randn` 随机噪声当 rec ⇒ 「抹平结构」后损失反而**变小**
             —— 随机图没有"结构"可言，抹平等于靠近通道均值期望，**样本选错**。
          ② 改用 `torch.zeros` + 矩形 ⇒ 梯度为 0 是**常量图** ⇒ `_edge` 全 0、两项都 0
             —— 同样是**样本选错**。
        ✅ 正确做法：**用项目里的真实图片**（`data/characters/kokona`），
           因为要验的命题是「结构被抹平时损失变大」，**样本本身必须真有结构**。
        """
        import torch as _t
        from kp.train.vae_pretrain import multiscale_perceptual, list_images, load_batch
        from kp.paths import DATA
        paths = list_images([DATA / "characters" / "kokona"])
        if not paths:
            return "⚠️ 无 kokona 图，跳过结构断言"
        x = load_batch(paths[:4], 32, _t.device("cpu"))
        # ① 可回传
        rec = x.clone().requires_grad_(True)
        v = multiscale_perceptual(rec, x)
        assert v.requires_grad, "多尺度感知损失不可回传 ⇒ 训练不了"
        v.backward()
        assert rec.grad is not None and _t.isfinite(rec.grad).all(), "梯度异常"
        # ② 结构抹平 ⇒ 损失必须**严格变大**（比值会除零，用绝对差）
        perfect = float(multiscale_perceptual(x, x))              # 完美重建 = 0
        flat = float(multiscale_perceptual(
            x.mean(1, keepdim=True).repeat(1, 3, 1, 1), x))       # 结构被抹平
        assert flat > perfect, f"结构抹平后损失没变大（{flat:.4f} vs {perfect:.4f}）⇒ 没用上结构"
        return f"可回传 ✓ · 完美重建 {perfect:.3f} → 结构抹平 {flat:.3f}（真的看重结构）"
    check("多尺度感知损失可回传且真的看重结构", _p1_multiscale_is_cheap_and_differentiable)

    def _p1_no_silent_data_substitution():
        """⛔ 没图必须报错，**不许静默用合成数据替代**。"""
        from kp.train.vae_pretrain import train_vae
        from kp.paths import OUT
        try:
            train_vae([OUT / "__definitely_no_such_dir__"], steps=1, batch=1, size=32)
        except FileNotFoundError as e:
            assert "不静默" in str(e) or "没找到" in str(e), f"报错信息没说明缺口：{e}"
            return "无图 → 显式报错并说明缺口（不静默替代）"
        raise AssertionError("没图却没报错 ⇒ 静默用了替代数据（这是最坏的一类 bug）")
    check("无图时显式报错（⛔ 不静默用合成数据替代）", _p1_no_silent_data_substitution)

    # ---------------- 25. P1 四旋钮数据扩增 ----------------
    section("25. P1 数据扩增（四旋钮 · ⛔ 增广≠新内容）")

    def _aug_knobs_deterministic():
        """增广必须**确定性**：同一 (图, idx) 永远同一结果 ⇒ 可复现、可对照。"""
        import hashlib
        from kp.data.augment import list_images, augment_one
        from kp.character.dataset import load_image
        from kp.paths import DATA
        paths = list_images([DATA / "characters" / "kokona"])
        if not paths:
            return "⚠️ 无 kokona 图，跳过"
        base = load_image(str(paths[0]), 32)
        h = lambda t: hashlib.md5(t.numpy().tobytes()).hexdigest()[:8]  # noqa: E731
        a1, a2 = augment_one(base, 3, size=32), augment_one(base, 3, size=32)
        b = augment_one(base, 4, size=32)
        assert h(a1) == h(a2), "同一 idx 两次调用结果不同 ⇒ 不可复现"
        assert h(a1) != h(b), "不同 idx 结果相同 ⇒ idx 没起作用"
        return f"确定性 ✓（idx=3 两次一致；idx=3 vs 4 不同：{h(a1)} / {h(b)}）"
    check("增广确定性（同 idx 可复现）", _aug_knobs_deterministic)

    def _aug_actually_changes():
        """🔴 增广必须**真的改变图像**（否则等于没扩）。"""
        from kp.data.augment import list_images, augment_one
        from kp.character.dataset import load_image
        from kp.paths import DATA
        paths = list_images([DATA / "characters" / "kokona"])
        if not paths:
            return "⚠️ 无 kokona 图，跳过"
        base = load_image(str(paths[0]), 32)
        diffs = [float((augment_one(base, i, size=32) - base).abs().mean())
                 for i in range(8)]
        mean_d = sum(diffs) / len(diffs)
        assert mean_d > 1e-3, f"增广平均差异仅 {mean_d:.5f} ⇒ 等于没扩"
        assert max(diffs) > mean_d, "所有增广差异相同 ⇒ 参数没起作用"
        return f"8 个增广平均差异 {mean_d:.4f}（最大 {max(diffs):.4f}）⇒ 确实在变"
    check("增广真的改变图像（不是复制）", _aug_actually_changes)

    def _aug_rot90_is_off_by_default():
        """⛔ 90° 旋转**必须默认关** —— 角色图「上」有语义，旋转会造出倒立的人。"""
        from kp.data.augment import build_dataset
        from kp.paths import DATA
        rep = build_dataset([DATA / "characters" / "kokona"], per_image=2,
                            size=32, dry_run=True)
        assert rep["allow_rot90"] is False, "rot90 不该默认开启"
        # 显式开启时**必须**报出警告
        rep2 = build_dataset([DATA / "characters" / "kokona"], per_image=2, size=32,
                             allow_rot90=True, dry_run=True)
        assert any("倒立" in w for w in rep2["⚠️_warnings"]), \
            "开启 rot90 却没报出「污染姿态先验」的警告 ⛔ 静默危险"
        return "rot90 默认关 ✓ · 显式开启时**必报**姿态污染警告 ✓"
    check("⛔ 90° 旋转默认关 + 开启时必报警告", _aug_rot90_is_off_by_default)

    def _aug_declares_nature():
        """⛔ 报告**必须声明「这是增广不是新内容」** —— 防止拿它冒充训练集规模。"""
        from kp.data.augment import build_dataset
        from kp.paths import DATA
        rep = build_dataset([DATA / "characters" / "kokona"], per_image=2,
                            size=32, dry_run=True)
        assert "⚠️_nature" in rep, "报告缺少性质声明"
        assert "增广" in rep["⚠️_nature"] and "不是" in rep["⚠️_nature"], \
            f"性质声明没说清是增广：{rep['⚠️_nature']}"
        return f"已声明：{rep['⚠️_nature'][:34]}…"
    check("⛔ 增广数据必须声明性质（不许冒充新内容）", _aug_declares_nature)

    # ---------------- 26. 死旋钮接线（第二批）----------------
    section("26. 死旋钮接线（gate_dtype / matryoshka_tokens）")

    def _gate_dtype_is_live():
        """🔴 `CAP.gate_dtype` 必须**真接线**（此前硬编码 `torch.float32`）。"""
        import dataclasses
        import torch as _t
        from kp.capability.delta_pack import DeltaPack
        from kp.config import CAP
        import kp.capability.bus as B
        try:
            d = DeltaPack("p", 32, 64, rank=4, seed=1)
            assert d.gate.dtype == getattr(_t, str(CAP.gate_dtype)), (
                f"gate dtype {d.gate.dtype} 与 config {CAP.gate_dtype} 不符")
            # 变异性：改 config ⇒ dtype 必须跟着变
            B.CAP = dataclasses.replace(CAP, gate_dtype="float64")
            d2 = DeltaPack("p", 32, 64, rank=4, seed=1)
            assert d2.gate.dtype == _t.float64, f"改 gate_dtype=64 后实测 {d2.gate.dtype} ⇒ 仍是死的"
            return f"接线 ✓（config={CAP.gate_dtype} → {d.gate.dtype}；改 64 → {d2.gate.dtype}）"
        finally:
            B.CAP = CAP

    def _gate_dtype_rejects_bad():
        """⛔ 非浮点 dtype 必须**显式报错**（不静默回退）。"""
        import dataclasses
        from kp.capability.delta_pack import DeltaPack
        from kp.config import CAP
        import kp.capability.bus as B
        try:
            B.CAP = dataclasses.replace(CAP, gate_dtype="int64")
            try:
                DeltaPack("p", 32, 64, rank=4, seed=1)
            except ValueError as e:
                assert "不静默" in str(e), f"报错信息没说明意图：{e}"
                return "gate_dtype=int64 → 显式报错 ✓（不静默回退）"
            raise AssertionError("非浮点 gate_dtype 被静默接受 ⇒ 回到死旋钮病根")
        finally:
            B.CAP = CAP
    check("gate_dtype 真接线 + 非法值显式报错", _gate_dtype_is_live)
    check("⛔ 非浮点 gate_dtype 显式报错（防回到病根）", _gate_dtype_rejects_bad)

    def _matryoshka_reads_config():
        """🔴 `DiTCfg.matryoshka_tokens` 必须被 `sample.sample` 真正读取。"""
        import inspect
        from kp.config import DIT_S
        import kp.sample as S
        sig = inspect.signature(S.sample)
        assert sig.parameters["matryoshka"].default is None, (
            f"sample 的 matryoshka 默认值是 {sig.parameters['matryoshka'].default!r}，"
            f"应是 None（=运行时从 config 读）⇒ 写死了")
        assert tuple(DIT_S.matryoshka_tokens), "config 里 matryoshka_tokens 为空"
        return (f"默认 None（运行时读 config）✓ ｜ config 值 {tuple(DIT_S.matryoshka_tokens)}")
    check("matryoshka_tokens 真读 config（默认 None 不写死）", _matryoshka_reads_config)

    # ---------------- 27. S1 单图自重建前提（InstantCharacter 路线）----------------
    section("27. S1 单图路线前提（⭐ 不依赖多视角）")

    def _s1_latent_keeps_identity():
        """🔴 **InstantCharacter S1 路线的前提**：latent 必须装得下身份信息。

        背景（v1.17）：用户明确**多视角图绝对补齐不了**。而联网核实 arXiv 2504.12395 §3.2 原文：
            "To achieve **character consistency**, we first train with **unpaired data**,
             where the character image is incorporated as **reference guidance to reconstruct
             itself**"
        ⇒ **身份一致性由「单图自重建」建立，不需要多视角。**

        ⭐ 但这条路线有个**前提**：**VAE 的 latent 必须真的保留身份/结构信息**。
        若 latent 装不下身份，单图路线根本不成立（后面 S2/S3 都不用谈）。
        ⇒ 这条断言守的就是这个前提。
        """
        from kp.models.vae import HybridVAE
        from kp.train.vae_pretrain import list_images, load_batch, _edge
        from kp.paths import DATA, OUT
        paths = list_images([DATA / "characters" / "kokona"])
        if not paths:
            return "⚠️ 无 kokona 图，跳过"
        x = load_batch(paths, 64, torch.device("cpu"))
        # 随机初始化的对照（同一形状）
        torch.manual_seed(0)
        v0 = HybridVAE(base=16)
        v0.eval()
        with torch.no_grad():
            l1_0 = float((torch.tanh(v0.decode(v0.encode_latent(x))) - x).abs().mean())
        # 已训练的 checkpoint（若不存在则只报随机基线，并明确说明没验成）
        ck = OUT / "vae" / "final.pt"
        if not ck.exists():
            return (f"⚠️ 无训练后 ckpt（{ck.name}），只测了随机基线 L1={l1_0:.4f} "
                    f"⇒ **前提未验证**，需先训 VAE")
        d = torch.load(ck, weights_only=False)
        v = HybridVAE(base=d["base"])
        v.load_state_dict(d["state_dict"])
        v.eval()
        with torch.no_grad():
            z = v.encode_latent(x)
            rec = torch.tanh(v.decode(z))
            l1 = float((rec - x).abs().mean())
            # 边缘保留：结构信息是否留在 latent 里（比 L1 更直接对应"身份骨架"）
            ec = float((_edge(rec) * _edge(x)).sum() / (_edge(x).abs().sum() + 1e-9))
        assert l1 < l1_0, f"训练后 L1({l1:.4f}) 不比随机({l1_0:.4f}) 好 ⇒ 白训了"
        assert ec > 0.5, f"边缘保留仅 {ec:.3f} ⇒ latent 没保住结构，单图路线前提不成立"
        return (f"随机 L1 {l1_0:.4f} → 训练后 {l1:.4f}；**边缘保留 {ec:.4f}** "
                f"⇒ latent 装得下身份结构 ✓（S1 前提成立）")
    check("S1 前提：latent 保留身份/结构信息（单图路线的基础）", _s1_latent_keeps_identity)

    # ---------------- 28. 数据分级过滤（⛔ 语料不需要额外准备）----------------
    section("28. 数据分级过滤（safe / sensitive / explicit）")

    def _rating_classify():
        """分级判定必须**三级来源可靠**，且判不出时**不默认放行**。

        ⭐ 背景：用户问「敏感内容怎么办，要不要我额外准备语料」。
        答：**不需要** —— Danbooru 系数据集**自带** `rating` / `is_explicit` /
        `tag_string` 里的 `rating:*` 元标签 ⇒ **过滤在读取时做，不是下载时**。
        """
        from kp.data.rating import classify_rating
        cases = [
            ({"rating": "safe"}, "safe"),
            ({"rating": "explicit"}, "explicit"),
            ({"rating": "sensitive"}, "sensitive"),
            ({"is_explicit": True}, "explicit"),
            ({"is_explicit": False}, "safe"),
            ({"tag_string": "1girl, rating:sensitive, long_hair"}, "sensitive"),
            ({"tag_string": "1boy, rating:explicit"}, "explicit"),
            ({"tag_string": "cat, rating:safe"}, "safe"),
            ({}, "unknown"),
        ]
        wrong = [(r, classify_rating(r), w) for r, w in cases if classify_rating(r) != w]
        assert not wrong, f"分级判定错误：{wrong}"
        return f"9/9 通过（含 unknown ⇒ 不默认放行）"
    check("分级判定三级来源可靠（判不出不放行）", _rating_classify)

    def _rating_policy_default_strict():
        """🔴 默认必须是**最严**（`strict` = 只留 safe）。"""
        from kp.data.rating import POLICIES
        assert POLICIES["strict"] == {"safe"}, f"默认档不是 strict：{POLICIES['strict']}"
        assert "explicit" not in POLICIES["strict"], "strict 档竟含 explicit"
        # 更严的档必须包含更严的（集合包含关系）
        assert POLICIES["sensitive_ok"] > POLICIES["strict"], "档位包含关系反了"
        assert POLICIES["explicit_ok"] >= POLICIES["sensitive_ok"], "档位包含关系反了"
        return (f"默认 strict={sorted(POLICIES['strict'])}；"
                f"三档包含关系正确（strict ⊂ sensitive_ok ⊂ explicit_ok）")
    check("默认最严（strict 只留 safe）+ 档位包含关系正确", _rating_policy_default_strict)

    # ---------------- 29. G5 语义层解算器（可插拔 + 质量尺子）----------------
    section("29. G5 语义层解算器（⭐ txt2img+角色卡 主线的第一道阻塞）")

    def _seg_contract():
        """语义层解算器必须**可插拔 + 自报质量**（否则换模型只能靠"看着像"）。"""
        from kp.character.segmenter import (LAYER_QUALITY, Segmenter,
                                           available, register_segmenter)
        assert "color_cluster" in available(), f"基线解算器未注册：{available()}"
        q = LAYER_QUALITY(mean_iou=0.0, name_hit=0.0)
        assert "mean_iou" in q.summary() and "name_hit" in q.summary()
        return f"已注册 {available()}；质量表字段 {sorted(q.__dict__)}"
    check("语义层解算器可插拔 + 有质量尺子", _seg_contract)

    def _seg_baseline_honest():
        """🔴 **基线必须诚实自报「我不是语义层」** —— 不许假装已实现 See-through。"""
        from kp.character.segmenter import ColorClusterBaseline
        seg = ColorClusterBaseline(k=8)
        # ⭐ 关键：颜色聚类**不按 19 类命名** ⇒ name_hit 结构上就是 0
        assert seg.quality.name_hit == 0.0, \
            f"颜色聚类基线的 name_hit 竟报 {seg.quality.name_hit} ⇒ 虚报"
        assert seg.quality.mean_iou == 0.0, "基线 mean_iou 虚报"
        assert "占位" in seg.name or "baseline" in seg.name.lower(), \
            f"解算器名未标出它是基线/占位：{seg.name!r}"
        return (f"诚实自报：{seg.quality.summary()[:52]}…")
    check("⛔ 基线诚实自报「不是语义层」（不虚报）", _seg_baseline_honest)

    def _seg_predicts_on_real_image():
        """解算器必须能在**真图**上跑通（不只是接口存在）。"""
        import numpy as _np
        from PIL import Image
        from kp.character.segmenter import ColorClusterBaseline
        from kp.paths import DATA
        imgs = sorted((DATA / "characters" / "kokona").rglob("*.png"))
        if not imgs:
            return "⚠️ 无 kokona 真图，跳过"
        rgba = _np.array(Image.open(imgs[0]).convert("RGBA"))
        lab = ColorClusterBaseline(k=8).predict(rgba)
        assert lab.shape == rgba.shape[:2], f"输出形状 {lab.shape} ≠ 输入 {rgba.shape[:2]}"
        assert lab.dtype == _np.int16, f"dtype 应为 int16（-1=背景），实为 {lab.dtype}"
        assert int(lab.min()) >= -1, "背景标记应 ≥ -1"
        return f"真图 {rgba.shape[:2]} -> {lab.shape} int16，背景占比 {float((lab<0).mean()):.2f}"
    check("解算器在真图上跑通（形状/dtype 契约）", _seg_predicts_on_real_image)

    # ---------------- 30. txt2img+角色卡 端到端（主线最后一块拼图）----------------
    section("30. txt2img+角色卡 · 端到端冒烟（⛔ 只证骨架）")

    def _e2e_rolecard_chain():
        """⭐ **主线端到端**：正/背视图 → Fitter / CharaBridge → 身份 token。

        证明三件事（都是「不打架」与「秒级换角色」的核心承诺）：
          ① Fitter **前向一次**出身份 token（秒级，不训练）
          ③ **换角色 ⇒ token 真的变**（否则 Fitter 是常量，什么都没学）
          ② **gate=0 ⇒ 返回 None**（不是零向量！）⇒ 身份完全不参与
        ⛔ **不涉及**：文本塔（未蒸馏）· 主干训练（未开始）· 语义层（占位）。
        """
        from kp.character.e2e_smoke import run
        r = run()
        if "缺口" in r:
            return f"⚠️ {r['缺口']}"
        s1, s3, s2 = r["步骤"]["①Fitter"], r["步骤"]["③换角色"], r["步骤"]["②b关断不变量"]
        assert s1["秒级"], f"Fitter 不够快：{s1['耗时秒']}s"
        assert s3["真的变了"], "换角色后 token 没变 ⇒ Fitter 是常量"
        assert s2["gate=0 两个角色都返回 None"], "gate=0 未返回 None"
        assert s2["开门时两角色token确实不同"], "开门时两角色 token 相同 ⇒ 没学到"
        return (f"①Fitter {s1['耗时秒']:.3f}s→{s1['身份token形状']}｜"
                f"③换角色 Δ={s3['两角色token平均差']:.4f}｜"
                f"②gate=0⇒None ✓ 开门⇒两角色不同 ✓")
    check("端到端：视图→身份token（秒级/可换/可关断）", _e2e_rolecard_chain)

    def _e2e_charabridge_input_contract():
        """🔴 守住一个**已踩过**的坑：CharaBridge 吃**视图**、不是 Fitter 的 token。

        ⚠️ 我第一版误以为二者串联（CharaBridge(Fitter(views))），实跑才报错才发现：
        读源码才知 **`CharaBridge.forward(refs)` 吃 `[B,V,3,H,W]`**，
        自己内部编码 ⇒ **Fitter 与 CharaBridge 是并列的两条身份通路**。
        ⇒ 这条断言把这个契约钉住。
        """
        from kp.models.charabridge import CharaBridge
        from kp.character.fitter import CharacterFitter
        # ⚠️ 取【类自己的】docstring（`__doc__`），不是继承来的 `__init__` docstring
        #    —— 我第一版用 `inspect.getdoc(__init__)` 拿到的是 torch.nn.Module 的通用说明。
        cdoc = CharaBridge.__doc__ or ""
        fdoc = CharacterFitter.forward.__doc__ or ""
        assert "B, V, 3" in cdoc, f"CharaBridge 的输入约定没写在**类** docstring：{cdoc!r}"
        assert "V, 3" in fdoc, f"Fitter 的输入约定没写在自己的 forward docstring：{fdoc!r}"
        return "两者都吃 [B,V,3,H,W] 视图（并列，非串联）✓ 契约已钉"
    check("🔴 CharaBridge/Fitter 输入契约（并列非串联）", _e2e_charabridge_input_contract)

    # ---------------- 31. P1.8 tokenizer 绑定（中文进模型的唯一通路）----------------
    section("31. P1.8 · tokenizer（⭐「中文能进模型」的唯一通路）")

    def _tok_chinese_encodable():
        """🔴 **中文必须真的能被编码成 ids** —— 这是 P1.8 那条「不可逆死线」的前提。

        ⭐ 背景：设计稿说「**中文是文本塔的原生能力**」，但
        `TextTower.forward(ids)` **只吃 token ids**，
        🔴 **而全库原先 grep 不到任何 tokenizer** ⇒ **中文根本进不了模型**。
        ⇒ 本条断言守住这个链路（`kp/text/tokenizer.py` 补上了它）。

        ⚠️ **严格区分两件事**（本项目反复吃的坑：指标口径不对，结论必错）：
            本条**只证明「能被编码」** ⛔ **不证明「模型懂中文」** ——
            塔**尚未蒸馏**，中文能力要等蒸馏后**单独测**。
        """
        from kp.text.tokenizer import load_tokenizer, han_coverage, HAN_COVERAGE_GATE
        try:
            tok = load_tokenizer()
        except Exception as e:                              # noqa: BLE001
            return f"⚠️ 拿不到 tokenizer（{type(e).__name__}），本条跳过"
        tot = ok = 0
        for s in ("一名黑色长发的少女站在樱花树下，穿着校服。",
                  "赛博朋克风格的城市夜景，霓虹灯牌，机械义肢。",
                  "心夏北极星 沫夏澄影"):
            t, o, _ = han_coverage(tok, s)
            tot += t
            ok += o
        cov = ok / max(tot, 1)
        assert cov >= HAN_COVERAGE_GATE, (
            f"汉字覆盖率仅 {cov:.3f} < 门线 {HAN_COVERAGE_GATE} ⇒ 中文配比/词表有问题")
        return f"汉字 {ok}/{tot} 覆盖率 {cov:.1%} ≥ 门线 {HAN_COVERAGE_GATE:.0%} ⛔(仅证'能编码')"
    check("🔴 中文可编码成 ids（P1.8 的前提）", _tok_chinese_encodable)

    def _tok_vocab_fits_embedding():
        """🔴 词表**必须能装进嵌入表**（防越界）；顺带报出白占的行数。

        ⭐ **三种口径必须分清**（实测踩过）：
            ① `tokenizer.vocab_size` = 151643 —— **不含** added_tokens
            ② `len(tok)`             = 151669 —— 含 26 个 added
            ③ `max(token id) + 1`    = 151669 —— 真正能安全索引的上界
        ⇒ 嵌入表**按 ③ 建才安全**；config 写 151936（比③大 267）**不会越界**，
          但**白占 267×768 = 0.2M 参数**（仅浪费，不报错 ⇒ 又是一个「静默」项）。
        """
        from kp.text.tokenizer import load_tokenizer
        from kp.models.text_tower import TextTowerCfg
        try:
            tok = load_tokenizer()
        except Exception:                                 # noqa: BLE001
            return "⚠️ 拿不到 tokenizer，跳过"
        need = max(tok.get_vocab().values()) + 1
        cfg_v = TextTowerCfg().vocab_size
        assert cfg_v >= need, (
            f"嵌入表 {cfg_v} < 实际需要的 {need} ⇒ **token id 会越界**（会崩）")
        waste = cfg_v - need
        extra = f"（白占 {waste} 行 ≈ {waste * TextTowerCfg().dim / 1e6:.1f}M 参数）" if waste else "（零浪费）"
        return f"需 {need} / 配置 {cfg_v} ⇒ 能容纳 ✓ {extra}"
    check("🔴 词表装得进嵌入表（防越界）", _tok_vocab_fits_embedding)

    def _tok_local_cache_first():
        """⛔ tokenizer 必须**本地优先** —— 慢网下远端会返回 HTML/限流页被当成 JSON。"""
        from pathlib import Path
        from kp.text.tokenizer import LOCAL_DIR, QWEN3
        assert LOCAL_DIR.is_dir(), f"本地 tokenizer 目录不存在：{LOCAL_DIR}"
        need = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
        missing = [f for f in need if not (LOCAL_DIR / f).exists()]
        assert not missing, (
            f"本地缓存缺 {missing} ⛔ **只有 tokenizer_config.json 不够**"
            f"（那是配置不是词表）⇒ 踩过：HF 缓存正好只有它 ⇒ JSONDecodeError")
        sz = sum((LOCAL_DIR / f).stat().st_size for f in need) / 1e6
        return f"本地齐备 {len(need)} 个文件 / {sz:.1f}MB（源 {QWEN3}）⇒ 离线可用"
    check("⛔ tokenizer 本地缓存齐备（离线可用）", _tok_local_cache_first)

    # ---------------- 32. P1.8 蒸馏通路（教师离线一次）----------------
    section("32. P1.8 · 文本塔蒸馏通路（⛔ 教师未就位，只证数据侧）")

    def _distill_encode_path():
        """蒸馏的**第一步**必须能跑：caption jsonl → ids（⛔ 不需教师权重）。"""
        import tempfile
        from pathlib import Path as _P
        from kp.text.distill import encode_captions
        d = _P(tempfile.gettempdir()) / "kp_distill_smoke"
        d.mkdir(parents=True, exist_ok=True)
        src = d / "caps.jsonl"
        src.write_text(
            '{"caption": "一名黑色长发的少女站在樱花树下。", "lang": "zh"}\n'
            '{"caption": "a girl under cherry blossoms", "lang": "en"}\n',
            encoding="utf-8")
        out = d / "ids.pt"
        r = encode_captions(str(src), str(out), max_len=64)
        assert r["count"] == 2, f"应编码 2 条，实得 {r['count']}"
        assert r["skipped"] == 0, f"不该有跳过，实得 {r['skipped']}"
        import torch
        obj = torch.load(out, weights_only=False)
        assert obj["rows"][0]["lang"] == "zh", "语言标记丢了"
        assert len(obj["rows"][0]["ids"]) > 0, "ids 为空"
        return (f"caption→ids ✓ {r['count']} 条，平均 {r['平均长度']} token，lang 保留")
    check("蒸馏①：caption→ids（⛔ 不需教师，可离线跑）", _distill_encode_path)

    def _distill_multilayer_config():
        """🔴 设计稿 §4.1 明写「**多层特征聚合**，不是只用最后一层」。

        ⚠️ 理由（设计稿原文）：「自回归 LLM 的末层是为 next-token 优化的，
        对图像生成并非最优」⇒ 必须取**浅层 + 多层**。
        ⇒ 这条断言防止有人「图省事只取最后一层」。
        """
        from kp.text.distill import TEACHER_LAYERS
        assert len(TEACHER_LAYERS) >= 3, f"只取 {len(TEACHER_LAYERS)} 层，不算多层聚合"
        assert TEACHER_LAYERS != tuple(sorted(TEACHER_LAYERS)[-1:]), \
            "只取最后一层 ⇒ 违背设计稿 §4.1"
        assert min(TEACHER_LAYERS) < max(TEACHER_LAYERS),             "应同时含浅层与深层（浅层给结构，深层给语义）"
        return f"多层 = {list(TEACHER_LAYERS)}（含浅层 {min(TEACHER_LAYERS)} + 深层 {max(TEACHER_LAYERS)}）✓"
    check("🔴 教师特征取多层（非仅最后一层）", _distill_multilayer_config)

    def _distill_three_things_separated():
        """⛔ 断言「能编码 / 懂中文 / 主干用中文」是**三件独立的事**。

        ⭐ 背景（防止把一件事的结论当另一件事的证据）：
          ① 中文**能被编码**   —— ✅ 自检 §31 已证（汉字覆盖 100%）
          ② 塔**懂中文**       —— ⛔ **未证**（塔尚未蒸馏）
          ③ 主干**用中文**     —— ⛔ 未开始
        ⇒ 任何「KP 支持中文」的说法，**必须指明是这三件里的哪一件**。
        """
        from kp.text.tokenizer import load_tokenizer
        try:
            load_tokenizer()
        except Exception:                                 # noqa: BLE001
            return "⚠️ 无 tokenizer，跳过"
        return ("严格区分：①能编码 ✅(§31) / ②塔懂中文 ⛔未证(未蒸馏) / "
                "③主干用中文 ⛔未开始 ⇒ 说'支持中文'必须指明是哪一件")
    check("⛔ 「支持中文」的三件事必须分开说", _distill_three_things_separated)

    # ---------------- 33. GPU 支持（白天满载）----------------
    section("33. GPU 支持（⭐ 白天放开满载的必查项）")

    def _gpu_encode_matches_cpu():
        """🔴 `encode_views` 的 GPU 路径必须与 CPU **判据层面一致**。

        ⚠️ **判据用「相关系数」而非「逐位相等」** —— 我第一版用 `max|Δ| < 1e-3` 当门槛，
        实测 CPU/GPU 差 1.82e-03 **误报红**，但**相关系数 = 1.00000000**。
        根因：fp32 卷积在 CPU/GPU 上的浮点累加顺序不同 ⇒ 末位差异是**正常的**，
        ⛔ **不该用逐位相等当判据**（那会把「设备差异」误判成「实现 bug」）。
        """
        import torch as _t
        from kp.latent.real_separation import load_real_views, encode_views
        if not _t.cuda.is_available():
            return "⚠️ 无 GPU，跳过"
        v = load_real_views(size=64, views_per_source=2, max_views=2)
        _, b1 = encode_views(v, device="cpu")
        _, b2 = encode_views(v, device="cuda")
        c = float(_t.corrcoef(_t.stack([b1.z.flatten(), b2.z.flatten()]))[0, 1])
        assert c > 0.9999, f"CPU/GPU 相关性仅 {c:.6f} ⇒ 不是浮点噪声，是实现不一致"
        return f"相关系数 {c:.8f}（>0.9999）✔ 判据层面一致"
    check("🔴 GPU 编码与 CPU 判据一致（用相关不用逐位）", _gpu_encode_matches_cpu)

    def _gpu_latent_returns_cpu():
        """⛔ latent 必须**搬回 CPU** —— 下游统计/判据都在 CPU 上做。"""
        import torch as _t
        from kp.latent.real_separation import load_real_views, encode_views
        if not _t.cuda.is_available():
            return "⚠️ 无 GPU，跳过"
        v = load_real_views(size=64, views_per_source=2, max_views=2)
        vae, b = encode_views(v, device="cuda")
        assert b.z.device.type == "cpu", f"latent 在 {b.z.device}，应在 CPU"
        assert next(vae.parameters()).device.type == "cpu", "vae 未搬回 CPU"
        return "latent 与 vae 均已搬回 CPU ✔"
    check("⛔ GPU 跑完把 latent/vae 搬回 CPU", _gpu_latent_returns_cpu)

    def _gpu_run_g2_accepts_device():
        """`run_real_g2` 必须接受 `device` 并透传（原先**只能跑 CPU**）。"""
        import inspect
        from kp.latent.real_separation import run_real_g2
        sig = inspect.signature(run_real_g2)
        assert "device" in sig.parameters, "run_real_g2 没有 device 参数 ⇒ GPU 用不了"
        assert sig.parameters["device"].default is None, \
            "device 默认应为 None（=CPU），不能默认 cuda（会变静默行为）"
        return f"run_real_g2(device=...) ✅ 默认 {sig.parameters['device'].default}（=CPU）"
    check("run_real_g2 支持 device 且默认 CPU", _gpu_run_g2_accepts_device)

    # ---------------- 34. M3 归因（🔴 最关键的实验结论）----------------
    section("34. M3 冗余归因（DC-AE 对照 · 🔴 结论已固化）")

    def _m3_dcae_available():
        """DC-AE（Sana 的 32× VAE）必须**真的能用** —— 它是 M3 归因的对照基准。

        ⚠️ **踩过的坑（别重做）**：`from_pretrained` 会 **键名重叠 0**（checkpoint 是
        Sana 官方 `stages/op_list` 命名，diffusers 的 `AutoencoderDC` 是 `down_blocks`）
        ⇒ 全部 meta ⇒ 崩。**正解是 `from_single_file(..., local_files_only=True)`**。
        """
        try:
            from kp.probe.m3_attribution import load_dcae
            import torch
            m = load_dcae("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:                              # noqa: BLE001
            return f"⚠️ DC-AE 不可用（{type(e).__name__}），M3 归因实验跳过"
        n = sum(p.numel() for p in m.parameters())
        assert n > 3e8, f"DC-AE 参数量仅 {n/1e6:.1f}M（应 312M）"
        return f"DC-AE 可用 ✅ {n/1e6:.1f}M"
    check("DC-AE 权重可用（M3 归因的对照基准）", _m3_dcae_available)

    def _m3_attribution_corrected():
        """🔴🔴 **M3 归因（修正版）**：训练**把冗余做满**，⇒ 必须走结构路线。

        ⛔ **2026-10-04 上午的判决已作废**（又一次「口径不全就下结论」）：
            曾报「我们 0.988 vs DC-AE 0.4375 ⇒ 是我们的问题」——
            ⛔ 但那个 0.988 是**随机初始化的 encoder** 测的
            （`real_separation.encode_views` 用 `HybridVAE(base=16)` 冷启动，从不训练），
            而 DC-AE 是**训练充分的** ⇒ **混淆了「架构」与「训练程度」**。

        ✅ **修正后的对照**（同一批真图）：
            我们 · 随机初始化      **0.2876**
            我们 · 训练后(11张)    **0.9997**
            我们 · 训练后(176张)   **0.9997**
            DC-AE · 训练充分       0.4375

        🔴 **修正后判决**：
            不是「训练越强越糟」，而是「**训练把冗余做满了**」
            ⇒ 归因是「**重建目标会主动抹平通道分工**」
            ⇒ ⭐ **推论：纯靠监督 penalty（修法①）很可能无效**（重建压力会把它按回去，
              与 DA-VAE「纯重建损失下细节通道吸收残差」方向一致）
            ⇒ **修法②（结构路线）从「待验证」上调为「首选」**。
        """
        import json
        from kp.paths import OUT
        p = OUT / "m3_attribution.json"
        if not p.exists():
            return "⚠️ 尚无归因实验结果"
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = d.get("修正后对照") or {}
        rnd = rows.get("我们_随机初始化")
        trained = [v for k, v in rows.items()
                   if k.startswith("我们_训练后") and isinstance(v, float)]
        assert rnd is not None and trained, f"结果不完整：{rows}"
        # 🔴 核心不变量：**训练后冗余必须显著高于随机**（证明"训练做满冗余"）
        assert min(trained) > rnd + 0.3, (
            f"训练后 {min(trained):.3f} 未显著高于随机 {rnd:.3f} ⇒ "
            f"「训练把冗余做满」这个结论不成立，归因需再查")
        return (f"随机 {rnd:.3f} → 训练后 {min(trained):.3f}（+{min(trained)-rnd:.2f}）"
                f" ⇒ 🔴 重建抹平分工 ⇒ 修法②（结构路线）为首选")
    check("🔴 M3 归因（修正版）：训练做满冗余 → 结构路线", _m3_attribution_corrected)


    # ---------------- 35. M3 的因果链（哪一环管哪一环）----------------
    section("35. M3 因果链（⭐ 别把 VAE 的问题派给主干的修法）")

    def _m3_fix_scope():
        """🔴 **修法② 作用在【主干入口】，修不了【VAE latent 里的冗余】** —— 两者互补。

        ⚠️ **这条极易搞混**（我自己也混过一轮）：
        · `cross_r2 = 0.9997` 测的是 **VAE 产出的 40ch latent**（信息层面）
        · `split_patch_embed` 改的是 **主干的 patch_embed**（`kp/models/dit.py`）
        · ⛔ **`kp/models/vae.py` 里没有 `patch_embed`**（实测 0 处）
          ⇒ 修法② **改不到** VAE 内部的冗余

        ⇒ **因果链（三段，别再混）**：
            ① VAE 训练把信息塞满两块  ⇒  latent **内部**冗余（信息层面，已实测 1.00）
            ② 共享 patch_embed 会顺手抄另一块 ⇒ **污染路径**存在
            ③ 修法② 切断 ②（结构保证）⇒ **管不了 ①**

        ⇒ **正确组合**：VAE 侧要改（重建目标本身）+ 主干侧用修法②（防抄）。
        ⛔ 只做 ② 不改 ① ⇒ latent 仍是冗余的（只是主干不容易抄）。
        """
        from pathlib import Path
        vae_src = (Path(__file__).parent / "models" / "vae.py").read_text(encoding="utf-8")
        assert "patch_embed" not in vae_src, \
            "vae.py 里出现 patch_embed ⇒ 本条前提变了，**因果链需重写**（别沿用旧结论）"
        from kp.config import DIT_S
        assert hasattr(DIT_S, "split_patch_embed"), "config 缺 split_patch_embed"
        return ("✅ 边界清晰：cross_r2 测【VAE latent】；修法② 改【主干 patch_embed】；"
                "vae.py 无 patch_embed ⇒ 修法② 管不到 latent 内部 ⇒ **两者互补，不是替代**")
    check("🔴 M3 因果链：修法② 管不到 VAE latent 的冗余", _m3_fix_scope)

    # ---------------- 36. 判据有效性守卫（防止拿失效的量下结论）----------------
    section("36. 🔴 判据有效性守卫（_r2 在小样本下**饱和**）")

    def _r2_calibration():
        """🔴 `_r2` 必须在**已知答案**上校准 —— 小样本下它会**饱和到 1.0**。

        ⚠️ **实测踩过的坑（2026-10-04）**：用 `base=8 / 32² / N=8` 训小 VAE 做 M3 预研，
        三组（无约束 / penalty / 分块重建）**全报 1.0000**，一度以为"结构解法也无效"。
        ⛔ **真因是判据在该规模下失效**：
            `_r2(前半, 全零后半)` 竟报 **1.0000** —— 而它**本该是 0**（后半全是常数）。
        ⇒ 「目标含大量常数维」时，ridge 回归仍能靠截距+噪声拟合出高 R²。
        ⇒ **结论：小样本下 `cross_r2` 不可用**；本条守卫已知答案，确保它仍在有效区间。
        """
        import torch as _t
        from kp.latent.real_separation import _r2
        g = _t.Generator().manual_seed(0)
        # ① 独立 ⇒ R² 应接近 0
        a = _t.randn(16, 8, 16, 16, generator=g)
        b = _t.randn(16, 32, 16, 16, generator=g)
        r_indep = _r2(a, b)
        assert r_indep < 0.35, f"独立数据竟报 R²={r_indep:.3f} ⇒ 判据已失效"
        # ② 完全可预测 ⇒ R² 应接近 1
        r_copy = _r2(a, a.repeat(1, 4, 1, 1))
        assert r_copy > 0.9, f"完全可预测竟报 R²={r_copy:.3f} ⇒ 判据已失效"
        return f"独立 {r_indep:.3f}（应≈0）/ 可预测 {r_copy:.3f}（应≈1）⇒ 判据有效"
    check("🔴 _r2 判据在有效区间（已知答案校准）", _r2_calibration)

    def _m3_vae_fix_result_is_unusable():
        """⛔ 记录：M3 VAE 侧预研的三个数**全部无效**（判据在该规模饱和）。

        ⚠️ 这条不是"守卫代码"，是**留档一个已知的无效结论**，
        防止以后有人从 `out/m3_vae_fix.json` 里读出"A/B/C 三者都是 1.0 ⇒ 结构解法无效"这个
        **错误推论**。真正的结论是：**那个实验的判据本身失效**（见 `_r2_calibration`）。
        """
        import json
        from kp.paths import OUT
        p = OUT / "m3_vae_fix.json"
        if not p.exists():
            return "⚠️ 尚无 m3_vae_fix.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        r = d.get("结果") or {}
        all_one = all(abs(float(v) - 1.0) < 0.02 for v in r.values() if isinstance(v, float))
        if all_one:
            return ("⛔ **这三个数无效**（全 1.0 是判据饱和，非真实结果）"
                    "⇒ **不可**推出「结构解法无效」；重做需更大规模或换判据")
        return f"结果 {r}（非全 1.0，可用）"
    check("⛔ M3 预研无效结论留档（三组全 1.0 是判据饱和）", _m3_vae_fix_result_is_unusable)

    # ---------------- 汇总 ----------------
    return _summary()


def _summary() -> int:
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n = len(RESULTS)
    print("\n" + "=" * 68)
    print(f"结果：{n_pass}/{n} 通过")
    failed = [r for r in RESULTS if not r[1]]
    if failed:
        print("失败项：")
        for name, _, msg in failed:
            print(f"  ❌ {name} — {msg}")
    else:
        print("全部通过 ✅")
    print("=" * 68)
    return 0 if not failed else 1


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台默认 GBK ⇒ 打印 emoji（✅/❌）会抛
    #    `UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'`，
    #    表现为「自检直接 exit 1、报告只打了一半」——**看起来像自检挂了，其实是编码**。
    #    交接文档 §8 的命令原样贴进 PowerShell 就会命中这个假故障，故在此兜住。
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    # ⚠️ 某一节的 import / 语法错误不应让整份报告消失 —— 兜底也要打出汇总
    try:
        code = main()
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"\n⚠️ 自检在某一节**中断**：{type(e).__name__}: {e}")
        code = _summary() or 1
    sys.exit(code)
