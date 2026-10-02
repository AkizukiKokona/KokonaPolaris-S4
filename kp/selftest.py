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
"""
from __future__ import annotations

import sys
import traceback
from typing import Callable, List, Tuple

import torch

RESULTS: List[Tuple[str, bool, str]] = []


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
    from kp.capability import EraseOperator, EraseLedger, kl_test, erasure_roundtrip

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
        import tempfile, os
        p = os.path.join(tempfile.gettempdir(), "_kp_card_selftest.pt")
        card.save(p)
        back = CharacterCard.load(p)
        assert back.name == card.name and back.num_tokens == card.num_tokens
        assert torch.equal(back.identity_token, card.identity_token)
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
                          quant_fp8_ste, relative_error, precision_table)

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
        return "量化改变输出；关闭后恢复 bit-exact"
    check("GatedLinear 量化开关（关后 bit-exact）", _gl_quant)

    # ---------------- 10. 能力包落盘 / 加载 ----------------
    section("10. 能力包落盘 / 加载往返（四接口之一）")
    from kp.capability import save_adapter, load_adapter
    import tempfile
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
        path = _os.path.join(tempfile.gettempdir(), "_kp_adapter_selftest.pt")
        n = save_adapter(gl, path)
        gl2 = GatedLinear(w)
        mounted = load_adapter(gl2, path)
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
        path = _os.path.join(tempfile.gettempdir(), "_kp_adapter_dit.pt")
        n = save_adapter(m1, path)
        mounted = load_adapter(m2, path)
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
        # ② 显式打开身份门（adaLN 输出的第 9 段 g3）后，注入才生效
        d = m.cfg.dim
        with torch.no_grad():
            for blk in m.blocks:
                if blk.identity_cross is not None:
                    blk.adaLN[-1].bias[8 * d:9 * d].fill_(0.5)
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
    # ⚠️ 某一节的 import / 语法错误不应让整份报告消失 —— 兜底也要打出汇总
    try:
        code = main()
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"\n⚠️ 自检在某一节**中断**：{type(e).__name__}: {e}")
        code = _summary() or 1
    sys.exit(code)
