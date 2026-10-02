"""KP 骨架自检 —— 纯 CPU、夜间安全、无外部依赖。

运行：
    python -m kp.selftest          （在 D:/model 下）
    python kp/selftest.py

逐项验证设计稿里的**可验收不变量**：
    1. latent 打包/解包 往返一致（fp32 / bf16）
    2. 通道分离监督件（split/join/swap/MI 惩罚）
    3. ★ 门控全 0 ⇒ 与裸模型 **bit-exact**
    4. ★ Δ-Pack 谱检查：子空间初始化【合格】/ 正交扰动【不合格】
    5. ★ 擦除 `E⁻¹∘E` 可逆（KL ≈ 0）
    6. ∥-Pack 零初始化 + 短路
    7. 主干前向形状 + 挂包前后 bit-exact
    8. VAE 32× 编解码形状
    9. CharacterFitter 输出身份 token 形状
   10. Rectified Flow + Matryoshka 采样可跑
   11. NVFP4 模拟量化：与参考实现对拍 + STE 可微 + 量化开关
   12. 能力包 落盘/加载 往返（save_adapter → load_adapter）
   13. QAD：冻结主干 + 只训 Δ-Pack（loss 下降且梯度只进包）
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

    # ---------------- 汇总 ----------------
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
    sys.exit(main())
