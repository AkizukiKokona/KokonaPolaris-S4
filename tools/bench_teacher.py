"""⭐ 测教师在各种配置下的 generate 速度（找最快的那一档）"""
import sys, time, torch
sys.path.insert(0, '.')
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
P = 'models/Qwen3.5-4B-Base'
tok = AutoTokenizer.from_pretrained(P)
tok.padding_side = 'right'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

TAGS = ['1girl, long hair, blue eyes, school uniform, classroom, smile']
prompts = ['Translate Danbooru tags to natural Chinese.\n'
           '1girl, beach, sunset -> 一个少女，海滩，夕阳\n'
           '2girls, school uniform, classroom -> 两个女孩，校服，教室\n'
           '1girl, long hair, smile, cherry blossoms ->' + t for t in TAGS]

MODE = sys.argv[1] if len(sys.argv) > 1 else '4bit'
B = int(sys.argv[2]) if len(sys.argv) > 2 else 32

if MODE == '4bit':
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(P, quantization_config=qc,
                                             device_map='cuda')
elif MODE == 'bf16':
    # ⭐ 4B bf16 = 8.8GB > 8GB ⇒ 只能 CPU offload 部分层
    m = AutoModelForCausalLM.from_pretrained(
        P, dtype=torch.bfloat16, device_map='auto',
        max_memory={0: '6.5GiB', 'cpu': '20GiB'})
else:
    raise SystemExit('mode?')

m.eval()
print('[%s] loaded VRAM=%.2fGB' % (MODE,
      torch.cuda.memory_allocated() / 2**30), flush=True)
enc = tok([prompts[0]] * B, return_tensors='pt', padding=True,
          truncation=True, max_length=384).to('cuda')
for _ in range(2):          # warmup（含 triton 编译）
    with torch.no_grad():
        m.generate(**enc, max_new_tokens=72, do_sample=False,
                   pad_token_id=tok.pad_token_id)
torch.cuda.synchronize()
t = time.time()
with torch.no_grad():
    m.generate(**enc, max_new_tokens=72, do_sample=False,
               pad_token_id=tok.pad_token_id)
torch.cuda.synchronize()
d = time.time() - t
print('[%s batch=%d] %.1fs  => %.2f 条/秒' % (MODE, B, d, B / d), flush=True)
