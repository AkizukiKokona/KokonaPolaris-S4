import torch, time, json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

P = 'models/Qwen3.5-4B-Base'
tok = AutoTokenizer.from_pretrained(P)
print('[1] tokenizer class:', type(tok).__name__)
ids = tok.get_vocab().values()
print(f'[1] vocab_size={tok.vocab_size}  len={len(tok)}  max_id+1={max(ids)+1}  added={len(getattr(tok,"added_tokens_decoder",{}))}')
print(f'[1] 中文测试: {tok.encode("夕阳海滩少女")}')
print()

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(P, quantization_config=bnb,
                                         device_map='cuda', trust_remote_code=True)
print(f'[2] loaded in {time.time()-t0:.0f}s  '
      f'VRAM={torch.cuda.memory_allocated()/2**30:.2f}GB')
print(f'[2] layers={m.config.num_hidden_layers} hidden={m.config.hidden_size}')
print()

m.eval()
# 试 few-shot 标签→中文
prompt = ("Translate Danbooru tags to natural Chinese.\n"
          "1girl, beach, sunset -> 一个少女，海滩，夕阳\n"
          "2girls, school uniform, classroom -> 两个女孩，校服，教室\n"
          "1girl, long hair, smile, cherry blossoms ->")
enc = tok(prompt, return_tensors='pt').to('cuda')
with torch.no_grad():
    out = m.generate(**enc, max_new_tokens=40, do_sample=False,
                     pad_token_id=tok.eos_token_id)
txt = tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
print('[3] few-shot 翻译测试 ->', repr(txt[:120]))
print()

# 多层 hidden states 能否取到
with torch.no_grad():
    o = m(tok('1girl, beach', return_tensors='pt').to('cuda').input_ids,
          output_hidden_states=True)
print(f'[4] hidden_states 层数 = {len(o.hidden_states)}  '
      f'每层 shape = {tuple(o.hidden_states[0].shape)}')
