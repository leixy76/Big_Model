# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name_or_path = r"D:\Program Projects\Python Projects\DB-GPT\models\Qwen2-0.5B"

from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True).cuda().eval()
model