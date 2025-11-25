from vllm import LLM, SamplingParams

# 사용하신 모델 경로와 설정을 그대로 사용하여 로드 테스트
try:
    llm = LLM(
        model="/home/isdslab/sangmin/Unify-Post-Training/qwen-sft",
        trust_remote_code=True,
        gpu_memory_utilization=0.2, # 설정과 동일하게
        tensor_parallel_size=1
    )
    print("vLLM Load Success")
except Exception as e:
    print(f"vLLM Load Failed: {e}")