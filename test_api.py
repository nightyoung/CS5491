from llm_api import generate_new_heuristic
print('Testing MiniMax API...')
result, code = generate_new_heuristic('Say hello in 3 words')
print('Result type:', type(result))
print('Result content:', result[:200] if result else 'None')
