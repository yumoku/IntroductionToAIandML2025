import os
import string
from huggingface_hub import login
from dotenv import load_dotenv
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from torch.cuda import device
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)
login(token=os.getenv("HAPPY_FACE_KEY"), new_session=False)

model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Part 1, question 2
# print(f"Model {model_id} has {tokenizer.vocab_size} Tokens avaliable to use. ")


# Part 1, question 2
# def find_english_tokens(token_content):
#     flag = True
#     for char in token_string:
#         if char not in string.ascii_letters:
#             flag = False
#
#     return flag
#
# for token_id in range(tokenizer.vocab_size):
#     token_string = tokenizer.decode(token_id)
#     flg = find_english_tokens(token_string)
#     if flg == True:
#         print("Token ID ", token_id, " is：", tokenizer.decode(token_id))

# Part 1, question 3
# text = "作業一"
# tokens = tokenizer.encode(text,add_special_tokens=False)
# print(text ,"->", tokens)

# Part 1, question 4
# tokens_with_length = [] #存每個 token 的 ID、對應字串與其長度
#
# # 將每個 token 的 ID、對應字串與其長度加入 tokens_with_length
# for token_id in range(tokenizer.vocab_size):
#     token = tokenizer.decode(token_id)
#     tokens_with_length.append((token_id, token, len(token)))
#
# # # 根據 token 的長度從長到短排序
# tokens_with_length.sort(key=lambda x: x[2], reverse=True)
# longest_token = tokens_with_length[0]
# print(f"The longest token pair is: token id = {longest_token[0]}, token length = {longest_token[2]}")

# Part 1, question 5
# prompt = "阿姆斯特朗旋風迴旋加速噴氣式阿姆斯特朗砲"
# print("輸入的 prompt 是:", prompt)
#
# # model 不能直接輸入文字，model 只能輸入以 PyTorch tensor 格式儲存的 token IDs
# # 把要輸入 prompt 轉成 model 可以處理的格式
# input_ids = tokenizer.encode(prompt, return_tensors="pt") # return_tensors="pt" 表示回傳 PyTorch tensor 格式
# print("這是 model 可以讀的輸入：",input_ids)
#
# # model 以 input_ids (根據 prompt 產生) 作為輸入，產生 outputs，
# outputs = model(input_ids)
#
# # outputs 裡面包含了大量的資訊
# # 我們在往後的課程還會看到 outputs 中還有甚麼
# # 在這裡我們只需要 "根據輸入的 prompt ，下一個 token 的機率分布" (也就是每一個 token 接在 prompt 之後的機率)
#
# # outputs.logits 是模型對輸入每個位置、每個 token 的信心分數（還沒轉成機率）
# # outputs.logits shape: (batch_size, sequence_length, vocab_size)
# last_logits = outputs.logits[:, -1, :] #得到一個 token 接在 prompt 後面的信心分數 (至於為什麼是這樣寫，留給各位同學自己研究)
# probabilities = torch.softmax(last_logits, dim=-1) #softmax 可以把原始信心分數轉換成 0~1 之間的機率值
#
# # 印出機率最高的前 top_k 名 token
# top_k = 1
# top_p, top_indices = torch.topk(probabilities, top_k)
# token_id = top_indices[0][0].item() # 取得第 i 名的 token ID
# probability = top_p[0][0].item() # 對應的機率
# token_str = tokenizer.decode(token_id) # 將 token ID 解碼成文字
# print(f"The next possible output is: '{token_str}', with the probability of: {probability:.4f}")

# Part 1, question 6
# prompts = ["皮卡丘源自於哪個動畫作品?", "Which anime is Pikachu derived from?"]
#
# for prompt in prompts:
#     print("Current prompt is:", prompt)
#
#     messages = [
#         {"role": "system", "content": "You are a smart agent"}, #在 system prompt 中告訴 AI 他的名字 (跟前一段程式唯一不同的地方)
#         {"role": "user", "content": prompt},
#     ]
#     print("Now the messages are:", messages)
#
#
#     input_ids = tokenizer.apply_chat_template(  #不只加上Chat Template，順便幫你 encode 了
#         messages,
#        add_generation_prompt=True,
#         return_tensors="pt"
#     )
#
#     print("tokenizer.apply_chat_template outputs are：\n", input_ids)
#     print("===============================================\n")
#     print("用 tokenizer.decode contents are：\n", tokenizer.decode(input_ids[0]))
#     print("===============================================\n")
#
#     outputs = model.generate(
#         input_ids,
#         max_length=100,
#         do_sample=True,
#         top_k=3,
#         pad_token_id=tokenizer.eos_token_id,
#         attention_mask=torch.ones_like(input_ids)
#     )
#
#     # Converted the generated token ids back to words
#     generated_text = tokenizer.decode(outputs[0])
#
#     print("Generated answers are：\n", generated_text)

# Part 1, question 7
# prompt = "皮卡丘源自於哪個動畫作品?"
# print("Current prompt is:", prompt)
#
# messages = [
#     {"role": "system", "content": "You can only answer: I don't know"}, #在Adding
#     {"role": "user", "content": prompt},
# ]
# print("Now the messages are:", messages)
#
#
# input_ids = tokenizer.apply_chat_template(  #不只加上Chat Template，順便幫你 encode 了
#     messages,
#    add_generation_prompt=True,
#     return_tensors="pt"
# )
#
# print("tokenizer.apply_chat_template outputs are：\n", input_ids)
# print("===============================================\n")
# print("用 tokenizer.decode contents are：\n", tokenizer.decode(input_ids[0]))
# print("===============================================\n")
#
# outputs = model.generate(
#     input_ids,
#     max_length=100,
#     do_sample=True,
#     top_k=3,
#     pad_token_id=tokenizer.eos_token_id,
#     attention_mask=torch.ones_like(input_ids)
# )
#
# # Converted the generated token ids back to words
# generated_text = tokenizer.decode(outputs[0])
#
# print("Generated answers are：\n", generated_text)

# Part 1, question 8
prompt = "皮卡丘源自於哪個動畫作品?"
print("Current prompt is:", prompt)

messages = [
    {"role": "system", "content": "Answer in English only"}, #在Adding
    {"role": "user", "content": prompt},
]
print("Now the messages are:", messages)


input_ids = tokenizer.apply_chat_template(  #不只加上Chat Template，順便幫你 encode 了
    messages,
   add_generation_prompt=True,
    return_tensors="pt"
)

print("tokenizer.apply_chat_template outputs are：\n", input_ids)
print("===============================================\n")
print("用 tokenizer.decode contents are：\n", tokenizer.decode(input_ids[0]))
print("===============================================\n")

outputs = model.generate(
    input_ids,
    max_length=100,
    do_sample=True,
    top_k=3,
    pad_token_id=tokenizer.eos_token_id,
    attention_mask=torch.ones_like(input_ids)
)

# Converted the generated token ids back to words
generated_text = tokenizer.decode(outputs[0])

print("Generated answers are：\n", generated_text)