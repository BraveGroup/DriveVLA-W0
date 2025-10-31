import numpy as np
from transformers import AutoProcessor
import pickle
from tqdm import tqdm

def split_action_into_subsegments(action, T):
    """
    将action矩阵按长度T分解成连续的子段，每个子段的形状是 (T, 7)
    滑动窗口为1
    """
    N = len(action)
    subsegments = []
    
    # 滑动窗口，生成子段
    for start in range(N - T + 1):
        subsegment = action[start:start+T]  # 取连续的T行
        subsegments.append(subsegment)
    
    return np.array(subsegments)


# Load the tokenizer from the Hugging Face hub
tokenizer = AutoProcessor.from_pretrained("/share/project/yuqi.wang/OmniSim/pretrain/fast", trust_remote_code=True)

# Tokenize & decode action chunks (we use dummy data here)
# action_data = np.random.rand(256, 50, 7)    # one batch of action chunks
# tokens = tokenizer(action_data)              # tokens = list[int]
# decoded_actions = tokenizer.decode(tokens)

# calvin_pickle = "/share/project/yuqi.wang/datasets/processed_data/meta/calvin_gripper.pkl"
# with open(calvin_pickle, 'rb') as f:
#     data = pickle.load(f)

libero_pickle = "/share/project/yuqi.wang/datasets/processed_data/meta/libero_raw.pkl"
with open(libero_pickle, 'rb') as f:
    data = pickle.load(f)

all_subsegments = []

T = 5

for value in tqdm(data):
    action = value["action"]
    subsegments = split_action_into_subsegments(action, T)
    all_subsegments.append(subsegments)

all_subsegments = np.concatenate(all_subsegments, axis=0)

print(all_subsegments.shape)  # 输出形状

# test original the tokenizer
tokens = tokenizer(all_subsegments)
decoded_actions = tokenizer.decode(tokens)

# compute the difference between the original and the new decoded actions
diff = np.abs(all_subsegments - decoded_actions)
# mean difference
mean_diff = np.mean(diff)
print(mean_diff)

# train the tokenizer
tokenizer = tokenizer.fit(all_subsegments, scale = 50.0)
# save the tokenizer
tokenizer.save_pretrained("/share/project/yuqi.wang/OmniSim/pretrain/fast_libero_raw_t5_s50")

# compute the difference between the original and the new decoded actions
tokens = tokenizer(all_subsegments)
# print average length of the tokens
print(np.mean([len(token) for token in tokens]))
print(np.max([len(token) for token in tokens]))
print(np.min([len(token) for token in tokens]))
decoded_actions = tokenizer.decode(tokens)
mean_diff = np.mean(np.abs(all_subsegments - decoded_actions))
print(mean_diff)




