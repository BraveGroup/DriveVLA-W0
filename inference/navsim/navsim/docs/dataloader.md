# filter scenes

1. 获取 navsim_logs/trainval 目录中的所有 logs，这些 logs 都是 pickle 文件。
2. 根据 scene_filter 中存在的 log_names 筛选出来其中的有效 logs。
3. 根据 scene_filter 中存在的 tokens 筛选出有效的 tokens。
4. 遍历每一个 log（pickle 文件），读取其中存储的 scene dicts，然后根据 num_frames 和 frame_interval 将这 198 个 scenes 分解成多个长度为 14 的 scene list，并筛选掉长度不满足要求的 scene list。上面的 token 存储的是最近一个历史帧的 token，如果这个 token 不是需要的，也不会被存储。
5. 最终返回的是一个字典，每个 token 都是最近一个历史帧的 scene token。