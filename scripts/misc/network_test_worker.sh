#!/bin/bash

# 多机网络通信速度测试 - Worker 节点
# 测试PyTorch分布式训练中的NCCL通信性能

echo "=== 多机网络通信速度测试 - Worker 节点 ==="

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <master_ip> <worker_rank>"
    echo "例如: $0 192.168.1.100 1"
    echo "     $0 192.168.1.100 2"
    exit 1
fi

MASTER_ADDR=$1
WORKER_RANK=$2

# 配置参数
WORLD_SIZE=3  # 总共3个节点：1个master + 2个worker
RANK=$WORKER_RANK  # worker节点rank=1或2
MASTER_PORT=29500
NGPUS=8       # 每个节点的GPU数量，根据实际情况调整

# 获取本机IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "Worker IP: $LOCAL_IP"
echo "Master IP: $MASTER_ADDR"
echo "World size: $WORLD_SIZE, Worker Rank: $RANK"
echo "Port: $MASTER_PORT, GPUs per node: $NGPUS"

# 检查master节点连通性
echo ""
echo "=== 检查网络连通性 ==="
echo "测试连接到master节点 $MASTER_ADDR..."
if ping -c 3 "$MASTER_ADDR" > /dev/null 2>&1; then
    echo "✓ Master节点连接正常"
    
    # 测试带宽
    echo "测试到master的网络带宽..."
    iperf3 -c "$MASTER_ADDR" -t 5 -P 2 2>/dev/null || echo "  iperf3 测试失败（可能未安装iperf3）"
else
    echo "✗ Master节点连接失败"
    exit 1
fi

# 启动iperf3服务器（用于master测试）
echo "启动iperf3服务器用于带宽测试..."
iperf3 -s -D 2>/dev/null || echo "iperf3服务器启动失败"

# NCCL优化配置 - 以太网环境
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO

# 以太网优化
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
export NCCL_P2P_DISABLE=1

# 缓冲区优化
export NCCL_BUFFSIZE=33554432  # 32MB
export NCCL_NTHREADS=16
export NCCL_MIN_NCHANNELS=8
export NCCL_MAX_NCHANNELS=16

# 超时配置
export NCCL_TIMEOUT=1800
export NCCL_OP_TIMEOUT=1800000
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

echo ""
echo "=== 环境检查 ==="
echo "CUDA版本: $(nvcc --version 2>/dev/null | grep release || echo 'CUDA未安装')"
echo "PyTorch版本: $(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo 'PyTorch未安装')"
echo "GPU数量: $(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo '0')"
echo "NCCL版本: $(python -c 'import torch; print(torch.cuda.nccl.version())' 2>/dev/null || echo '未知')"

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi 不可用"

# 创建测试脚本（与master相同）
cat > /tmp/network_test.py << 'EOF'
import torch
import torch.distributed as dist
import time
import os
import argparse

def test_communication():
    # 初始化分布式
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK'])
    )
    
    device = torch.cuda.current_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"节点 {rank}/{world_size} 初始化完成，GPU: {device}")
    
    # 测试不同大小的张量通信
    test_sizes = [
        (1024, "1KB"),
        (1024*1024, "1MB"), 
        (1024*1024*10, "10MB"),
        (1024*1024*100, "100MB"),
        (1024*1024*500, "500MB")
    ]
    
    results = []
    
    for size, desc in test_sizes:
        print(f"\n=== 测试 {desc} 数据传输 ===")
        
        # 创建测试张量
        tensor = torch.randn(size, device=device, dtype=torch.float32)
        
        # 预热
        for _ in range(3):
            dist.all_reduce(tensor)
        
        torch.cuda.synchronize()
        
        # 正式测试 - AllReduce
        num_iterations = 10
        start_time = time.time()
        
        for _ in range(num_iterations):
            dist.all_reduce(tensor)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        data_size_mb = size * 4 / (1024 * 1024)  # float32 = 4 bytes
        bandwidth = data_size_mb / avg_time
        
        result = {
            'size': desc,
            'avg_time': avg_time,
            'bandwidth': bandwidth
        }
        results.append(result)
        
        print(f"  平均时间: {avg_time:.4f}s")
        print(f"  带宽: {bandwidth:.2f} MB/s")
        
        # 测试点对点通信 (只在master节点)
        if rank == 0:
            print(f"  测试点对点通信...")
            for target_rank in range(1, world_size):
                # Send to target
                send_start = time.time()
                dist.send(tensor, dst=target_rank)
                
                # Receive from target  
                dist.recv(tensor, src=target_rank)
                torch.cuda.synchronize()
                send_end = time.time()
                
                p2p_time = send_end - send_start
                p2p_bandwidth = (data_size_mb * 2) / p2p_time  # 双向传输
                
                print(f"    节点0 <-> 节点{target_rank}: {p2p_time:.4f}s, {p2p_bandwidth:.2f} MB/s")
        
        else:
            # Worker节点响应点对点测试
            dist.recv(tensor, src=0)
            dist.send(tensor, dst=0)
    
    # 汇总结果
    if rank == 0:
        print(f"\n=== 通信测试总结 ===")
        print(f"参与节点数: {world_size}")
        print(f"NCCL后端: {dist.get_backend()}")
        print("\nAllReduce性能:")
        for result in results:
            print(f"  {result['size']:>6}: {result['avg_time']:>8.4f}s, {result['bandwidth']:>8.2f} MB/s")
    else:
        print(f"\n=== Worker {rank} 测试完成 ===")
        print("AllReduce性能 (本节点视角):")
        for result in results:
            print(f"  {result['size']:>6}: {result['avg_time']:>8.4f}s, {result['bandwidth']:>8.2f} MB/s")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    test_communication()
EOF

echo ""
echo "=== 开始网络通信测试 ==="
echo "Worker节点 $RANK 准备就绪，等待master节点启动测试..."

echo "启动通信测试..."

# 启动分布式测试
torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    /tmp/network_test.py

echo ""
echo "=== 测试完成 ==="

# 清理临时文件
rm -f /tmp/network_test.py

# 停止iperf3服务器
pkill iperf3 2>/dev/null

echo "Worker节点 $RANK 测试结束" 