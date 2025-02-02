#!/bin/bash

# 生成工作负载
python platforms/generate_jobs.py

sleep 1
# 生成集群配置
python platforms/generate_cluster_xml.py

# 启动BATSIM（带能耗监控）
batsim --platform data/cluster.xml \
       --workload data/jobs.json \
       --energy \
       --export data/result/out \
       --verbosity information

# 在另一个终端中运行调度器
# python schedulers/basic_scheduler.py
