#!/bin/bash

# GPT-5 Visual Progress Evaluation Processor Runner Script
# 使用方法: ./run_processor.sh

set -e  # 遇到错误立即退出

# ===========================
# 配置区域 - 根据需要修改
# ===========================

# OpenAI API密钥 - 可以从环境变量读取或直接设置
API_KEY="${OPENAI_API_KEY:-}"

# 默认路径配置
DEFAULT_INPUT="/home/vcj9002/jianshu/workspace/code/ProgressLM/data/train/sft/h5_tienkung_xsens_sft.jsonl"
DEFAULT_OUTPUT="/home/vcj9002/jianshu/Results/progressLM/cold-data/h5_tienkung_xsens_output_$(date +%Y%m%d_%H%M%S).jsonl"
DEFAULT_IMAGE_DIR="/home/vcj9002/jianshu/workspace/data/robomind/data/images"
DEFAULT_MODEL="gpt-5-mini"
DEFAULT_WORKERS=3

# 拓展功能默认配置
DEFAULT_LIMIT="5"        # 默认不限制数量，如需限制设置为数字，例如: DEFAULT_LIMIT=100
DEFAULT_RESUME=true    # 默认不启用断点续传，如需启用设置为: DEFAULT_RESUME=true
DEFAULT_NO_RETRY=false  # 默认重试失败样本，如不需要重试设置为: DEFAULT_NO_RETRY=true

# Python脚本名称
PROCESSOR_SCRIPT="gpt5_processor.py"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ===========================
# 函数定义
# ===========================

print_banner() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo "          GPT-5 Visual Progress Evaluation Processor       "
    echo "============================================================"
    echo -e "${NC}"
}

print_help() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -k, --api-key KEY      设置OpenAI API密钥 (必需)"
    echo "  -i, --input FILE       输入JSONL文件 (默认: $DEFAULT_INPUT)"
    echo "  -o, --output FILE      输出JSONL文件 (默认: output_时间戳.jsonl)"
    echo "  -d, --image-dir DIR    图像目录路径 (默认: $DEFAULT_IMAGE_DIR)"
    echo "  -m, --model MODEL      GPT-5模型版本 (默认: $DEFAULT_MODEL)"
    echo "                         可选: gpt-5, gpt-5-mini, gpt-5-nano"
    echo "  -w, --workers NUM      最大并发数 (默认: $DEFAULT_WORKERS)"
    echo "  -l, --limit NUM        限制处理的样本数量"
    echo "  -r, --resume           启用断点续传（从上次中断处继续）"
    echo "  -n, --no-retry         不重试失败的样本"
    echo "  -h, --help             显示此帮助信息"
    echo ""
    echo "环境变量:"
    echo "  OPENAI_API_KEY         可以通过环境变量设置API密钥"
    echo ""
    echo "示例:"
    echo "  # 基本使用"
    echo "  $0 -k sk-xxx -i data.jsonl -o results.jsonl -d ./images"
    echo "  "
    echo "  # 限制处理100个样本"
    echo "  $0 -k sk-xxx -i data.jsonl -l 100 -d ./images"
    echo "  "
    echo "  # 断点续传"
    echo "  $0 -k sk-xxx -i data.jsonl -o results.jsonl -r -d ./images"
    echo "  "
    echo "  # 使用环境变量"
    echo "  OPENAI_API_KEY=sk-xxx $0 -i data.jsonl"
}

check_requirements() {
    echo -e "${YELLOW}检查系统要求...${NC}"
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 未安装${NC}"
        exit 1
    fi
    
    # 检查Python版本
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}"
    
    # 检查处理器脚本
    if [ ! -f "$PROCESSOR_SCRIPT" ]; then
        echo -e "${RED}❌ 找不到处理器脚本: $PROCESSOR_SCRIPT${NC}"
        echo "请确保脚本在当前目录"
        exit 1
    fi
    echo -e "${GREEN}✅ 处理器脚本存在${NC}"
}

check_python_packages() {
    echo -e "${YELLOW}检查Python包...${NC}"
    
    MISSING_PACKAGES=""
    
    # 检查必需的包
    for package in openai tqdm; do
        if ! python3 -c "import $package" 2>/dev/null; then
            MISSING_PACKAGES="$MISSING_PACKAGES $package"
        else
            echo -e "${GREEN}✅ $package${NC}"
        fi
    done
    
    # 如果有缺失的包，提示安装
    if [ -n "$MISSING_PACKAGES" ]; then
        echo -e "${RED}缺少必需的Python包:$MISSING_PACKAGES${NC}"
        echo -e "${YELLOW}是否自动安装？ (y/n)${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            pip install openai tqdm pandas openpyxl
            echo -e "${GREEN}✅ 包安装完成${NC}"
        else
            echo -e "${RED}请手动安装: pip install openai tqdm${NC}"
            exit 1
        fi
    fi
}

validate_inputs() {
    # 检查API密钥
    if [ -z "$API_KEY" ]; then
        echo -e "${RED}❌ 错误: 未提供API密钥${NC}"
        echo "请使用 -k 选项或设置 OPENAI_API_KEY 环境变量"
        exit 1
    fi
    
    # 检查输入文件
    if [ ! -f "$INPUT_FILE" ]; then
        echo -e "${RED}❌ 错误: 输入文件不存在: $INPUT_FILE${NC}"
        exit 1
    fi
    
    # 检查图像目录
    if [ ! -d "$IMAGE_DIR" ]; then
        echo -e "${RED}❌ 错误: 图像目录不存在: $IMAGE_DIR${NC}"
        exit 1
    fi
    
    # 统计输入文件行数
    LINE_COUNT=$(wc -l < "$INPUT_FILE")
    echo -e "${GREEN}✅ 输入文件包含 $LINE_COUNT 个样本${NC}"
    
    # 如果是续传模式，检查输出文件
    if [ "$RESUME" = true ] && [ -f "$OUTPUT_FILE" ]; then
        PROCESSED_COUNT=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
        echo -e "${GREEN}✅ 发现已处理文件，包含 $PROCESSED_COUNT 条记录${NC}"
        echo -e "${YELLOW}📌 将从断点继续处理${NC}"
    elif [ "$RESUME" = true ] && [ ! -f "$OUTPUT_FILE" ]; then
        echo -e "${YELLOW}⚠️  续传模式但未找到输出文件，将从头开始${NC}"
    fi
    
    # 如果设置了限制，显示信息
    if [ -n "$LIMIT" ]; then
        echo -e "${BLUE}📌 将处理最多 $LIMIT 个样本${NC}"
    fi
}

run_processor() {
    echo -e "${BLUE}开始处理...${NC}"
    echo "================================"
    
    # 构建命令
    CMD="python3 $PROCESSOR_SCRIPT"
    CMD="$CMD --api-key \"$API_KEY\""
    CMD="$CMD --input \"$INPUT_FILE\""
    CMD="$CMD --output \"$OUTPUT_FILE\""
    CMD="$CMD --image-dir \"$IMAGE_DIR\""
    CMD="$CMD --model $MODEL"
    CMD="$CMD --max-workers $MAX_WORKERS"
    
    if [ -n "$LIMIT" ]; then
        CMD="$CMD --limit $LIMIT"
    fi
    
    if [ "$RESUME" = true ]; then
        CMD="$CMD --resume"
    fi
    
    if [ "$NO_RETRY" = true ]; then
        CMD="$CMD --no-retry"
    fi
    
    # 执行命令
    eval $CMD
    RESULT=$?
    
    # 根据退出码显示结果
    if [ $RESULT -eq 0 ]; then
        echo -e "${GREEN}✨ 处理完成！所有样本成功处理${NC}"
    elif [ $RESULT -eq 1 ]; then
        echo -e "${YELLOW}⚠️ 处理完成，但部分样本失败${NC}"
    elif [ $RESULT -eq 130 ]; then
        echo -e "${YELLOW}⚠️ 用户中断处理${NC}"
    else
        echo -e "${RED}❌ 处理失败${NC}"
    fi
    
    return $RESULT
}

show_results_summary() {
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo -e "${BLUE}结果摘要:${NC}"
        echo "================================"
        
        # 统计成功和失败
        SUCCESS_COUNT=$(grep -c '"status": "success"' "$OUTPUT_FILE" || true)
        ERROR_COUNT=$(grep -c '"status": "error"' "$OUTPUT_FILE" || true)
        
        echo -e "成功: ${GREEN}$SUCCESS_COUNT${NC}"
        echo -e "失败: ${RED}$ERROR_COUNT${NC}"
        
        # 显示最近的错误（如果有）
        if [ $ERROR_COUNT -gt 0 ]; then
            echo ""
            echo -e "${YELLOW}最近的错误:${NC}"
            grep '"status": "error"' "$OUTPUT_FILE" | tail -3 | while IFS= read -r line; do
                ERROR_ID=$(echo "$line" | python3 -c "import json, sys; data=json.loads(sys.stdin.read()); print(data['meta_data'].get('id', 'unknown'))" 2>/dev/null || echo "unknown")
                ERROR_MSG=$(echo "$line" | python3 -c "import json, sys; data=json.loads(sys.stdin.read()); print(data['meta_data'].get('error', 'No error message')[:100])" 2>/dev/null || echo "Parse error")
                echo "  - $ERROR_ID: $ERROR_MSG..."
            done
        fi
        
        echo ""
        echo -e "${GREEN}输出文件: $OUTPUT_FILE${NC}"
    fi
}

# ===========================
# 主程序
# ===========================

# 初始化变量
INPUT_FILE="$DEFAULT_INPUT"
OUTPUT_FILE="$DEFAULT_OUTPUT"
IMAGE_DIR="$DEFAULT_IMAGE_DIR"
MODEL="$DEFAULT_MODEL"
MAX_WORKERS="$DEFAULT_WORKERS"
NO_RETRY="$DEFAULT_NO_RETRY"
LIMIT="$DEFAULT_LIMIT"
RESUME="$DEFAULT_RESUME"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -d|--image-dir)
            IMAGE_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -w|--workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME=true
            shift
            ;;
        -n|--no-retry)
            NO_RETRY=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

# 主流程
print_banner

# 如果没有参数，显示帮助
if [ -z "$API_KEY" ] && [ "$INPUT_FILE" = "$DEFAULT_INPUT" ]; then
    print_help
    exit 0
fi

# 执行检查和处理
check_requirements
check_python_packages
validate_inputs

echo ""
echo -e "${BLUE}配置信息:${NC}"
echo "================================"
echo "输入文件:   $INPUT_FILE"
echo "输出文件:   $OUTPUT_FILE"
echo "图像目录:   $IMAGE_DIR"
echo "模型:       $MODEL"
echo "并发数:     $MAX_WORKERS"
if [ -n "$LIMIT" ]; then
    echo "处理限制:   $LIMIT 个样本"
fi
echo "断点续传:   $([ "$RESUME" = true ] && echo '是' || echo '否')"
echo "重试失败:   $([ "$NO_RETRY" = true ] && echo '否' || echo '是')"
echo "================================"
echo ""

# 确认执行
echo -e "${YELLOW}是否开始处理？ (y/n)${NC}"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 记录开始时间
START_TIME=$(date +%s)

# 运行处理器
run_processor

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# 显示结果摘要
show_results_summary

# 显示总耗时
echo ""
echo -e "${BLUE}总耗时: $(($ELAPSED / 60)) 分 $(($ELAPSED % 60)) 秒${NC}"

# 提示下一步
echo ""
echo -e "${GREEN}处理完成！${NC}"
echo "您可以使用以下命令查看结果:"
echo "  cat $OUTPUT_FILE | python3 -m json.tool | less"
echo ""
echo "或者使用jq格式化查看:"
echo "  cat $OUTPUT_FILE | jq '.'"
echo ""
echo "查看特定字段:"
echo "  cat $OUTPUT_FILE | jq '{ref: .ref, score: .score, gt: .ground_truth_score}'"

exit 0