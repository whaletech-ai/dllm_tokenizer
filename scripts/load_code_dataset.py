#!/usr/bin/env python3
"""
从 HuggingFace 加载代码数据集（Parquet 格式）
数据结构: {"content": chunk, "length": chunk_length}
以指定采样率提取 content 字段，生成代码语料库
"""

import os
import argparse
from pathlib import Path
from typing import Iterator, Optional
from tqdm import tqdm
import random

try:
    from datasets import load_dataset
except ImportError:
    print("错误: 需要安装 datasets 库")
    print("请运行: pip install datasets")
    exit(1)


class CodeDatasetLoader:
    """代码数据集加载器（Parquet 格式）"""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        streaming: bool = True,
        use_auth_token: Optional[str] = None,
    ):
        """
        初始化加载器

        Args:
            dataset_name: HuggingFace dataset 名称
            split: 数据集分割（train/test/validation）
            streaming: 是否使用流式加载（节省内存）
            use_auth_token: HuggingFace 认证 token（私有数据集需要）
        """
        self.dataset_name = dataset_name
        self.split = split
        self.streaming = streaming
        self.use_auth_token = use_auth_token
        self.dataset = None

    def load(self):
        """加载数据集"""
        print(f"正在加载数据集: {self.dataset_name}")
        print(f"分割: {self.split}")
        print(f"流式模式: {self.streaming}")
        print(f"格式: Parquet")

        try:
            self.dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=self.streaming,
                token=self.use_auth_token,
            )
            print("✓ 数据集加载成功")

            # 显示数据集信息
            if not self.streaming:
                print(f"数据集大小: {len(self.dataset)} 条")

            # 显示示例
            print("\n数据示例:")
            sample = next(iter(self.dataset))
            print(f"可用字段: {list(sample.keys())}")

            if "content" in sample:
                content_preview = sample["content"][:200] if len(sample["content"]) > 200 else sample["content"]
                print(f"\ncontent 预览:\n{content_preview}...")

            if "length" in sample:
                print(f"length: {sample['length']}")

        except Exception as e:
            print(f"✗ 加载失败: {e}")
            print("\n提示:")
            print("  - 确认数据集名称正确")
            print("  - 如果是私有数据集，请使用 --token 参数提供认证")
            print("  - 检查网络连接")
            raise

    def extract_code_corpus(
        self,
        output_path: str,
        sample_rate: float = 0.15,
        min_length: int = 10,
        max_length: Optional[int] = None,
        max_samples: Optional[int] = None,
    ):
        """
        提取代码语料库并保存到文件

        Args:
            output_path: 输出文件路径
            sample_rate: 采样概率（默认: 0.15）
            min_length: 最小代码长度（字符数，默认: 10）
            max_length: 最大代码长度（None 表示不限制）
            max_samples: 最大样本数（None 表示全部）
        """
        if self.dataset is None:
            raise ValueError("请先调用 load() 方法加载数据集")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print("开始提取代码语料库")
        print(f"{'='*60}")
        print(f"输出文件: {output_path}")
        print(f"采样概率: {sample_rate*100:.1f}%")
        print(f"最小长度: {min_length} 字符")
        if max_length:
            print(f"最大长度: {max_length} 字符")
        if max_samples:
            print(f"最大样本数: {max_samples}")
        print()

        sampled_count = 0
        total_count = 0
        total_chars = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in tqdm(self.dataset, desc="处理数据"):
                # 检查是否达到最大样本数
                if max_samples and sampled_count >= max_samples:
                    break

                total_count += 1

                # 随机采样
                if random.random() > sample_rate:
                    continue

                # 提取 content 字段
                content = item.get("content", "")
                if not content or not isinstance(content, str):
                    continue

                content = content.strip()

                # 长度过滤
                content_length = len(content)
                if content_length < min_length:
                    continue
                if max_length and content_length > max_length:
                    content = content[:max_length]
                    content_length = max_length

                # 写入文件
                f.write(content + '\n')
                sampled_count += 1
                total_chars += content_length

        # 显示统计信息
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        actual_sample_rate = (sampled_count / total_count * 100) if total_count > 0 else 0
        avg_length = total_chars / sampled_count if sampled_count > 0 else 0

        print(f"\n{'='*60}")
        print("✓ 提取完成")
        print(f"{'='*60}")
        print(f"总处理数: {total_count:,} 条")
        print(f"采样数: {sampled_count:,} 条")
        print(f"实际采样率: {actual_sample_rate:.2f}%")
        print(f"总字符数: {total_chars:,}")
        print(f"平均长度: {avg_length:.0f} 字符")
        print(f"文件大小: {file_size:.2f} MB")
        print(f"文件路径: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='从 HuggingFace 加载代码数据集（Parquet 格式）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法（默认采样率 0.15）
  python load_code_dataset.py -d yizhang4799/sedd_pretrain_data -o data/code_corpus.txt

  # 指定采样率
  python load_code_dataset.py -d yizhang4799/sedd_pretrain_data -o data/code_corpus.txt --sample-rate 0.15

  # 使用流式加载（推荐，节省内存）
  python load_code_dataset.py -d yizhang4799/sedd_pretrain_data -o data/code_corpus.txt --streaming

  # 限制最大样本数
  python load_code_dataset.py -d yizhang4799/sedd_pretrain_data -o data/code_corpus.txt --max-samples 100000

  # 私有数据集（需要 token）
  python load_code_dataset.py -d yizhang4799/sedd_pretrain_data -o data/code_corpus.txt --token YOUR_HF_TOKEN

  # 完整示例
  python load_code_dataset.py \\
    -d yizhang4799/sedd_pretrain_data \\
    -o data/code_corpus.txt \\
    --sample-rate 0.15 \\
    --streaming \\
    --min-length 50 \\
    --max-length 10000

数据格式要求:
  - Parquet 文件
  - 包含 "content" 字段（代码内容）
  - 包含 "length" 字段（代码长度，可选）
        """
    )

    # 数据集参数
    parser.add_argument('--dataset', '-d', required=True,
                       help='HuggingFace 数据集名称')
    parser.add_argument('--split', default='train',
                       help='数据集分割（默认: train）')
    parser.add_argument('--streaming', action='store_true',
                       help='使用流式加载（推荐，节省内存）')
    parser.add_argument('--token', default=None,
                       help='HuggingFace 认证 token（私有数据集需要）')

    # 输出参数
    parser.add_argument('--output', '-o', default='data/code_corpus.txt',
                       help='输出文件路径（默认: data/code_corpus.txt）')

    # 采样参数
    parser.add_argument('--sample-rate', type=float, default=0.15,
                       help='采样概率（0.0-1.0，默认: 0.15）')
    parser.add_argument('--min-length', type=int, default=10,
                       help='最小代码长度（字符数，默认: 10）')
    parser.add_argument('--max-length', type=int, default=None,
                       help='最大代码长度（字符数，None 表示不限制）')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大采样数（None 表示全部）')

    args = parser.parse_args()

    # 验证参数
    if not 0.0 < args.sample_rate <= 1.0:
        raise ValueError("sample-rate 必须在 (0.0, 1.0] 区间内")

    # 创建加载器
    print("="*60)
    print("代码数据集加载器")
    print("="*60)
    print()

    loader = CodeDatasetLoader(
        dataset_name=args.dataset,
        split=args.split,
        streaming=args.streaming,
        use_auth_token=args.token,
    )

    # 加载数据集
    loader.load()

    # 提取代码语料库
    loader.extract_code_corpus(
        output_path=args.output,
        sample_rate=args.sample_rate,
        min_length=args.min_length,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )

    print(f"\n{'='*60}")
    print("下一步：训练 Tokenizer")
    print(f"{'='*60}")
    print(f"\n可以使用以下命令训练 tokenizer:")
    print(f"\npython scripts/train_tokenizer.py \\")
    print(f"  -i {args.output} \\")
    print(f"  -o output/tokenizer \\")
    print(f"  --vocab-size 50000 \\")
    print(f"  --use-iterator \\")
    print(f"  --sample-rate 1.0")
    print()


if __name__ == '__main__':
    main()
