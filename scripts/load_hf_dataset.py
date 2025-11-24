#!/usr/bin/env python3
"""
从 HuggingFace 加载 dataset 并准备用于 tokenizer 训练
支持流式加载、数据采样、多进程处理等功能
"""

import os
import argparse
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any
from tqdm import tqdm
import random

try:
    from datasets import load_dataset, Dataset, IterableDataset
except ImportError:
    print("错误: 需要安装 datasets 库")
    print("请运行: pip install datasets")
    exit(1)


class HuggingFaceDatasetLoader:
    """HuggingFace Dataset 加载器"""

    def __init__(
        self,
        dataset_name: str,
        config_name: Optional[str] = None,
        split: str = "train",
        text_field: str = "text",
        streaming: bool = True,
        use_auth_token: Optional[str] = None,
    ):
        """
        初始化加载器

        Args:
            dataset_name: HuggingFace dataset 名称（如 "wikitext", "openwebtext"）
            config_name: 配置名称（如 "wikitext-103-raw-v1"）
            split: 数据集分割（train/test/validation）
            text_field: 文本字段名
            streaming: 是否使用流式加载（节省内存）
            use_auth_token: HuggingFace 认证 token（私有数据集需要）
        """
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.text_field = text_field
        self.streaming = streaming
        self.use_auth_token = use_auth_token
        self.dataset = None

    def load(self):
        """加载数据集"""
        print(f"加载数据集: {self.dataset_name}")
        if self.config_name:
            print(f"配置: {self.config_name}")
        print(f"分割: {self.split}")
        print(f"流式模式: {self.streaming}")

        try:
            self.dataset = load_dataset(
                self.dataset_name,
                self.config_name,
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
            if self.text_field in sample:
                text_preview = sample[self.text_field][:200]
                print(f"{self.text_field}: {text_preview}...")

        except Exception as e:
            print(f"✗ 加载失败: {e}")
            raise

    def get_text_iterator(
        self,
        max_samples: Optional[int] = None,
        sample_rate: float = 1.0,
        min_length: int = 10,
        max_length: Optional[int] = None,
    ) -> Iterator[str]:
        """
        获取文本迭代器

        Args:
            max_samples: 最大样本数（None 表示全部）
            sample_rate: 抽样比例（0.0-1.0）
            min_length: 最小文本长度（字符数）
            max_length: 最大文本长度（None 表示不限制）

        Yields:
            文本内容
        """
        if self.dataset is None:
            raise ValueError("请先调用 load() 方法加载数据集")

        count = 0
        total = max_samples if max_samples else "所有"

        print(f"\n开始提取文本...")
        print(f"目标样本数: {total}")
        print(f"抽样比例: {sample_rate*100:.1f}%")
        print(f"最小长度: {min_length} 字符")
        if max_length:
            print(f"最大长度: {max_length} 字符")

        for item in tqdm(self.dataset, desc="处理数据"):
            # 检查是否达到最大样本数
            if max_samples and count >= max_samples:
                break

            # 抽样
            if sample_rate < 1.0 and random.random() > sample_rate:
                continue

            # 提取文本
            text = item.get(self.text_field, "")
            if not text or not isinstance(text, str):
                continue

            text = text.strip()

            # 过滤长度
            if len(text) < min_length:
                continue
            if max_length and len(text) > max_length:
                text = text[:max_length]

            yield text
            count += 1

        print(f"\n✓ 成功提取 {count} 条文本")

    def save_to_file(
        self,
        output_path: str,
        max_samples: Optional[int] = None,
        sample_rate: float = 1.0,
        min_length: int = 10,
        max_length: Optional[int] = None,
    ):
        """
        保存文本到文件

        Args:
            output_path: 输出文件路径
            max_samples: 最大样本数
            sample_rate: 抽样比例
            min_length: 最小文本长度
            max_length: 最大文本长度
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n保存到文件: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for text in self.get_text_iterator(
                max_samples=max_samples,
                sample_rate=sample_rate,
                min_length=min_length,
                max_length=max_length,
            ):
                f.write(text + '\n')

        # 显示文件信息
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ 文件已保存")
        print(f"文件大小: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='从 HuggingFace 加载 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 加载 wikitext 数据集
  python load_hf_dataset.py -d wikitext -c wikitext-103-raw-v1 -o data/wikitext.txt

  # 加载 OpenWebText（流式）
  python load_hf_dataset.py -d openwebtext --streaming -o data/openwebtext.txt

  # 限制样本数和抽样
  python load_hf_dataset.py -d wikitext -c wikitext-103-raw-v1 \\
    --max-samples 100000 --sample-rate 0.5 -o data/wikitext_sampled.txt

  # 从 JSON/Parquet 文件加载
  python load_hf_dataset.py -d json --data-files data.json --text-field content -o output.txt

常用数据集:
  - wikitext (配置: wikitext-103-raw-v1, wikitext-2-raw-v1)
  - openwebtext
  - bookcorpus
  - wikipedia (配置: 20220301.en)
  - c4 (配置: en, realnewslike)
  - the_pile
        """
    )

    # 数据集参数
    parser.add_argument('--dataset', '-d', required=True,
                       help='HuggingFace 数据集名称')
    parser.add_argument('--config', '-c', default=None,
                       help='数据集配置名称')
    parser.add_argument('--split', default='train',
                       help='数据集分割（默认: train）')
    parser.add_argument('--text-field', default='text',
                       help='文本字段名（默认: text）')

    # 加载选项
    parser.add_argument('--streaming', action='store_true',
                       help='使用流式加载（节省内存）')
    parser.add_argument('--token', default=None,
                       help='HuggingFace 认证 token（私有数据集需要）')
    parser.add_argument('--data-files', default=None,
                       help='本地数据文件路径（用于 json/csv/parquet 等格式）')

    # 输出参数
    parser.add_argument('--output', '-o', required=True,
                       help='输出文件路径')

    # 过滤参数
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大样本数（None 表示全部）')
    parser.add_argument('--sample-rate', type=float, default=1.0,
                       help='抽样比例（0.0-1.0，默认: 1.0）')
    parser.add_argument('--min-length', type=int, default=10,
                       help='最小文本长度（字符数，默认: 10）')
    parser.add_argument('--max-length', type=int, default=None,
                       help='最大文本长度（字符数，None 表示不限制）')

    args = parser.parse_args()

    # 验证参数
    if not 0.0 < args.sample_rate <= 1.0:
        raise ValueError("sample-rate 必须在 (0.0, 1.0] 区间内")

    # 处理数据文件参数
    dataset_kwargs = {}
    if args.data_files:
        dataset_kwargs['data_files'] = args.data_files

    # 创建加载器
    print("="*60)
    print("HuggingFace Dataset 加载器")
    print("="*60)

    loader = HuggingFaceDatasetLoader(
        dataset_name=args.dataset,
        config_name=args.config,
        split=args.split,
        text_field=args.text_field,
        streaming=args.streaming,
        use_auth_token=args.token,
    )

    # 加载数据集
    loader.load()

    # 保存到文件
    loader.save_to_file(
        output_path=args.output,
        max_samples=args.max_samples,
        sample_rate=args.sample_rate,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    print("\n" + "="*60)
    print("✓ 完成！")
    print("="*60)
    print(f"\n可以使用以下命令训练 tokenizer:")
    print(f"python scripts/train_tokenizer.py -i {args.output} -o output/tokenizer")


if __name__ == '__main__':
    main()
