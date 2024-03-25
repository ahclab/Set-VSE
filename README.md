# Set-VSE
集合型視覚言語埋め込み
※本リポジトリは動作未チェックです。うまく動かない場合はIssueを立ててください。

# 使い方

## インストール
Dockerの環境を構築
```
git clone https://github.com/ahclab/Set-VSE.git
cd Set-VSE/env
bash build.sh
```

## Dockerの起動
```
bash run_docker.sh
```

## 実験の実行

- `experiment_VSE.py`: CLIPによる従来のVSEの実験を実行
- `experiment_set_of_VSE.py`: Set-VSEの実験を実行
- `experiment_set_of_VSE_complete_sentence.py`: オリジナルの説明文を文単位で分割したSet-VSEの実験を実行

## 実行例

`run_NL_experiments.sh`を参照

### オプションの説明

- `--IPOT`：最適輸送を使う 
- `--img_type`：画像の条件
  - `global`：大域埋め込み
  - `partial`： 部分埋め込み
  - `hybrid`：大域＋部分埋め込み
- `--text_type`：テキストの条件
  - `global`：大域埋め込み
  - `partial`：部分埋め込み
  - `hybrid`：大域＋部分埋め込み


### 書誌情報
```bibtex
品川政太朗, 中村哲. 集合型視覚言語埋め込み.  
第259回自然言語処理研究発表会, 2024-NL-259, 2024.  
```
```bibtex
@misc{shinagawa2024setvse,
  title={集合型視覚言語埋め込み},
  author={品川政太朗 and 中村哲},
  booktitle={第259回自然言語処理研究発表会},
  pages={2024-NL-259},
  year={2024}
}
```