<div align="center">

# RAVEdit-NFT

### Joint Audio-Visual Editing with Role-Aware Cross-Modal Attention and Negative-Aware Fine-Tuning

**Ye Tao¹² · Shuiyang Mao² · Wei Liu² · Shuai Wang¹**

¹ Nanjing University &nbsp; ² Video Rebirth

[Project Page](https://ty0402.github.io/ravediting/)

</div>

RAVEdit jointly edits video and audio from a source video and a text instruction.
Built on LTX-2.5, it combines role-aware cross-modal attention with negative-aware fine-tuning.

## Model Weights

**Coming soon.**

## Installation

Requires Linux, an NVIDIA GPU, Python 3.11, and FFmpeg.
Run the following commands from the repository root:

```bash
conda create -n ravedit python=3.11 -y
conda activate ravedit
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Usage

Download the full dev model components from [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)
and set their paths and your RAVEdit checkpoint path in [configs/inference.yaml](configs/inference.yaml).

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config configs/inference.yaml \
  --source /path/to/source.mp4 \
  --prompt 'Replace the dialogue with: "This is the truth.", keeping the person and background unchanged.' \
  --output-dir outputs/example
```

Source audio is read from the input MP4. The edited video with generated audio is saved to `outputs/example/videos/edited.mp4`.

For batch inference, use the CSV format in [examples/inputs.csv](examples/inputs.csv):

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config configs/inference.yaml \
  --input-csv examples/inputs.csv \
  --output-dir outputs/batch
```

## Citation

```bibtex
@misc{tao2026raveditnft,
  title  = {RAVEdit-NFT: Joint Audio-Visual Editing with Role-Aware Cross-Modal Attention and Negative-Aware Fine-Tuning},
  author = {Tao, Ye and Mao, Shuiyang and Liu, Wei and Wang, Shuai},
  year   = {2026},
  url    = {https://ty0402.github.io/ravediting/}
}
```

## Acknowledgements

Built on [LTX-2](https://github.com/Lightricks/LTX-2). See [LICENSE.md](LICENSE.md) for licensing.
